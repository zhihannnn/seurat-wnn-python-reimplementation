"""
Seurat V4 WNN — Full Python pipeline
Reads R-exported raw counts, runs preprocessing → WNN → UMAP
WNN algorithm matches wnn_v4.py exactly.
"""

import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix
from scipy.io import mmread
import scanpy as sc
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.decomposition import PCA

# ═══════════════════════════════════════════════════════════════
# 1. Load R-exported raw counts
# ═══════════════════════════════════════════════════════════════
print("Loading raw data from R export ...")
rna_sp = mmread("F:/workplace/python/WNN/rna_raw_from_R.mtx").tocsr()  # (features, cells)
adt_sp = mmread("F:/workplace/python/WNN/adt_raw_from_R.mtx").tocsr()

rna_genes = pd.read_csv("F:/workplace/python/WNN/rna_genes_from_R.csv")["gene"].values
rna_cells = pd.read_csv("F:/workplace/python/WNN/rna_cells_from_R.csv")["cell"].values
adt_features = pd.read_csv("F:/workplace/python/WNN/adt_features_from_R.csv")["feature"].values

print(f"RNA: {rna_sp.shape}   ADT: {adt_sp.shape}   Cells: {rna_sp.shape[1]}")
assert rna_sp.shape[1] == adt_sp.shape[1], "Cell count mismatch"

N_FEAT, N_CELLS = rna_sp.shape

# ═══════════════════════════════════════════════════════════════
# 2. RNA preprocessing: LogNormalize → FindVariableFeatures → ScaleData → PCA
# ═══════════════════════════════════════════════════════════════
print("\n=== RNA Preprocessing ===")

# 2a. LogNormalize (per-cell) + compute per-gene mean & variance in batches
print("  LogNormalize ...")
lib_sizes = np.array(rna_sp.sum(axis=0)).ravel() + 1e-8

BATCH = 1000
rna_mean = np.zeros(N_FEAT, dtype=np.float32)
rna_var  = np.zeros(N_FEAT, dtype=np.float32)

for start in range(0, N_FEAT, BATCH):
    end = min(start + BATCH, N_FEAT)
    batch = rna_sp[start:end].toarray()
    batch_norm = np.log1p(batch / lib_sizes * 10000)
    rna_mean[start:end] = batch_norm.mean(axis=1)
    rna_var[start:end]  = batch_norm.var(axis=1, ddof=1)

print(f"  mean range: [{rna_mean.min():.4f}, {rna_mean.max():.4f}]")

# 2b. Use R's variable features directly
print("  Loading R variable features ...")
hv_genes = pd.read_csv("F:/workplace/python/WNN/rna_variable_features_R.csv")["feature"].values
# Map gene names to indices
gene_to_idx = {g: i for i, g in enumerate(rna_genes)}
hv_idx = np.array([gene_to_idx[g] for g in hv_genes if g in gene_to_idx])
hv_idx = np.sort(hv_idx)
N_HVF = len(hv_idx)
print(f"  Variable features: {N_HVF}")

# 2c. Extract normalized data for variable features
print("  Extracting normalized HV data ...")
rna_log_hv = np.zeros((N_CELLS, N_HVF), dtype=np.float32)
for i, gene in enumerate(hv_idx):
    counts = rna_sp[gene].toarray().ravel()
    rna_log_hv[:, i] = np.log1p(counts / lib_sizes * 10000)

# 2d. ScaleData
print("  ScaleData ...")
mean = rna_log_hv.mean(axis=0, keepdims=True)
sd = rna_log_hv.std(axis=0, ddof=1, keepdims=True)
sd[sd < 1e-10] = 1
rna_scaled = np.clip((rna_log_hv - mean) / sd, -10, 10)

# 2e. PCA
print("  RunPCA ...")
rna_pca = PCA(n_components=50)
rna_pca_emb = rna_pca.fit_transform(rna_scaled)
rna_stdev = np.sqrt(rna_pca.explained_variance_)
print(f"  embeddings: {rna_pca_emb.shape}  stdev[0:3]: {rna_stdev[:3].round(4)}")

# ═══════════════════════════════════════════════════════════════
# 3. ADT preprocessing: CLR → ScaleData → PCA
# ═══════════════════════════════════════════════════════════════
print("\n=== ADT Preprocessing ===")
adt_dense = adt_sp.toarray().T  # (cells, features) — ADT is tiny
adt_t = torch.FloatTensor(adt_dense)

# 3a. CLR (matches Seurat v4)
print("  CLR ...")
n_adt = adt_t.size(1)
log1p_adt = torch.log1p(adt_t)
g = torch.exp(log1p_adt.sum(dim=1, keepdim=True) / n_adt)
adt_clr = torch.log1p(adt_t / g).numpy()

# 3b. ScaleData
print("  ScaleData ...")
m = adt_clr.mean(axis=0, keepdims=True)
s = adt_clr.std(axis=0, ddof=1, keepdims=True)
s[s < 1e-10] = 1
adt_scaled = np.clip((adt_clr - m) / s, -10, 10)

# 3c. PCA
print("  RunPCA ...")
ADT_PCS = min(25, N_CELLS - 1)
adt_pca = PCA(n_components=ADT_PCS)
adt_pca_emb = adt_pca.fit_transform(adt_scaled)
adt_stdev = np.sqrt(adt_pca.explained_variance_)
print(f"  embeddings: {adt_pca_emb.shape}  stdev[0:3]: {adt_stdev[:3].round(4)}")

# ═══════════════════════════════════════════════════════════════
# 4. WNN (IDENTICAL to wnn_v4.py)
# ═══════════════════════════════════════════════════════════════
print("\n=== WNN ===")

N_RNA_PCS = min(30, rna_pca_emb.shape[1])
N_ADT_PCS = min(18, adt_pca_emb.shape[1])

rna_t = torch.tensor(rna_pca_emb[:, :N_RNA_PCS], dtype=torch.float32)
adt_t = torch.tensor(adt_pca_emb[:, :N_ADT_PCS], dtype=torch.float32)
N = rna_t.shape[0]

# L2-normalize (matches l2.norm = TRUE in Seurat)
rna_t = torch.nn.functional.normalize(rna_t, p=2, dim=1)
adt_t = torch.nn.functional.normalize(adt_t, p=2, dim=1)

K_RANGE, K_SIGMA, K_NN, EPS = 200, 20, 20, 1e-8

# 4a. Batched KNN (avoid O(N^2) memory)
print("  Computing KNN (batched) ...")

def batched_knn(data, k, batch_size=500):
    """Compute KNN indices and distances for each cell, processing in batches.
    Returns: knn_idx (N, k-1) — excludes self
             knn_dist (N, k-1)
    """
    N = data.shape[0]
    knn_idx = torch.zeros((N, k - 1), dtype=torch.long)
    knn_dist = torch.zeros((N, k - 1))
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        d_batch = torch.cdist(data[start:end], data)  # (batch, N)
        top_d, top_i = d_batch.topk(k, largest=False, dim=1)
        # Exclude self (distance 0)
        for j in range(end - start):
            ii = start + j
            mask = top_i[j] != ii
            sel_i = top_i[j][mask][:k-1]
            sel_d = top_d[j][mask][:k-1]
            knn_idx[ii, :len(sel_i)] = sel_i
            knn_dist[ii, :len(sel_i)] = sel_d
    return knn_idx, knn_dist

knn_rna, knn_dist_rna = batched_knn(rna_t, K_RANGE + 1)
knn_adt, knn_dist_adt = batched_knn(adt_t, K_RANGE + 1)

# Helper: compute distance from cell i to specific neighbors on the fly
def dist_to(rna_or_adt, i, nbrs):
    """Euclidean distance from cell i to list of neighbor indices."""
    return torch.norm(rna_or_adt[i] - rna_or_adt[nbrs], dim=1)

# 4b. SNN
print("  Building SNN ...")
def build_snn(knn_idx):
    rows, cols, vals = [], [], []
    for i in range(N):
        si = set(knn_idx[i].tolist())
        for j in knn_idx[i]:
            jj = j.item()
            sj = set(knn_idx[jj].tolist())
            inter = len(si & sj)
            val = inter / (len(si | sj) + EPS)
            if val > 0:
                rows.append(i); cols.append(jj); vals.append(val)
    idx = torch.tensor([rows, cols], dtype=torch.long)
    vv  = torch.tensor(vals, dtype=torch.float32)
    return torch.sparse_coo_tensor(idx, vv, (N, N)).coalesce()

snn_rna = build_snn(knn_rna)
snn_adt = build_snn(knn_adt)

# 4c. Sigma bandwidth
print("  Computing sigma ...")
def compute_sigma(knn_dist, knn_idx, snn, k_sig):
    sigma = torch.zeros(N)
    coo = snn.coalesce()
    idx = coo.indices()
    val = coo.values()
    for i in range(N):
        m = (idx[0] == i)
        if m.sum() == 0:
            sigma[i] = knn_dist[i, :k_sig].mean()
            continue
        c = idx[1, m]; v = val[m]
        s = torch.argsort(v)
        n_i = min(k_sig, v.size(0))
        use = v[s] <= v[s[n_i - 1]]
        sel = c[s][use]
        # Get distances to SNN neighbors from stored KNN distances
        d_sel = []
        for nb in sel:
            nb = nb.item()
            pos = (knn_idx[i] == nb).nonzero(as_tuple=True)
            if pos[0].numel() > 0:
                d_sel.append(knn_dist[i, pos[0][0]])
        if d_sel:
            d_sel = torch.stack(d_sel)
            sigma[i] = d_sel.sort(descending=True).values[:min(k_sig, len(d_sel))].mean()
        else:
            sigma[i] = knn_dist[i, :k_sig].mean()
    return sigma

sigma_rna = compute_sigma(knn_dist_rna, knn_rna, snn_rna, K_SIGMA)
sigma_adt = compute_sigma(knn_dist_adt, knn_adt, snn_adt, K_SIGMA)

# 4d. Within- and cross-modality prediction
rna_pred_self  = rna_t[knn_rna[:, :K_NN]].mean(dim=1)
rna_pred_cross = rna_t[knn_adt[:, :K_NN]].mean(dim=1)
adt_pred_self  = adt_t[knn_adt[:, :K_NN]].mean(dim=1)
adt_pred_cross = adt_t[knn_rna[:, :K_NN]].mean(dim=1)

d_rna_self  = torch.norm(rna_t - rna_pred_self,  dim=1)
d_rna_cross = torch.norm(rna_t - rna_pred_cross, dim=1)
d_adt_self  = torch.norm(adt_t - adt_pred_self,  dim=1)
d_adt_cross = torch.norm(adt_t - adt_pred_cross, dim=1)

dnn_rna = knn_dist_rna[:, 0]  # nearest neighbor distance
dnn_adt = knn_dist_adt[:, 0]

# 4e. Kernel & affinity (float64 to match R precision)
print("  Computing kernel & affinity ...")
bw_rna = torch.clamp(sigma_rna - dnn_rna, min=EPS).double()
bw_adt = torch.clamp(sigma_adt - dnn_adt, min=EPS).double()

sim_rna_self  = torch.exp(-(d_rna_self  - dnn_rna).clamp(min=0).double() / bw_rna).float()
sim_rna_cross = torch.exp(-(d_rna_cross - dnn_rna).clamp(min=0).double() / bw_rna).float()
sim_adt_self  = torch.exp(-(d_adt_self  - dnn_adt).clamp(min=0).double() / bw_adt).float()
sim_adt_cross = torch.exp(-(d_adt_cross - dnn_adt).clamp(min=0).double() / bw_adt).float()

aff_rna = sim_rna_self / (sim_rna_cross + EPS)
aff_adt = sim_adt_self / (sim_adt_cross + EPS)

# 4f. Modality weights (stable softmax)
aff_max = torch.maximum(aff_rna, aff_adt)
w_rna = torch.exp(aff_rna - aff_max) / (torch.exp(aff_rna - aff_max) + torch.exp(aff_adt - aff_max))
w_adt = 1.0 - w_rna

# 4g. WNN graph
print("  Building WNN graph ...")
wnn_knn = torch.zeros((N, K_NN), dtype=torch.long)
for i in range(N):
    nbrs = torch.cat([knn_rna[i], knn_adt[i]]).unique()
    dw = w_rna[i] * dist_to(rna_t, i, nbrs) + w_adt[i] * dist_to(adt_t, i, nbrs)
    if nbrs.size(0) <= K_NN:
        wnn_knn[i, :nbrs.size(0)] = nbrs
    else:
        wnn_knn[i] = nbrs[dw.topk(K_NN, largest=False).indices]

# 4h. WNN-SNN
print("  Building WNN-SNN ...")
rows, cols, vals = [], [], []
for i in range(N):
    si = set(wnn_knn[i].tolist())
    for j in wnn_knn[i]:
        jj = j.item()
        sj = set(wnn_knn[jj].tolist())
        inter = len(si & sj)
        val = inter / (len(si | sj) + EPS)
        if val > 0:
            rows.append(i); cols.append(jj); vals.append(val)
idx = torch.tensor([rows, cols], dtype=torch.long)
vv  = torch.tensor(vals, dtype=torch.float32)
snn_wnn = torch.sparse_coo_tensor(idx, vv, (N, N)).coalesce()

# ═══════════════════════════════════════════════════════════════
# 5. Compare with R outputs
# ═══════════════════════════════════════════════════════════════
print("\n=== Comparison with R ===")
OUT = "F:/workplace/python/WNN/"

def compare(name, py_val, r_file):
    try:
        r_val = pd.read_csv(OUT + r_file, index_col=0).values.astype(np.float64)
        if r_val.shape[1] == 1:
            r_val = r_val.ravel()
        p_val = py_val.cpu().numpy()
        if p_val.ndim == 2 and p_val.shape[1] == 1:
            p_val = p_val.ravel()
        if r_val.shape != p_val.shape:
            print(f"  SKIP {name:20s} shape mismatch py={p_val.shape} R={r_val.shape}")
            return
        d = np.abs(p_val - r_val)
        corr = np.corrcoef(p_val.ravel(), r_val.ravel())[0, 1]
        ok = "PASS" if d.max() < 0.01 else ("~OK" if d.max() < 0.1 else "DIFF")
        print(f"  {ok:4s} {name:20s} corr={corr:.4f}  max={d.max():.6f}  mean={d.mean():.6f}")
    except FileNotFoundError:
        print(f"  SKIP {name:20s} R file not found")

compare("rna.weight",    w_rna,     "w_rna_manual_R.csv")
compare("adt.weight",    w_adt,     "w_adt_manual_R.csv")
compare("sigma_rna",     sigma_rna, "sigma_rna_R.csv")
compare("sigma_adt",     sigma_adt, "sigma_adt_R.csv")
compare("sim_rna_self",  sim_rna_self,  "sim_rna_self_R.csv")
compare("sim_rna_cross", sim_rna_cross, "sim_rna_cross_R.csv")
compare("sim_adt_self",  sim_adt_self,  "sim_adt_self_R.csv")
compare("sim_adt_cross", sim_adt_cross, "sim_adt_cross_R.csv")
compare("aff_rna",       aff_rna,       "aff_rna_R.csv")
compare("aff_adt",       aff_adt,       "aff_adt_R.csv")

# Save WNN outputs for quick UMAP rerun
print("  Saving WNN outputs ...")
torch.save({"snn_wnn": snn_wnn, "w_rna": w_rna, "K_NN": K_NN},
           "F:/workplace/python/WNN/wnn_outputs.pt")

# ═══════════════════════════════════════════════════════════════
# 6. UMAP
# ═══════════════════════════════════════════════════════════════
print("\n=== UMAP ===")
coo = snn_wnn.coalesce()
idx_s = coo.indices().cpu().numpy()
val_s = coo.values().cpu().numpy()
snn_csr = csr_matrix((val_s, (idx_s[0], idx_s[1])), shape=(N, N))

# Ensure no isolated nodes — add self as neighbor with min weight
degrees = np.array(snn_csr.sum(axis=1)).ravel()
isolated = np.where(degrees == 0)[0]
print(f"  Isolated nodes: {len(isolated)}")
if len(isolated) > 0:
    # Add self-loops for isolated nodes
    snn_csr = snn_csr.tolil()
    for node in isolated:
        snn_csr[node, node] = 0.01
    snn_csr = snn_csr.tocsr()

# Symmetrize
snn_csr = (snn_csr + snn_csr.T) / 2

adata = sc.AnnData(X=csr_matrix((N, 2)))
adata.obsp["connectivities"] = snn_csr
adata.uns["neighbors"] = {"connectivities_key": "connectivities",
                          "params": {"method": "umap", "n_neighbors": K_NN}}
sc.tl.umap(adata, min_dist=0.3, spread=1.0)
sc.tl.leiden(adata, resolution=2, flavor="igraph", n_iterations=2)

adata.obs["RNA.weight"] = w_rna.cpu().numpy()
try:
    celltype = pd.read_csv(OUT + "celltype_R.csv", index_col=0).iloc[:, 0]
    adata.obs["celltype.l2"] = celltype.values
except FileNotFoundError:
    print("[WARN] celltype_R.csv not found")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sc.pl.umap(adata, color="leiden",     title="Leiden (res=2)", ax=axes[0], show=False)
if "celltype.l2" in adata.obs:
    sc.pl.umap(adata, color="celltype.l2", title="celltype.l2", ax=axes[1], show=False)
plt.tight_layout()
plt.savefig("F:/workplace/python/WNN/pipeline_umap.png", dpi=150)
# plt.show()  # saved to disk, skip interactive

# Violin plot: RNA.weight by celltype.l2 (matches R's VlnPlot)
if "celltype.l2" in adata.obs:
    # Sort categories by mean RNA.weight
    order = (
        adata.obs.groupby("celltype.l2")["RNA.weight"]
        .mean().sort_values(ascending=False).index.tolist()
    )
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    sc.pl.violin(adata, keys="RNA.weight", groupby="celltype.l2",
                 order=order, rotation=90, ax=ax, show=False,
                 legend_loc=None)
    ax.set_title("RNA modality weight by cell type")
    ax.set_ylabel("RNA.weight")
    plt.tight_layout()
    plt.savefig("F:/workplace/python/WNN/violin_rna_weight.png", dpi=150)
    # plt.show()  # saved to disk, skip interactive

print("Done.")

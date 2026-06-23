"""
WNN (Weighted Nearest Neighbor) — Seurat v4 algorithm.
Reproduces: Hao et al., Cell 2021, "Integrated analysis of multimodal single-cell data"
"""

import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix
from scipy.stats import pearsonr, spearmanr
import scanpy as sc
import matplotlib.pyplot as plt

# =============================================================================
# 1. Load data
# =============================================================================
rna_all = pd.read_csv("F:/workplace/python/WNN/rna_pca.csv", index_col=0)
adt_all = pd.read_csv("F:/workplace/python/WNN/adt_pca.csv", index_col=0)
w_rna_seurat = pd.read_csv("F:/workplace/python/WNN/rna_weight_seurat.csv", index_col=0)

rna = rna_all.iloc[:, :30].copy()   # first 30 PCs
adt = adt_all.iloc[:, :18].copy()   # first 18 PCs
print("Cells:", rna.shape[0], "| RNA PCs:", rna.shape[1], "| ADT PCs:", adt.shape[1])

# =============================================================================
# 2. Hyper-parameters (matching Seurat defaults)
# =============================================================================
K_RANGE = 200   # knn.range: initial KNN pool size
K_SIGMA = 20    # k.sigma: cells for sigma bandwidth
K_NN    = 20    # k.nn: neighbours for prediction and final WNN
EPS     = 1e-8

# =============================================================================
# 3. L2-normalize (l2.norm = TRUE)
# =============================================================================
rna_t = torch.tensor(rna.values, dtype=torch.float32)
adt_t = torch.tensor(adt.values, dtype=torch.float32)
rna_t = torch.nn.functional.normalize(rna_t, p=2, dim=1)
adt_t = torch.nn.functional.normalize(adt_t, p=2, dim=1)
N = rna_t.shape[0]

# =============================================================================
# 4. Batched KNN (avoid O(N^2) memory)
# =============================================================================
print("Computing KNN (batched) …")

def batched_knn(data, k, batch_size=500):
    N = data.shape[0]
    knn_idx = torch.zeros((N, k - 1), dtype=torch.long)
    knn_dist = torch.zeros((N, k - 1))
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        d_batch = torch.cdist(data[start:end], data)
        top_d, top_i = d_batch.topk(k, largest=False, dim=1)
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

# Helper: look up distance from cell i to specific neighbor j from stored KNN
def knn_dist_to(knn_idx, knn_dist, i, nbrs):
    """Look up distances to neighbors from stored KNN distances."""
    d_vals = []
    for nb in nbrs:
        nb = nb.item() if isinstance(nb, torch.Tensor) else nb
        pos = (knn_idx[i] == nb).nonzero(as_tuple=True)
        if pos[0].numel() > 0:
            d_vals.append(knn_dist[i, pos[0][0]])
        else:
            d_vals.append(torch.tensor(float('inf')))
    return torch.stack(d_vals) if d_vals else torch.zeros(0)

# =============================================================================
# 5. Compute Jaccard similarity between neighbour sets
#    (used for sigma bandwidth and SNN graph)
# =============================================================================
print("Computing Jaccard / SNN …")
def build_snn(knn_idx):
    """Sparse SNN matrix (N x N) from KNN indices via Jaccard overlap."""
    rows, cols, vals = [], [], []
    for i in range(N):
        si = set(knn_idx[i].tolist())
        for j in knn_idx[i]:
            jj = j.item()
            sj = set(knn_idx[jj].tolist())
            inter = len(si & sj)
            val = inter / (len(si | sj) + EPS)
            if val > 0:
                rows.append(i)
                cols.append(jj)
                vals.append(val)
    idx = torch.tensor([rows, cols], dtype=torch.long)
    vv  = torch.tensor(vals, dtype=torch.float32)
    return torch.sparse_coo_tensor(idx, vv, (N, N)).coalesce()

snn_rna = build_snn(knn_rna)
snn_adt = build_snn(knn_adt)

# =============================================================================
# 6. Sigma bandwidth (paper method)
#    sigma[i] = mean distance to K_SIGMA neighbours with lowest non-zero Jaccard
# =============================================================================
print("Computing sigma …")
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
        d = knn_dist_to(knn_idx, knn_dist, i, sel)
        sigma[i] = d.sort(descending=True).values[:min(k_sig, d.size(0))].mean()
    return sigma

sigma_rna = compute_sigma(knn_dist_rna, knn_rna, snn_rna, K_SIGMA)
sigma_adt = compute_sigma(knn_dist_adt, knn_adt, snn_adt, K_SIGMA)

# =============================================================================
# 7. Within- and cross-modality prediction (paper method)
#    Predict cell from mean of K_NN neighbours, then L2 distance.
# =============================================================================
print("Computing within/cross prediction …")
rna_pred_self  = rna_t[knn_rna[:, :K_NN]].mean(dim=1)
rna_pred_cross = rna_t[knn_adt[:, :K_NN]].mean(dim=1)
adt_pred_self  = adt_t[knn_adt[:, :K_NN]].mean(dim=1)
adt_pred_cross = adt_t[knn_rna[:, :K_NN]].mean(dim=1)

d_rna_self  = torch.norm(rna_t - rna_pred_self,  dim=1)
d_rna_cross = torch.norm(rna_t - rna_pred_cross, dim=1)
d_adt_self  = torch.norm(adt_t - adt_pred_self,  dim=1)
d_adt_cross = torch.norm(adt_t - adt_pred_cross, dim=1)

# dnn = distance to nearest neighbour (2nd-smallest distance, 1st is self=0)
dnn_rna = knn_dist_rna[:, 0]
dnn_adt = knn_dist_adt[:, 0]

# =============================================================================
# 8. Kernel: exp(-max(0, d - dnn) / (sigma - dnn))  — paper eq.
#    Use float64 to match R precision, then cast back to float32.
# =============================================================================
print("Computing kernel & affinity …")
bw_rna = torch.clamp(sigma_rna - dnn_rna, min=EPS).double()
bw_adt = torch.clamp(sigma_adt - dnn_adt, min=EPS).double()

d_self_rna_d  = (d_rna_self  - dnn_rna).clamp(min=0).double()
d_cross_rna_d = (d_rna_cross - dnn_rna).clamp(min=0).double()
d_self_adt_d  = (d_adt_self  - dnn_adt).clamp(min=0).double()
d_cross_adt_d = (d_adt_cross - dnn_adt).clamp(min=0).double()

sim_rna_self  = torch.exp(-d_self_rna_d  / bw_rna).float()
sim_rna_cross = torch.exp(-d_cross_rna_d / bw_rna).float()
sim_adt_self  = torch.exp(-d_self_adt_d  / bw_adt).float()
sim_adt_cross = torch.exp(-d_cross_adt_d / bw_adt).float()

aff_rna = sim_rna_self / (sim_rna_cross + EPS)
aff_adt = sim_adt_self / (sim_adt_cross + EPS)

# =============================================================================
# 9. Modality weights via stable softmax
# =============================================================================
print("Computing modality weights …")
aff_max = torch.maximum(aff_rna, aff_adt)
w_rna = torch.exp(aff_rna - aff_max) / (torch.exp(aff_rna - aff_max) + torch.exp(aff_adt - aff_max))
w_adt = 1.0 - w_rna

# =============================================================================
# 10. Build WNN graph via weighted distances
#     D_ij = w_rna[i] * d_rna[i,j] + w_adt[i] * d_adt[i,j]
# =============================================================================
print("Building WNN graph …")
wnn_knn = torch.zeros((N, K_NN), dtype=torch.long)
for i in range(N):
    nbrs = torch.cat([knn_rna[i], knn_adt[i]]).unique()
    d_rna_nb = knn_dist_to(knn_rna, knn_dist_rna, i, nbrs)
    d_adt_nb = knn_dist_to(knn_adt, knn_dist_adt, i, nbrs)
    dw = w_rna[i] * d_rna_nb + w_adt[i] * d_adt_nb
    if nbrs.size(0) <= K_NN:
        wnn_knn[i, :nbrs.size(0)] = nbrs
    else:
        wnn_knn[i] = nbrs[dw.topk(K_NN, largest=False).indices]

# =============================================================================
# 11. Build SNN from WNN neighbours
# =============================================================================
print("Building WNN-SNN graph …")
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

# =============================================================================
# 12. Step-by-step comparison with R manual output
# =============================================================================
print("\n=== Step-by-step comparison (Python vs R manual) ===\n")
OUT_R = "F:/workplace/python/WNN/"

def compare(name, py_val, r_file, tol=1e-3):
    r_val = pd.read_csv(OUT_R + r_file, index_col=0).iloc[:, 0].values
    p_val = py_val.cpu().numpy().ravel()
    corr = np.corrcoef(p_val, r_val)[0, 1]
    max_d = np.abs(p_val - r_val).max()
    close = np.isclose(p_val, r_val, rtol=tol, atol=tol).mean()
    ok = "PASS" if close > 0.99 else "FAIL"
    print(f"  {ok} {name:20s}  corr={corr:.4f}  max_diff={max_d:.6f}  match={close:.3f}")

# 1. KNN indices overlap (R 1-based, Python 0-based → subtract 1 from R)
r_knn = pd.read_csv(OUT_R + "knn_rna_R.csv", index_col=0).values[:, :5] - 1
p_knn = knn_rna.cpu().numpy()[:, :5]
ov_rna = [len(set(r_knn[i]) & set(p_knn[i])) for i in range(10)]
r_knn_a = pd.read_csv(OUT_R + "knn_adt_R.csv", index_col=0).values[:, :5] - 1
p_knn_a = knn_adt.cpu().numpy()[:, :5]
ov_adt = [len(set(r_knn_a[i]) & set(p_knn_a[i])) for i in range(10)]
print(f"  RNA KNN overlap (first 10 cells, first 5 nbr): {ov_rna}")
print(f"  ADT KNN overlap (first 10 cells, first 5 nbr): {ov_adt}")

# 2. KNN distances
p_rna_d5 = knn_dist_rna[:, :5].mean(dim=1)
compare("knn_dist_rna",    p_rna_d5,  "knn_dist_rna_R.csv")

# 3. Sigma
compare("sigma_rna",       sigma_rna, "sigma_rna_R.csv")
compare("sigma_adt",       sigma_adt, "sigma_adt_R.csv")

# 4. Prediction distances
compare("d_rna_self",      d_rna_self,     "d_rna_self_R.csv")
compare("d_rna_cross",     d_rna_cross,    "d_rna_cross_R.csv")
compare("d_adt_self",      d_adt_self,     "d_adt_self_R.csv")
compare("d_adt_cross",     d_adt_cross,    "d_adt_cross_R.csv")

# Debug: print a few cells to find the discrepancy source
print("\n--- Debug: first 5 cells ---")
for i in range(5):
    print(f"Cell {i}: sigma={sigma_rna[i]:.6f} dnn={dnn_rna[i]:.6f} bw={bw_rna[i]:.6f}")
    print(f"       d_self={d_rna_self[i]:.6f} clamp={(d_rna_self[i]-dnn_rna[i]).clamp(min=0):.6f}")
    print(f"       exponent={((d_rna_self[i]-dnn_rna[i]).clamp(min=0)/bw_rna[i]):.6f} sim={sim_rna_self[i]:.6f}")

# Check for problematic cells where bw is very small
small_bw = (bw_rna < 1e-6).sum().item()
print(f"\nCells with bw_rna < 1e-6: {small_bw}")
if small_bw > 0:
    idx = (bw_rna < 1e-6).nonzero(as_tuple=True)[0][:3]
    for i in idx:
        print(f"  Cell {i}: sigma={sigma_rna[i]:.10f} dnn={dnn_rna[i]:.10f} bw={bw_rna[i]:.10e}")

# Check for NaN/Inf in sim
print(f"sim_rna_self  NaN: {torch.isnan(sim_rna_self).sum().item()}  Inf: {torch.isinf(sim_rna_self).sum().item()}")
print(f"sim_rna_cross NaN: {torch.isnan(sim_rna_cross).sum().item()}  Inf: {torch.isinf(sim_rna_cross).sum().item()}")

# 5. Kernel values
compare("sim_rna_self",    sim_rna_self,   "sim_rna_self_R.csv")
compare("sim_rna_cross",   sim_rna_cross,  "sim_rna_cross_R.csv")
compare("sim_adt_self",    sim_adt_self,   "sim_adt_self_R.csv")
compare("sim_adt_cross",   sim_adt_cross,  "sim_adt_cross_R.csv")

# 6. Affinity
compare("aff_rna",         aff_rna,        "aff_rna_R.csv")
compare("aff_adt",         aff_adt,        "aff_adt_R.csv")

# 7. Final weights vs R-manual
w_rna_R = pd.read_csv(OUT_R + "rna_weight_seurat.csv", index_col=0).iloc[:, 0].values
w_py_vals = w_rna.cpu().numpy().ravel()
print(f"\n  Python w_rna:   mean={w_py_vals.mean():.4f}  std={w_py_vals.std():.4f}")
print(f"  R-manual w_rna:  mean={w_rna_R.mean():.4f}  std={w_rna_R.std():.4f}")
print(f"  Pearson r = {np.corrcoef(w_py_vals, w_rna_R)[0,1]:.4f}")
torch.save({"w_rna": w_rna}, "F:/workplace/python/WNN/w_rna_from_R.pt")
print("  Saved w_rna to w_rna_from_R.pt")

# =============================================================================
# 13. UMAP & clustering
# =============================================================================
print("\nRunning UMAP …")
coo = snn_wnn.coalesce()
idx_s = coo.indices().cpu().numpy()
val_s = coo.values().cpu().numpy()
snn_csr = csr_matrix((val_s, (idx_s[0], idx_s[1])), shape=(N, N))

adata = sc.AnnData(X=csr_matrix((N, 2)), obs=pd.DataFrame(index=rna.index))
adata.obsp["connectivities"] = snn_csr
adata.uns["neighbors"] = {"connectivities_key": "connectivities",
                          "params": {"method": "umap", "n_neighbors": K_NN}}

sc.tl.umap(adata)
sc.tl.leiden(adata, resolution=2)

# add metadata
adata.obs["RNA.weight"] = w_rna.cpu().numpy()
try:
    celltype = pd.read_csv("F:/workplace/python/WNN/celltype_R.csv", index_col=0).iloc[:, 0]
    adata.obs["celltype.l2"] = celltype.values
except FileNotFoundError:
    print("[WARN] celltype_R.csv not found — run: write.csv(bm@meta.data[,'celltype.l2',drop=FALSE], ...) in R")

# plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sc.pl.umap(adata, color="leiden",     title="Leiden (res=2)", ax=axes[0], show=False)
if "celltype.l2" in adata.obs:
    sc.pl.umap(adata, color="celltype.l2", title="celltype.l2",   ax=axes[1], show=False)
#sc.pl.umap(adata, color="RNA.weight", cmap="RdYlBu",
#           title="RNA modality weight",        ax=axes[2], show=False)
plt.tight_layout()
plt.savefig("F:/workplace/python/WNN/wnn_umap.png", dpi=150)
plt.show()
print("Done.")



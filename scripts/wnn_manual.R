# =============================================================================
# WNN step-by-step reproduction in R (matching Seurat v4)
# Each step exported to CSV for cross-checking with Seurat and Python.
# =============================================================================
library(Seurat)
library(RANN)
library(Matrix)

#bm <- readRDS("F:/workplace/python/WNN/bm.rds")
OUT <- "F:/workplace/python/WNN/"

# =============================================================================
# 1. PCA + L2 normalisation (l2.norm = TRUE)
# =============================================================================
rna_pca <- Embeddings(bm, "pca")[,  1:30]
adt_pca <- Embeddings(bm, "apca")[, 1:18]

rna_l2 <- t(apply(rna_pca, 1, function(x) x / sqrt(sum(x^2))))
adt_l2 <- t(apply(adt_pca, 1, function(x) x / sqrt(sum(x^2))))

write.csv(rna_l2, paste0(OUT, "rna_l2_R.csv"))
write.csv(adt_l2, paste0(OUT, "adt_l2_R.csv"))

N <- nrow(rna_l2)

# =============================================================================
# 2. KNN for each modality (knn.range = 200)
# =============================================================================
k_range <- 200

rna_nn <- nn2(rna_l2, k = k_range + 1)
adt_nn <- nn2(adt_l2, k = k_range + 1)

rna_knn <- rna_nn$nn.idx[, -1, drop = FALSE]
adt_knn <- adt_nn$nn.idx[, -1, drop = FALSE]
rna_knn_dist <- rna_nn$nn.dists[, -1, drop = FALSE]
adt_knn_dist <- adt_nn$nn.dists[, -1, drop = FALSE]

write.csv(rna_knn,      paste0(OUT, "knn_rna_R.csv"))
write.csv(adt_knn,      paste0(OUT, "knn_adt_R.csv"))
write.csv(rna_knn_dist, paste0(OUT, "knn_dist_rna_R.csv"))
write.csv(adt_knn_dist, paste0(OUT, "knn_dist_adt_R.csv"))

# nearest neighbour distance (dnn) — column 1 after removing self
dnn_rna <- rna_knn_dist[, 1]
dnn_adt <- adt_knn_dist[, 1]

# =============================================================================
# 3. Jaccard similarity & sigma bandwidth (k.sigma = 20)
# =============================================================================
k_sigma   <- 20

cat("Computing sigma (Jaccard-based) …\n")
sigma_rna <- numeric(N)
sigma_adt <- numeric(N)

pb <- txtProgressBar(0, N, style = 3)
for (i in 1:N) {
    setTxtProgressBar(pb, i)

    # --- RNA sigma ---
    neighbors <- rna_knn[i, ]
    jac <- sapply(1:k_range, function(t) {
        si <- neighbors
        sj <- rna_knn[neighbors[t], ]
        length(intersect(si, sj)) / length(union(si, sj))
    })
    nz <- which(jac > 0)
    if (length(nz) == 0) {
        sigma_rna[i] <- mean(rna_knn_dist[i, 1:k_sigma])
    } else {
        ord <- order(jac[nz])
        use <- nz[ord[jac[nz[ord]] <= jac[nz[ord[min(k_sigma, length(nz))]]]]]
        d <- rna_knn_dist[i, use]
        k <- min(k_sigma, length(d))
        sigma_rna[i] <- mean(sort(d, decreasing = TRUE)[1:k])
    }

    # --- ADT sigma ---
    neighbors <- adt_knn[i, ]
    jac <- sapply(1:k_range, function(t) {
        si <- neighbors
        sj <- adt_knn[neighbors[t], ]
        length(intersect(si, sj)) / length(union(si, sj))
    })
    nz <- which(jac > 0)
    if (length(nz) == 0) {
        sigma_adt[i] <- mean(adt_knn_dist[i, 1:k_sigma])
    } else {
        ord <- order(jac[nz])
        use <- nz[ord[jac[nz[ord]] <= jac[nz[ord[min(k_sigma, length(nz))]]]]]
        d <- adt_knn_dist[i, use]
        k <- min(k_sigma, length(d))
        sigma_adt[i] <- mean(sort(d, decreasing = TRUE)[1:k])
    }
}
close(pb)

write.csv(sigma_rna, paste0(OUT, "sigma_rna_R.csv"))
write.csv(sigma_adt, paste0(OUT, "sigma_adt_R.csv"))

# =============================================================================
# 4. Within- and cross-modality prediction (k.nn = 20)
# =============================================================================
k_nn <- 20
eps  <- 1e-8

cat("Computing within/cross prediction …\n")

d_rna_self  <- numeric(N)
d_rna_cross <- numeric(N)
d_adt_self  <- numeric(N)
d_adt_cross <- numeric(N)

pb <- txtProgressBar(0, N, style = 3)
for (i in 1:N) {
    setTxtProgressBar(pb, i)

    # RNA within: predict from RNA neighbours
    pred_self <- colMeans(rna_l2[rna_knn[i, 1:k_nn], , drop = FALSE])
    d_rna_self[i] <- sqrt(sum((rna_l2[i, ] - pred_self)^2))

    # RNA cross: predict from ADT neighbours
    pred_cross <- colMeans(rna_l2[adt_knn[i, 1:k_nn], , drop = FALSE])
    d_rna_cross[i] <- sqrt(sum((rna_l2[i, ] - pred_cross)^2))

    # ADT within: predict from ADT neighbours
    pred_self <- colMeans(adt_l2[adt_knn[i, 1:k_nn], , drop = FALSE])
    d_adt_self[i] <- sqrt(sum((adt_l2[i, ] - pred_self)^2))

    # ADT cross: predict from RNA neighbours
    pred_cross <- colMeans(adt_l2[rna_knn[i, 1:k_nn], , drop = FALSE])
    d_adt_cross[i] <- sqrt(sum((adt_l2[i, ] - pred_cross)^2))
}
close(pb)

write.csv(d_rna_self,  paste0(OUT, "d_rna_self_R.csv"))
write.csv(d_rna_cross, paste0(OUT, "d_rna_cross_R.csv"))
write.csv(d_adt_self,  paste0(OUT, "d_adt_self_R.csv"))
write.csv(d_adt_cross, paste0(OUT, "d_adt_cross_R.csv"))

# =============================================================================
# 5. Kernel: exp(-max(0, d - dnn) / (sigma - dnn))
# =============================================================================
cat("Computing kernel …\n")

bw_rna <- sigma_rna - dnn_rna
bw_adt <- sigma_adt - dnn_adt

sim_rna_self  <- exp(-pmax(d_rna_self  - dnn_rna, 0) / bw_rna)
sim_rna_cross <- exp(-pmax(d_rna_cross - dnn_rna, 0) / bw_rna)
sim_adt_self  <- exp(-pmax(d_adt_self  - dnn_adt, 0) / bw_adt)
sim_adt_cross <- exp(-pmax(d_adt_cross - dnn_adt, 0) / bw_adt)

write.csv(sim_rna_self,  paste0(OUT, "sim_rna_self_R.csv"))
write.csv(sim_rna_cross, paste0(OUT, "sim_rna_cross_R.csv"))
write.csv(sim_adt_self,  paste0(OUT, "sim_adt_self_R.csv"))
write.csv(sim_adt_cross, paste0(OUT, "sim_adt_cross_R.csv"))

# =============================================================================
# 6. Affinity = within / cross
# =============================================================================
aff_rna <- sim_rna_self / (sim_rna_cross + eps)
aff_adt <- sim_adt_self / (sim_adt_cross + eps)

write.csv(aff_rna, paste0(OUT, "aff_rna_R.csv"))
write.csv(aff_adt, paste0(OUT, "aff_adt_R.csv"))

# =============================================================================
# 7. Modality weights via softmax
# =============================================================================
aff_max <- pmax(aff_rna, aff_adt)
w_rna_manual <- exp(aff_rna - aff_max) / (exp(aff_rna - aff_max) + exp(aff_adt - aff_max))
w_adt_manual <- exp(aff_adt - aff_max) / (exp(aff_rna - aff_max) + exp(aff_adt - aff_max))

write.csv(w_rna_manual, paste0(OUT, "w_rna_manual_R.csv"))
write.csv(w_adt_manual, paste0(OUT, "w_adt_manual_R.csv"))

# =============================================================================
# 8. Compare with Seurat's weights
# =============================================================================
names(w_rna_manual) <- rownames(rna_l2)
idx <- match(names(bm$RNA.weight), names(w_rna_manual))
w_man <- w_rna_manual[idx[!is.na(idx)]]
w_seu <- bm$RNA.weight[!is.na(idx)]

cat("\n========== Comparison ==========\n")
cat(sprintf("Manual w_rna:  mean=%.4f  sd=%.4f\n", mean(w_man), sd(w_man)))
cat(sprintf("Seurat w_rna:  mean=%.4f  sd=%.4f\n", mean(w_seu), sd(w_seu)))
cat(sprintf("Pearson  r = %.4f\n",  cor(w_man, w_seu)))
cat(sprintf("Spearman ρ = %.4f\n",  cor(w_man, w_seu, method = "spearman")))

cat("\nDone. All intermediate files written to:\n  ", OUT, "\n")

bm <- RunUMAP(bm, nn.name = "weighted.nn", reduction.name = "wnn.umap", reduction.key = "wnnUMAP_")
bm <- FindClusters(bm, graph.name = "wsnn", algorithm = 3, resolution = 2, verbose = FALSE)
p1 <- DimPlot(bm, reduction = 'wnn.umap', label = TRUE, repel = TRUE, label.size = 2.5) + NoLegend()
p2 <- DimPlot(bm, reduction = 'wnn.umap', group.by = 'celltype.l2', label = TRUE, repel = TRUE, label.size = 2.5) + NoLegend()
p1 + p2

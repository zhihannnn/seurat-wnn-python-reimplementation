"""
Seurat V4 preprocessing in Python
NormalizeData, FindVariableFeatures, ScaleData, RunPCA
"""
import numpy as np
import torch
import statsmodels.api as sm
from sklearn.decomposition import PCA


class NormalizeData:
    def __init__(self, method="log", scale_factor=10000, eps=1e-8, margin=1):
        self.method = method
        self.scale_factor = scale_factor
        self.eps = eps
        self.margin = margin

    def __call__(self, x):
        x = x.clone()
        if self.method == "log":
            return self._log_normalize(x)
        elif self.method == "clr":
            return self._clr_normalize(x)
        elif self.method == "rc":
            return self._rc_normalize(x)
        raise ValueError(f"Unknown method: {self.method}")

    def _log_normalize(self, x):
        lib = x.sum(dim=self.margin, keepdim=True) + self.eps
        return torch.log1p(x / lib * self.scale_factor)

    def _clr_normalize(self, x):
        n_feat = x.size(self.margin)
        log1p_x = torch.log1p(x)
        g = torch.exp(log1p_x.sum(dim=self.margin, keepdim=True) / n_feat)
        return torch.log1p(x / g)

    def _rc_normalize(self, x):
        lib = x.sum(dim=self.margin, keepdim=True) + self.eps
        return (x / lib) * self.scale_factor


class FindVariableFeatures:
    def __init__(self, n_features=2000, span=0.3,
                 mean_cutoff=(0.0125, 3), dispersion_cutoff=(0.5, float('inf'))):
        self.n_features = n_features
        self.span = span
        self.mean_cutoff = mean_cutoff
        self.dispersion_cutoff = dispersion_cutoff

    def __call__(self, x):
        """x: (cells, features) log-normalized tensor. Returns top_idx."""
        mean = x.mean(dim=0).numpy()
        var = x.var(dim=0, unbiased=True).numpy()

        var_expected = np.zeros_like(var)
        var_std = np.zeros_like(var)

        mask = var > 0
        if mask.sum() < 3:
            raise ValueError("Fewer than 3 features have non-zero variance.")

        log_mean = np.log10(mean[mask])
        log_var = np.log10(var[mask])
        fit = sm.nonparametric.lowess(
            log_var, log_mean, frac=self.span, it=3,
            delta=0.01 * (log_mean.max() - log_mean.min()), return_sorted=True
        )
        exp_log_var = np.interp(log_mean, fit[:, 0], fit[:, 1])
        var_expected[mask] = 10 ** exp_log_var
        var_std[mask] = var[mask] / var_expected[mask]

        keep = np.ones(len(mean), dtype=bool)
        if self.mean_cutoff is not None:
            lo, hi = self.mean_cutoff
            keep &= (mean >= lo) & (mean <= hi)
        if self.dispersion_cutoff is not None:
            lo, hi = self.dispersion_cutoff
            keep &= (var_std >= lo) & (var_std <= hi)
        keep &= (var_std > 0)

        kept_idx = np.where(keep)[0]
        order = np.argsort(var_std[keep])[::-1]
        n_select = min(self.n_features, len(kept_idx))
        top_idx = kept_idx[order[:n_select]]

        self.mean = mean
        self.variance = var
        self.variance_expected = var_expected
        self.variance_standardized = var_std
        self.top_idx = top_idx
        return top_idx


class ScaleData:
    """Per-feature z-score: (x - mean) / sd, clip at scale_max."""
    def __init__(self, scale_max=10, margin=1):
        self.scale_max = scale_max
        self.margin = margin

    def fit(self, x):
        """x: numpy (features, cells). Compute mean/std across cells."""
        self.mean = x.mean(axis=self.margin, keepdims=True)
        self.sd = x.std(axis=self.margin, ddof=1, keepdims=True)
        self.sd[self.sd < 1e-10] = 1

    def transform(self, x):
        scaled = (x - self.mean) / self.sd
        return np.clip(scaled, -self.scale_max, self.scale_max)

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)


class RunPCA:
    """PCA via sklearn. Input: (cells, features) scaled array. Returns embeddings, loadings, stdev."""
    def __init__(self, n_pcs=50):
        self.n_pcs = n_pcs

    def fit(self, x):
        """x: (cells, features) numpy float64."""
        self.pca = PCA(n_components=min(self.n_pcs, x.shape[0], x.shape[1]))
        self.pca.fit(x)
        self.stdev = np.sqrt(self.pca.explained_variance_)
        self.loadings = self.pca.components_.T

    def transform(self, x):
        emb = self.pca.transform(x)
        # Align sign with known convention: first cell's PC1 should match R if needed
        return emb

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)

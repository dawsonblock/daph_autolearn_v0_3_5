from __future__ import annotations
import numpy as np
import torch
from typing import Optional

def cca(X:np.ndarray,Y:np.ndarray,n_components:int=10,reg:float=1e-6)->dict:
    """Canonical Correlation Analysis between two representation matrices.

    Finds linear projections U, V such that corr(U^T X, V^T Y) is maximized.

    Used to identify shared subspaces between natural-language and symbolic
    reasoning representations (corrected plan section 12).

    Caveat: on arithmetic, NL and symbolic forms share the same numbers and
    operator, so CCA will find a shared subspace dominated by operand
    embeddings — the uninteresting part. Interpret CCA results on arithmetic
    accordingly.

    Args:
        X: [N, D1] representation matrix (e.g., natural language activations)
        Y: [N, D2] representation matrix (e.g., symbolic activations)
        n_components: number of canonical components to extract
        reg: regularization for numerical stability

    Returns:
        dict with:
          - correlations: list of canonical correlations
          - U: [D1, k] projection for X
          - V: [D2, k] projection for Y
          - shared_subspace: [N, k] shared representation
    """
    X=np.asarray(X,float);Y=np.asarray(Y,float)
    N=X.shape[0]
    if X.shape[0]!=Y.shape[0]:
        raise ValueError(f"X and Y must have same number of samples: {X.shape[0]} vs {Y.shape[0]}")

    # Center
    X=X-X.mean(axis=0,keepdims=True)
    Y=Y-Y.mean(axis=0,keepdims=True)

    # Compute covariance matrices
    Cxx=X.T@X+reg*np.eye(X.shape[1])
    Cyy=Y.T@Y+reg*np.eye(Y.shape[1])
    Cxy=X.T@Y

    # CCA via SVD of the whitened cross-covariance
    # K = Cxx^{-1/2} @ Cxy @ Cyy^{-1/2}
    # where Cxx^{-1/2} = L_x^{-T} with Cxx = L_x @ L_x^T (Cholesky)
    Lx=np.linalg.cholesky(Cxx)
    Ly=np.linalg.cholesky(Cyy)
    Lx_inv=np.linalg.inv(Lx)
    Ly_inv=np.linalg.inv(Ly)
    K=Lx_inv@Cxy@Ly_inv.T

    U_svd,S_svd,Vt_svd=np.linalg.svd(K)
    # Canonical correlations = singular values of K
    correlations=S_svd[:n_components]

    # Projection matrices
    # U maps X-space to canonical variables: a = U^T @ X_centered
    # V maps Y-space to canonical variables: b = V^T @ Y_centered
    U=Lx_inv.T@U_svd[:,:n_components]  # [D1, k]
    V=Ly_inv.T@Vt_svd[:n_components,:].T  # [D2, k]

    # Shared subspace representation
    shared_subspace=X@U

    return {
        "correlations":correlations.tolist(),
        "U":U,
        "V":V,
        "shared_subspace":shared_subspace,
        "mean_correlation":float(np.mean(correlations)),
        "max_correlation":float(np.max(correlations)) if len(correlations)>0 else 0.0,
    }

def project_to_shared(h:np.ndarray,U:np.ndarray)->np.ndarray:
    """Project a representation h onto the shared subspace using U.

    h' = h @ U  gives the shared subspace coordinates.
    To steer along the shared subspace: h + lambda * (h @ U @ U^T)
    """
    return h@U

def shared_subspace_steering(h:np.ndarray,U:np.ndarray,lam:float=1.0)->np.ndarray:
    """Steer h along the shared subspace: h' = h + lambda * h @ P_shared.

    P_shared = U @ U^T is the projection onto the shared subspace.
    h is [N, D], P_shared is [D, D], so h @ P_shared gives [N, D].
    """
    P_shared=U@U.T
    return h+lam*(h@P_shared)

def cca_summary(cca_result:dict)->dict:
    """Produce a human-readable summary of CCA results."""
    corrs=cca_result["correlations"]
    return {
        "n_components":len(corrs),
        "mean_correlation":cca_result["mean_correlation"],
        "max_correlation":cca_result["max_correlation"],
        "correlations":[float(c) for c in corrs],
        "top5_correlations":[float(c) for c in corrs[:5]],
        "U_shape":list(cca_result["U"].shape),
        "V_shape":list(cca_result["V"].shape),
    }

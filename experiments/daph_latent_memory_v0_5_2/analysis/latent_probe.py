#!/usr/bin/env python
"""Latent probes — diagnostic only, never in the production pathway.

Tests whether the latent predicts:
  Problem properties: operation_type, operand_1, operand_2, answer, difficulty
  Process properties: current_step, next_operation, intermediate_result

Measures: linear-probe accuracy, nonlinear-probe accuracy,
          mutual information, nearest-neighbor label agreement,
          cosine clustering, centered kernel alignment.
"""
from __future__ import annotations
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

def linear_probe(features:np.ndarray,labels:np.ndarray,seed:int=1337)->dict:
    """Linear probe accuracy."""
    if len(np.unique(labels))<2 or len(labels)<20: return {"accuracy":None,"status":"insufficient_data"}
    try:
        xtr,xte,ytr,yte=train_test_split(features,labels,test_size=0.3,random_state=seed,stratify=labels)
    except ValueError:
        xtr,xte,ytr,yte=train_test_split(features,labels,test_size=0.3,random_state=seed)
    clf=LogisticRegression(max_iter=2000,C=1.0);clf.fit(xtr,ytr)
    return {"accuracy":float(accuracy_score(yte,clf.predict(xte))),"status":"ok"}

def nonlinear_probe(features:np.ndarray,labels:np.ndarray,seed:int=1337)->dict:
    """Nonlinear (MLP) probe accuracy."""
    if len(np.unique(labels))<2 or len(labels)<20: return {"accuracy":None,"status":"insufficient_data"}
    try:
        xtr,xte,ytr,yte=train_test_split(features,labels,test_size=0.3,random_state=seed,stratify=labels)
    except ValueError:
        xtr,xte,ytr,yte=train_test_split(features,labels,test_size=0.3,random_state=seed)
    clf=MLPClassifier(hidden_layer_sizes=(128,64),max_iter=500,random_state=seed)
    clf.fit(xtr,ytr)
    return {"accuracy":float(accuracy_score(yte,clf.predict(xte))),"status":"ok"}

def nearest_neighbor_agreement(features:np.ndarray,labels:np.ndarray,k:int=5)->dict:
    """Nearest-neighbor label agreement: fraction of k-NN that share the same label."""
    if len(np.unique(labels))<2 or len(labels)<k+1: return {"agreement":None,"status":"insufficient_data"}
    clf=KNeighborsClassifier(n_neighbors=k);clf.fit(features,labels)
    pred=clf.predict(features)
    return {"agreement":float(accuracy_score(labels,pred)),"status":"ok"}

def cosine_clustering(features:np.ndarray,labels:np.ndarray,n_clusters:int|None=None)->dict:
    """Cluster by cosine similarity and measure cluster purity."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import homogeneity_completeness_v_measure
    if len(np.unique(labels))<2 or len(labels)<10: return {"purity":None,"status":"insufficient_data"}
    if n_clusters is None: n_clusters=len(np.unique(labels))
    normalized=features/(np.linalg.norm(features,axis=1,keepdims=True).clip(min=1e-8))
    km=KMeans(n_clusters=n_clusters,n_init=10,random_state=1337);km.fit(normalized)
    hom,comp,vmeas=homogeneity_completeness_v_measure(labels,km.labels_)
    return {"homogeneity":float(hom),"completeness":float(comp),"v_measure":float(vmeas),"status":"ok"}

def centered_kernel_alignment(features:np.ndarray,labels:np.ndarray)->dict:
    """CKA between the latent representation and a one-hot label matrix."""
    from sklearn.preprocessing import OneHotEncoder
    if len(np.unique(labels))<2 or len(labels)<10: return {"cka":None,"status":"insufficient_data"}
    enc=OneHotEncoder(sparse_output=False);onehot=enc.fit_transform(labels.reshape(-1,1))
    # Linear CKA
    X=features-features.mean(axis=0,keepdims=True)
    Y=onehot-onehot.mean(axis=0,keepdims=True)
    XTX=X.T@X;YTY=Y.T@Y;YTX=Y.T@X
    numerator=float(np.trace(YTX@YTX.T))
    denominator=float(np.sqrt(np.sum(XTX**2)*np.sum(YTY**2)))
    cka=numerator/denominator if denominator>0 else 0.0
    return {"cka":cka,"status":"ok"}

def run_all_probes(latents:np.ndarray,properties:dict[str,np.ndarray],seed:int=1337)->dict:
    """Run all probes on a set of latents with labeled properties.

    Args:
        latents: [N, D] latent representations
        properties: dict mapping property name to [N] label array

    Returns:
        dict of probe results per property
    """
    results={}
    for prop_name,labels in properties.items():
        results[prop_name]={
            "linear_probe":linear_probe(latents,labels,seed=seed),
            "nonlinear_probe":nonlinear_probe(latents,labels,seed=seed),
            "nearest_neighbor":nearest_neighbor_agreement(latents,labels),
            "cosine_clustering":cosine_clustering(latents,labels),
            "cka":centered_kernel_alignment(latents,labels),
        }
    return results

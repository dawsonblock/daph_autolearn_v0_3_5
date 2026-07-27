from __future__ import annotations
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def leakage_probe(features:np.ndarray,labels:np.ndarray,seed:int=1337)->dict:
    if len(np.unique(labels))<2 or len(labels)<20:return {"status":"insufficient_data","accuracy":None}
    xtr,xte,ytr,yte=train_test_split(features,labels,test_size=.3,random_state=seed,stratify=labels)
    clf=LogisticRegression(max_iter=2000);clf.fit(xtr,ytr);pred=clf.predict(xte)
    return {"status":"ok","accuracy":float(accuracy_score(yte,pred))}

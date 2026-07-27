from __future__ import annotations
import math
import numpy as np
from scipy import stats as sp_stats

# =============================================================================
# v0.5.2 Statistical Tests (corrected plan section 18)
#
# McNemar: two-condition paired binary accuracy comparison
# Cochran's Q: multi-condition (e.g., corruption sweep) binary accuracy
# Wilcoxon: paired continuous (per-example accuracy curves)
# Paired bootstrap: CI on the difference
# =============================================================================

def mcnemar_test(matched_correct:list[int],other_correct:list[int])->dict:
    """McNemar test for paired binary outcomes.

    b = count(matched=1, other=0)
    c = count(matched=0, other=1)

    χ² = (|b-c| - 1)² / (b + c)   (with continuity correction)

    Also computes exact McNemar (binomial) p-value.
    """
    if len(matched_correct)!=len(other_correct):
        raise ValueError("Conditions must have same length")
    b=sum(1 for m,o in zip(matched_correct,other_correct) if m==1 and o==0)
    c=sum(1 for m,o in zip(matched_correct,other_correct) if m==0 and o==1)
    n=b+c
    if n==0:
        return {"chi2":0.0,"p_value":1.0,"b":0,"c":0,"exact_p":1.0,"effect_size":0.0}
    chi2=((abs(b-c)-1)**2)/n
    # Exact McNemar (binomial test)
    exact_p=float(sp_stats.binomtest(min(b,c),n,0.5).pvalue)
    # Effect size (proportion difference)
    diff=(b-c)/len(matched_correct)
    return {"chi2":float(chi2),"p_value":float(sp_stats.chi2.sf(chi2,1)),"b":b,"c":c,"exact_p":exact_p,"effect_size":float(diff)}

def cochran_q_test(conditions:list[list[int]])->dict:
    """Cochran's Q test for k paired binary conditions.

    Used for the corruption sweep (multiple α levels).
    """
    k=len(conditions)
    n=len(conditions[0])
    if k<3: raise ValueError("Cochran's Q requires at least 3 conditions")
    for cond in conditions:
        if len(cond)!=n: raise ValueError("All conditions must have same length")
    # Number of successes per condition
    Lj=[sum(cond) for cond in conditions]
    # Number of successes per subject
    Li=[sum(conditions[j][i] for j in range(k)) for i in range(n)]
    L_sum=sum(Lj)
    L_sq_sum=sum(l*l for l in Lj)
    Li_sq_sum=sum(l*l for l in Li)
    numerator=k*L_sq_sum-L_sum*L_sum
    denominator=k*Li_sq_sum-L_sum*L_sum
    if denominator==0: return {"Q":0.0,"p_value":1.0,"df":k-1}
    Q=numerator/denominator
    p_value=float(sp_stats.chi2.sf(Q,k-1))
    return {"Q":float(Q),"p_value":p_value,"df":k-1}

def wilcoxon_signed_rank(x:list[float],y:list[float])->dict:
    """Wilcoxon signed-rank test for paired continuous data.

    Used for per-example accuracy curves across the corruption sweep.
    """
    if len(x)!=len(y): raise ValueError("x and y must have same length")
    diffs=[a-b for a,b in zip(x,y) if a!=b]
    if not diffs: return {"statistic":0.0,"p_value":1.0,"n":0}
    result=sp_stats.wilcoxon(x,y,alternative="two-sided")
    return {"statistic":float(result.statistic),"p_value":float(result.pvalue),"n":len(diffs)}

def paired_bootstrap_ci(
    a_values:list[float],
    b_values:list[float],
    samples:int=1000,
    seed:int=1337,
)->dict:
    """Paired bootstrap CI on the difference a - b."""
    if len(a_values)!=len(b_values): raise ValueError("a and b must have same length")
    rng=np.random.RandomState(seed);n=len(a_values)
    a=np.array(a_values);b=np.array(b_values);diffs=a-b
    boot_means=[]
    for _ in range(samples):
        idx=rng.randint(0,n,n)
        boot_means.append(diffs[idx].mean())
    boot_means=np.sort(boot_means)
    observed=float(diffs.mean())
    return {
        "mean_diff":observed,
        "lo":float(boot_means[int(0.025*(samples-1))]),
        "hi":float(boot_means[int(0.975*(samples-1))]),
        "se":float(diffs.std(ddof=1)/math.sqrt(n)),
    }

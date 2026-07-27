from __future__ import annotations
import math,random,re,statistics
import numpy as np

def normalize_answer(text:str)->str:
    text=text.strip();nums=re.findall(r"[-+]?\d+(?:\.\d+)?",text);return nums[-1] if nums else text.lower().strip()

def exact_match(pred:str,gold:str)->int:
    return int(normalize_answer(pred)==normalize_answer(gold))

def bootstrap_ci(values:list[float],samples:int=1000,seed:int=1337)->dict:
    if not values: return {"mean":math.nan,"lo":math.nan,"hi":math.nan}
    rng=random.Random(seed);means=[];n=len(values)
    for _ in range(samples): means.append(statistics.fmean([values[rng.randrange(n)] for _ in range(n)]))
    means.sort()
    return {"mean":statistics.fmean(values),"lo":means[int(.025*(samples-1))],"hi":means[int(.975*(samples-1))]}

def utility(accuracy:float,latency_s:float,tokens:int,cfg:dict)->float:
    return cfg["accuracy_weight"]*accuracy-cfg["latency_weight"]*latency_s-cfg["token_weight"]*tokens

# =============================================================================
# v0.5.2 Headline Metrics
# =============================================================================

def causal_specificity_score(a_matched:float,a_shuffled:float)->float:
    """CSS = A_matched - A_shuffled (in percentage points)."""
    return (a_matched-a_shuffled)*100.0

def wrong_memory_penalty(a_matched:float,a_wrong_class:float)->float:
    """WMP = A_matched - A_wrong_class (in percentage points)."""
    return (a_matched-a_wrong_class)*100.0

def latent_utility_gain(a_latent:float,a_base:float,latent_token_cost:int)->float:
    """LUG = (A_latent - A_base) / (latent_token_cost + 1)."""
    return (a_latent-a_base)/(latent_token_cost+1)

def generalization_gain(a_ood_latent:float,a_ood_base:float)->float:
    """GG = A_OOD,latent - A_OOD,base."""
    return a_ood_latent-a_ood_base

# =============================================================================
# CSS Null Distribution (corrected plan section 18.3)
# =============================================================================

def css_null_distribution(
    matched_correct:list[int],
    shuffled_correct:list[int],
    n_permutations:int=200,
    seed:int=1337,
)->list[float]:
    """Compute CSS under label permutation (null distribution).

    For each permutation, randomly swap the matched/shuffled assignment for
    each example and recompute CSS. Under the null (latent carries no
    query-specific info), E[CSS]=0.

    Args:
        matched_correct: binary list (1=correct, 0=incorrect) for matched condition
        shuffled_correct: binary list for shuffled condition
        n_permutations: number of permutations
        seed: RNG seed

    Returns:
        List of CSS values under the null (in percentage points)
    """
    rng=random.Random(seed)
    n=len(matched_correct)
    if n!=len(shuffled_correct): raise ValueError("matched and shuffled must have same length")
    null_css:list[float]=[]
    a_matched_true=statistics.fmean(matched_correct)
    a_shuffled_true=statistics.fmean(shuffled_correct)
    for _ in range(n_permutations):
        # For each example, randomly decide whether to swap its matched/shuffled labels
        perm_matched=[];perm_shuffled=[]
        for i in range(n):
            if rng.random()<0.5:
                perm_matched.append(shuffled_correct[i]);perm_shuffled.append(matched_correct[i])
            else:
                perm_matched.append(matched_correct[i]);perm_shuffled.append(shuffled_correct[i])
        a_m=statistics.fmean(perm_matched);a_s=statistics.fmean(perm_shuffled)
        null_css.append((a_m-a_s)*100.0)
    return null_css

def css_threshold(null_css_values:list[float],percentile:float=99.0,floor_pp:float=5.0)->float:
    """Compute the CSS gate threshold from the null distribution.

    threshold = max(null_percentile, floor_pp)
    """
    if not null_css_values: return floor_pp
    null_array=np.array(null_css_values)
    threshold=float(np.percentile(null_array,percentile))
    return max(threshold,floor_pp)

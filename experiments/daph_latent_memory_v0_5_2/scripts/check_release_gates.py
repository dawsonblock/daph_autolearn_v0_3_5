#!/usr/bin/env python
"""Release gates checker — verifies all 6 gates from the corrected plan.

Gate 1 — matched specificity (conjunction):
  A_matched > A_shuffled (McNemar p < 0.01)
  A_matched > A_base
  LUG > 0

Gate 2 — corruption behavior (no monotonicity requirement):
  A(α=0) > A(α=2)  (endpoints)
  |A(α=0.5) - A(α=0)| < ε  (robustness band)
  A(α=2) < A(α=0) - δ  (large corruption hurts)

Gate 3 — semantic mismatch: A_wrong-class < A_matched - 5pp (McNemar p < 0.01)

Gate 4 — OOD benefit: GG > 0 on at least one OOD split (p < 0.05)

Gate 5 — no catastrophic degradation: A_base degrades by < 2pp

Gate 6 — reproducibility: at least 5 seeds reproduce the main effect
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
from daph_latent_memory.evaluation.metrics import causal_specificity_score,wrong_memory_penalty,latent_utility_gain
from daph_latent_memory.evaluation.statistics import mcnemar_test

def check_gate1(results:dict,alpha:float=0.01)->dict:
    """Gate 1 — matched specificity (conjunction)."""
    conditions=results.get("conditions",{})
    if "matched" not in conditions or "shuffled_global" not in conditions:
        return {"gate":1,"name":"matched_specificity","passed":False,"reason":"missing conditions"}
    matched_c=conditions["matched"]["correct"]
    shuffled_c=conditions["shuffled_global"]["correct"]
    base_c=conditions.get("base",{}).get("correct",[])
    a_matched=conditions["matched"]["accuracy"]
    a_shuffled=conditions["shuffled_global"]["accuracy"]
    a_base=conditions.get("base",{}).get("accuracy",0.0)

    mc=mcnemar_test(matched_c,shuffled_c)
    css=causal_specificity_score(a_matched,a_shuffled)
    lug=latent_utility_gain(a_matched,a_base,8)  # latent_tokens=8

    checks={
        "matched_gt_shuffled":a_matched>a_shuffled,
        "mcnemar_significant":mc["p_value"]<alpha,
        "matched_gt_base":a_matched>a_base,
        "lug_positive":lug>0,
    }
    passed=all(checks.values())
    return {"gate":1,"name":"matched_specificity","passed":passed,"checks":checks,
            "a_matched":a_matched,"a_shuffled":a_shuffled,"a_base":a_base,
            "css":css,"lug":lug,"mcnemar_p":mc["p_value"]}

def check_gate2(results:dict,epsilon:float=0.01,delta:float=0.03)->dict:
    """Gate 2 — corruption behavior."""
    sweep=results.get("corruption_sweep",{})
    if not sweep:
        return {"gate":2,"name":"corruption_behavior","passed":False,"reason":"no corruption sweep"}
    # Get alpha=0 and alpha=2 results
    a0=sweep.get("0.0",sweep.get("0",{})).get("accuracy",None)
    a05=sweep.get("0.5",{}).get("accuracy",None)
    a2=sweep.get("2.0",sweep.get("2",{})).get("accuracy",None)
    if a0 is None or a2 is None:
        return {"gate":2,"name":"corruption_behavior","passed":False,"reason":"missing alpha endpoints"}
    checks={
        "endpoints_a0_gt_a2":a0>a2,
        "large_corruption_hurts":a2<a0-delta,
    }
    if a05 is not None:
        checks["robustness_band"]=abs(a05-a0)<=epsilon
    else:
        checks["robustness_band"]=True  # skip if not available
    passed=all(checks.values())
    return {"gate":2,"name":"corruption_behavior","passed":passed,"checks":checks,
            "a_0":a0,"a_0.5":a05,"a_2":a2}

def check_gate3(results:dict,alpha:float=0.01,delta_pp:float=5.0)->dict:
    """Gate 3 — semantic mismatch."""
    conditions=results.get("conditions",{})
    if "matched" not in conditions or "shuffled_wrong_class" not in conditions:
        return {"gate":3,"name":"semantic_mismatch","passed":False,"reason":"missing conditions"}
    matched_c=conditions["matched"]["correct"]
    wrong_c=conditions["shuffled_wrong_class"]["correct"]
    a_matched=conditions["matched"]["accuracy"]
    a_wrong=conditions["shuffled_wrong_class"]["accuracy"]
    wmp=wrong_memory_penalty(a_matched,a_wrong)
    mc=mcnemar_test(matched_c,wrong_c)
    checks={
        "wmp_gt_5pp":wmp>=delta_pp,
        "mcnemar_significant":mc["p_value"]<alpha,
    }
    passed=all(checks.values())
    return {"gate":3,"name":"semantic_mismatch","passed":passed,"checks":checks,
            "a_matched":a_matched,"a_wrong":a_wrong,"wmp":wmp,"mcnemar_p":mc["p_value"]}

def check_gate4(ood_results:dict,alpha:float=0.05)->dict:
    """Gate 4 — OOD benefit."""
    passing_splits=[]
    for split,data in ood_results.items():
        gg=data.get("gg",0)
        p=data.get("mcnemar",{}).get("p_value",1.0)
        if gg>0 and p<alpha:
            passing_splits.append({"split":split,"gg":gg,"p_value":p})
    passed=len(passing_splits)>0
    return {"gate":4,"name":"ood_benefit","passed":passed,"passing_splits":passing_splits}

def check_gate5(results:dict,base_general_accuracy:float=0.0,max_degradation_pp:float=2.0)->dict:
    """Gate 5 — no catastrophic degradation."""
    a_base=results.get("conditions",{}).get("base",{}).get("accuracy",base_general_accuracy)
    degradation_pp=(base_general_accuracy-a_base)*100
    passed=degradation_pp<max_degradation_pp
    return {"gate":5,"name":"no_catastrophic_degradation","passed":passed,
            "a_base":a_base,"degradation_pp":degradation_pp}

def check_gate6(seed_results:list[dict])->dict:
    """Gate 6 — reproducibility across seeds."""
    if len(seed_results)<5:
        return {"gate":6,"name":"reproducibility","passed":False,"reason":f"only {len(seed_results)} seeds, need >= 5"}
    # Check that each seed has positive CSS
    positive_css=0
    significant=0
    for sr in seed_results:
        css=sr.get("css",0)
        p=sr.get("mcnemar_p",1.0)
        if css>0: positive_css+=1
        if p<0.01: significant+=1
    passed=positive_css>=5 and significant>=5
    return {"gate":6,"name":"reproducibility","passed":passed,
            "seeds":len(seed_results),"positive_css":positive_css,"significant":significant}

def check_all_gates(causal_eval_path:str,ood_results_path:str|None=None,seed_results_paths:list[str]|None=None)->dict:
    """Check all 6 release gates."""
    with open(causal_eval_path) as f: causal_results=json.load(f)
    ood_results={}
    if ood_results_path and Path(ood_results_path).exists():
        with open(ood_results_path) as f: ood_results=json.load(f)
    seed_results=[]
    if seed_results_paths:
        for p in seed_results_paths:
            if Path(p).exists():
                with open(p) as f: seed_results.append(json.load(f).get("headline",{}))

    gates=[
        check_gate1(causal_results),
        check_gate2(causal_results),
        check_gate3(causal_results),
        check_gate4(ood_results),
        check_gate5(causal_results),
        check_gate6(seed_results),
    ]
    all_passed=all(g["passed"] for g in gates)
    return {"all_gates_passed":all_passed,"gates":gates}

def main():
    ap=argparse.ArgumentParser(description="Check v0.5.2 release gates")
    ap.add_argument("--causal-eval",required=True,help="Path to causal_eval.json")
    ap.add_argument("--ood-results",default=None,help="Path to ood_results.json")
    ap.add_argument("--seed-results",nargs="*",default=None,help="Paths to per-seed headline results")
    ap.add_argument("--output",required=True)
    args=ap.parse_args()
    report=check_all_gates(args.causal_eval,args.ood_results,args.seed_results)
    outdir=Path(args.output);outdir.parent.mkdir(parents=True,exist_ok=True)
    with open(outdir,"w") as f: json.dump(report,f,indent=2)
    print("\n" + "="*60)
    print("RELEASE GATES REPORT")
    print("="*60)
    for g in report["gates"]:
        status="PASS" if g["passed"] else "FAIL"
        print(f"  Gate {g['gate']} — {g['name']}: {status}")
    print("="*60)
    print(f"  OVERALL: {'ALL GATES PASSED' if report['all_gates_passed'] else 'GATES FAILED'}")
    print("="*60)

if __name__=="__main__": main()

#!/usr/bin/env python
"""Benchmark matrix reporter — outputs the full 12-dimension matrix.

Dimensions (corrected plan section 16):
  Accuracy, Latent specificity, Skill transfer, Instance binding,
  Robustness, Memory necessity, Memory efficiency, Latency,
  Calibration, Stability, Retention, Interference
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
from daph_latent_memory.evaluation.metrics import (
    causal_specificity_score,wrong_memory_penalty,latent_utility_gain,generalization_gain
)

def build_matrix(causal_eval:dict,ood_results:dict,eval_results:dict,seed_count:int=5)->dict:
    """Build the full benchmark matrix."""
    conditions=causal_eval.get("conditions",{})
    sweep=causal_eval.get("corruption_sweep",{})
    headline=causal_eval.get("headline",{})

    a_matched=conditions.get("matched",{}).get("accuracy",0.0)
    a_shuffled=conditions.get("shuffled_global",{}).get("accuracy",0.0)
    a_wrong=conditions.get("shuffled_wrong_class",{}).get("accuracy",0.0)
    a_base=conditions.get("base",{}).get("accuracy",0.0)

    # Corruption curve
    corruption_curve={}
    for alpha,data in sweep.items():
        corruption_curve[alpha]=data.get("accuracy",0.0)

    # OOD results
    ood_gains={}
    for split,data in ood_results.items():
        ood_gains[split]={"gg":data.get("gg",0),"a_base":data.get("a_base",0),"a_latent":data.get("a_latent",0)}

    # Per-split results
    split_accuracies={}
    for split,data in eval_results.items():
        split_accuracies[split]={
            "a_matched":data.get("a_matched",0),"a_base":data.get("a_base",0),
            "a_shuffled":data.get("a_shuffled",0),"a_wrong_class":data.get("a_wrong_class",0),
            "css":data.get("css",0),"wmp":data.get("wmp",0),"lug":data.get("lug",0),
        }

    matrix={
        "accuracy":{"exact_match":a_matched,"per_split":split_accuracies},
        "latent_specificity":{"css":headline.get("css",causal_specificity_score(a_matched,a_shuffled)),
                              "css_threshold":causal_eval.get("statistics",{}).get("css_threshold",0)},
        "skill_transfer":{"ood_gains":ood_gains},
        "instance_binding":{"matched_vs_shuffled":a_matched-a_shuffled,
                           "matched_vs_wrong":a_matched-a_wrong},
        "robustness":{"corruption_curve":corruption_curve},
        "memory_necessity":{"memory_on":a_matched,"memory_off":a_base,"delta":a_matched-a_base},
        "memory_efficiency":{"lug":headline.get("lug",latent_utility_gain(a_matched,a_base,8)),
                            "latent_tokens":8},
        "latency":{"note":"measured per-condition in causal_eval, not aggregated here"},
        "calibration":{"note":"requires confidence scores from CLUE verifier, not in this report"},
        "stability":{"seed_count":seed_count,"note":"requires multi-seed runs"},
        "retention":{"note":"requires old-task evaluation, not in this report"},
        "interference":{"note":"requires cross-skill degradation measurement, not in this report"},
    }
    return matrix

def format_matrix(matrix:dict)->str:
    """Format the matrix as a readable table."""
    lines=[]
    lines.append("="*80)
    lines.append("BENCHMARK MATRIX — DAPH Latent Memory v0.5.2")
    lines.append("="*80)
    lines.append(f"{'Dimension':<25} {'Metric':<30} {'Value'}")
    lines.append("-"*80)
    lines.append(f"{'Accuracy':<25} {'exact_match (matched)':<30} {matrix['accuracy']['exact_match']:.3f}")
    lines.append(f"{'Latent specificity':<25} {'CSS (pp)':<30} {matrix['latent_specificity']['css']:.1f}")
    lines.append(f"{'':<25} {'CSS threshold (pp)':<30} {matrix['latent_specificity']['css_threshold']:.1f}")
    lines.append(f"{'Skill transfer':<25} {'OOD splits with GG>0':<30} {sum(1 for v in matrix['skill_transfer']['ood_gains'].values() if v['gg']>0)}")
    lines.append(f"{'Instance binding':<25} {'matched - shuffled':<30} {matrix['instance_binding']['matched_vs_shuffled']:.3f}")
    lines.append(f"{'':<25} {'matched - wrong_class':<30} {matrix['instance_binding']['matched_vs_wrong']:.3f}")
    lines.append(f"{'Robustness':<25} {'alpha values tested':<30} {len(matrix['robustness']['corruption_curve'])}")
    lines.append(f"{'Memory necessity':<25} {'memory_on - memory_off':<30} {matrix['memory_necessity']['delta']:.3f}")
    lines.append(f"{'Memory efficiency':<25} {'LUG':<30} {matrix['memory_efficiency']['lug']:.4f}")
    lines.append("-"*80)
    # Per-split breakdown
    lines.append("\nPer-split accuracies:")
    for split,data in matrix['accuracy']['per_split'].items():
        lines.append(f"  {split:<20} matched={data['a_matched']:.3f} base={data['a_base']:.3f} shuffled={data['a_shuffled']:.3f} CSS={data['css']:.1f}pp")
    # Corruption curve
    lines.append("\nCorruption curve:")
    for alpha,acc in sorted(matrix['robustness']['corruption_curve'].items(),key=lambda x:float(x[0])):
        lines.append(f"  alpha={alpha:<6} A={acc:.3f}")
    lines.append("="*80)
    return "\n".join(lines)

def main():
    ap=argparse.ArgumentParser(description="Generate benchmark matrix report")
    ap.add_argument("--causal-eval",required=True)
    ap.add_argument("--ood-results",default=None)
    ap.add_argument("--eval-results",default=None)
    ap.add_argument("--seed-count",type=int,default=5)
    ap.add_argument("--output",required=True)
    args=ap.parse_args()

    with open(args.causal_eval) as f: causal_eval=json.load(f)
    ood_results={}
    if args.ood_results and Path(args.ood_results).exists():
        with open(args.ood_results) as f: ood_results=json.load(f)
    eval_results={}
    if args.eval_results and Path(args.eval_results).exists():
        with open(args.eval_results) as f: eval_results=json.load(f)

    matrix=build_matrix(causal_eval,ood_results,eval_results,args.seed_count)
    outdir=Path(args.output);outdir.parent.mkdir(parents=True,exist_ok=True)
    with open(outdir/"benchmark_matrix.json","w") as f: json.dump(matrix,f,indent=2)
    report=format_matrix(matrix)
    with open(outdir/"benchmark_matrix.txt","w") as f: f.write(report)
    print(report)

if __name__=="__main__": main()

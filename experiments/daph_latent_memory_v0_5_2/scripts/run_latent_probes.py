#!/usr/bin/env python
"""Latent probe runner — runs all diagnostic probes on saved latents.

Tests whether the latent predicts:
  Problem properties: operation_type, operand_1, operand_2, answer, difficulty
  Process properties: current_step, next_operation, intermediate_result

Measures: linear-probe, nonlinear-probe, nearest-neighbor, cosine clustering, CKA.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import torch
from analysis.latent_probe import run_all_probes

def extract_properties(examples:list[dict])->dict[str,np.ndarray]:
    """Extract property labels from example metadata."""
    import re
    properties={}
    # Operation type
    properties["operation_type"]=np.array([hash(ex.get("operation","unknown"))%100 for ex in examples])
    # Skill label
    properties["skill_label"]=np.array([hash(ex.get("skill_label","unknown"))%100 for ex in examples])
    # Difficulty (from template)
    properties["difficulty"]=np.array([ex.get("difficulty",1) for ex in examples])
    # Operand extraction from question
    operands=[]
    for ex in examples:
        nums=re.findall(r"-?\d+",ex.get("question",""))
        operands.append(int(nums[0]) if nums else 0)
    properties["operand_1"]=np.array(operands)
    # Answer
    answers=[]
    for ex in examples:
        nums=re.findall(r"-?\d+",ex.get("answer",""))
        answers.append(int(nums[-1]) if nums else 0)
    properties["answer"]=np.array(answers)
    return properties

def main():
    ap=argparse.ArgumentParser(description="Run latent probes")
    ap.add_argument("--latents",required=True,help="Path to saved latents .pt file")
    ap.add_argument("--examples",required=True,help="Path to examples.json metadata")
    ap.add_argument("--output",required=True)
    ap.add_argument("--seed",type=int,default=1337)
    args=ap.parse_args()

    # Load latents
    z=torch.load(args.latents,map_location="cpu",weights_only=False)
    if z.dim()==3: z=z.reshape(z.size(0),-1)  # flatten [N, tokens, dim] -> [N, tokens*dim]
    features=z.float().numpy()

    # Load examples
    with open(args.examples) as f: examples=json.load(f)
    properties=extract_properties(examples)

    # Run all probes
    results=run_all_probes(features,properties,seed=args.seed)

    # Save
    outdir=Path(args.output);outdir.parent.mkdir(parents=True,exist_ok=True)
    with open(outdir/"probe_results.json","w") as f: json.dump(results,f,indent=2)

    # Print summary
    print("\n" + "="*60)
    print("LATENT PROBE RESULTS")
    print("="*60)
    for prop,data in results.items():
        print(f"\n  {prop}:")
        for probe_name,probe_result in data.items():
            if "accuracy" in probe_result and probe_result["accuracy"] is not None:
                print(f"    {probe_name}: {probe_result['accuracy']:.3f}")
            elif "agreement" in probe_result and probe_result["agreement"] is not None:
                print(f"    {probe_name}: {probe_result['agreement']:.3f}")
            elif "v_measure" in probe_result and probe_result["v_measure"] is not None:
                print(f"    {probe_name}: v_measure={probe_result['v_measure']:.3f}")
            elif "cka" in probe_result and probe_result["cka"] is not None:
                print(f"    {probe_name}: cka={probe_result['cka']:.3f}")
    print("="*60)

if __name__=="__main__": main()

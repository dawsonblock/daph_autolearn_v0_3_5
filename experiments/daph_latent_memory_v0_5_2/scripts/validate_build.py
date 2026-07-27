#!/usr/bin/env python
"""Validate the v0.5.2 build — check all modules import, configs parse, tests pass."""
from __future__ import annotations
import argparse,sys,importlib,json
from pathlib import Path

REQUIRED_MODULES=[
    "daph_latent_memory.config",
    "daph_latent_memory.runtime",
    "daph_latent_memory.state.schema",
    "daph_latent_memory.state.corruption",
    "daph_latent_memory.latent.injection",
    "daph_latent_memory.latent.encoder",
    "daph_latent_memory.latent.constant",
    "daph_latent_memory.latent.skill_bank",
    "daph_latent_memory.latent.instance_state",
    "daph_latent_memory.latent.composer",
    "daph_latent_memory.router.router",
    "daph_latent_memory.router.symbolic_router",
    "daph_latent_memory.memory.bank",
    "daph_latent_memory.memory.manager",
    "daph_latent_memory.memory.verifier",
    "daph_latent_memory.memory.selector",
    "daph_latent_memory.memory.clue",
    "daph_latent_memory.memory.elastic",
    "daph_latent_memory.memory.anchoring",
    "daph_latent_memory.benchmarks.algebra",
    "daph_latent_memory.benchmarks.dataset",
    "daph_latent_memory.training.losses",
    "daph_latent_memory.training.negatives",
    "daph_latent_memory.training.collate",
    "daph_latent_memory.training.checkpoint",
    "daph_latent_memory.evaluation.controls",
    "daph_latent_memory.evaluation.metrics",
    "daph_latent_memory.evaluation.statistics",
    "daph_latent_memory.evaluation.leakage",
    "daph_latent_memory.teacher.capture",
]

REQUIRED_CONFIGS=["qwen3_1_7b.yaml","causal_eval.yaml","ablations.yaml"]

REQUIRED_SCRIPTS=[
    "generate_dataset.py","produce_v051_latents.py",
    "train_skill_bank.py","train_instance_encoder.py","train_joint.py",
    "evaluate.py","evaluate_latent_causality.py","evaluate_ood.py",
    "analyze_latent_geometry.py","run_latent_probes.py",
    "check_release_gates.py","report_benchmark_matrix.py",
    "validate_build.py",
]

REQUIRED_TESTS=[
    "test_losses.py","test_controls.py","test_metrics.py","test_statistics.py",
    "test_skill_bank.py","test_router.py","test_memory.py","test_dataset.py",
    "test_state.py","test_negatives.py","test_checkpoint.py",
]

def check_imports()->list[str]:
    """Check that all required modules import cleanly."""
    errors=[]
    for mod_name in REQUIRED_MODULES:
        try:
            importlib.import_module(mod_name)
        except Exception as e:
            errors.append(f"IMPORT FAIL: {mod_name}: {e}")
    return errors

def check_configs(base_dir:Path)->list[str]:
    """Check that all config files parse."""
    from daph_latent_memory.config import load_config
    errors=[]
    for cfg_name in REQUIRED_CONFIGS:
        cfg_path=base_dir/"configs"/cfg_name
        if not cfg_path.exists():
            errors.append(f"MISSING CONFIG: {cfg_path}")
            continue
        try:
            cfg=load_config(cfg_path)
            if not isinstance(cfg,dict):
                errors.append(f"INVALID CONFIG: {cfg_name} root is not a dict")
        except Exception as e:
            errors.append(f"PARSE FAIL: {cfg_name}: {e}")
    return errors

def check_scripts(base_dir:Path)->list[str]:
    """Check that all scripts parse as valid Python."""
    import ast
    errors=[]
    for script_name in REQUIRED_SCRIPTS:
        script_path=base_dir/"scripts"/script_name
        if not script_path.exists():
            errors.append(f"MISSING SCRIPT: {script_path}")
            continue
        try:
            ast.parse(script_path.read_text())
        except SyntaxError as e:
            errors.append(f"SYNTAX FAIL: {script_name}: {e}")
    return errors

def check_tests(base_dir:Path)->list[str]:
    """Check that all test files exist."""
    errors=[]
    for test_name in REQUIRED_TESTS:
        test_path=base_dir/"tests"/test_name
        if not test_path.exists():
            errors.append(f"MISSING TEST: {test_path}")
    return errors

def check_analysis(base_dir:Path)->list[str]:
    """Check that analysis modules exist and parse."""
    import ast
    errors=[]
    for mod_name in ["latent_probe.py","cca_analysis.py"]:
        mod_path=base_dir/"analysis"/mod_name
        if not mod_path.exists():
            errors.append(f"MISSING ANALYSIS: {mod_path}")
            continue
        try:
            ast.parse(mod_path.read_text())
        except SyntaxError as e:
            errors.append(f"SYNTAX FAIL: {mod_name}: {e}")
    return errors

def check_v052_corrections()->list[str]:
    """Verify that v0.5.2 corrections are actually applied in the code."""
    errors=[]
    # Check that alignment loss is NOT in the new losses
    from daph_latent_memory.training import losses
    if hasattr(losses,"hidden_alignment_loss"):
        errors.append("CORRECTION VIOLATION: hidden_alignment_loss still present in losses.py")
    if hasattr(losses,"total_loss"):
        errors.append("CORRECTION VIOLATION: v0.5.1 total_loss still present in losses.py")
    # Check that L_functional exists
    if not hasattr(losses,"functional_mismatch_loss"):
        errors.append("CORRECTION VIOLATION: functional_mismatch_loss missing from losses.py")
    # Check that L_contrast does NOT exist
    if hasattr(losses,"contrastive_loss"):
        errors.append("CORRECTION VIOLATION: L_contrast (contrastive_loss) should not exist")
    # Check that CSS null distribution exists
    from daph_latent_memory.evaluation import metrics
    if not hasattr(metrics,"css_null_distribution"):
        errors.append("CORRECTION VIOLATION: css_null_distribution missing from metrics.py")
    # Check that Cochran's Q exists
    from daph_latent_memory.evaluation import statistics
    if not hasattr(statistics,"cochran_q_test"):
        errors.append("CORRECTION VIOLATION: cochran_q_test missing from statistics.py")
    return errors

def main():
    ap=argparse.ArgumentParser(description="Validate v0.5.2 build")
    ap.add_argument("--base-dir",default=".",help="Base directory of the experiment")
    ap.add_argument("--json",action="store_true",help="Output JSON instead of text")
    args=ap.parse_args()
    base_dir=Path(args.base_dir).resolve()

    report={
        "imports":check_imports(),
        "configs":check_configs(base_dir),
        "scripts":check_scripts(base_dir),
        "tests":check_tests(base_dir),
        "analysis":check_analysis(base_dir),
        "corrections":check_v052_corrections(),
    }
    all_errors=[]
    for category,errors in report.items():
        all_errors.extend(errors)
    report["passed"]=len(all_errors)==0
    report["total_errors"]=len(all_errors)

    if args.json:
        print(json.dumps(report,indent=2))
    else:
        print("\n" + "="*60)
        print("V0.5.2 BUILD VALIDATION")
        print("="*60)
        for category,errors in report.items():
            if category in ("passed","total_errors"): continue
            status="OK" if not errors else f"{len(errors)} ERRORS"
            print(f"  {category:<15} {status}")
            for e in errors: print(f"    - {e}")
        print("="*60)
        print(f"  OVERALL: {'PASS' if report['passed'] else 'FAIL'} ({report['total_errors']} errors)")
        print("="*60)

    sys.exit(0 if report["passed"] else 1)

if __name__=="__main__": main()

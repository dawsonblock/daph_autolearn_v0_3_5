"""Benchmark integrity: semantic-family-aware splitting + leak detection
(AutoLearn v2, Phase 10).

Replaces random prompt splitting with splits that respect
generator/template family identity. A template family may only exist in
one of train / validation / test. Leak detection flags:

* exact prompt duplication across splits
* normalized prompt duplication
* generator-family overlap
* template overlap
* task fingerprint overlap

Qualification fails when leakage exists. The checker emits a
:class:`LeakageReport`.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from daph_learning.autolearn_v2.experience import fingerprint_task


def normalize_prompt(text: str) -> str:
    """Lowercase, collapse whitespace, strip non-alphanumeric for dedup."""
    s = str(text).lower()
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"[^a-z0-9 ]", "", s)
    return s


def prompt_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class LeakageReport:
    """Result of a leak-detection scan over a set of splits."""

    has_leak: bool
    exact_duplicates: list[tuple[str, str, str]] = field(default_factory=list)
    normalized_duplicates: list[tuple[str, str, str]] = field(default_factory=list)
    family_overlap: list[tuple[str, str, str]] = field(default_factory=list)
    template_overlap: list[tuple[str, str, str]] = field(default_factory=list)
    fingerprint_overlap: list[tuple[str, str, str]] = field(default_factory=list)
    split_sizes: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "has_leak": self.has_leak,
            "exact_duplicates": [list(t) for t in self.exact_duplicates],
            "normalized_duplicates": [list(t) for t in self.normalized_duplicates],
            "family_overlap": [list(t) for t in self.family_overlap],
            "template_overlap": [list(t) for t in self.template_overlap],
            "fingerprint_overlap": [list(t) for t in self.fingerprint_overlap],
            "split_sizes": dict(self.split_sizes),
        }

    def summary(self) -> str:
        if not self.has_leak:
            return "no leakage detected"
        parts = []
        if self.exact_duplicates:
            parts.append(f"{len(self.exact_duplicates)} exact duplicate(s)")
        if self.normalized_duplicates:
            parts.append(f"{len(self.normalized_duplicates)} normalized duplicate(s)")
        if self.family_overlap:
            parts.append(f"{len(self.family_overlap)} family overlap(s)")
        if self.template_overlap:
            parts.append(f"{len(self.template_overlap)} template overlap(s)")
        if self.fingerprint_overlap:
            parts.append(f"{len(self.fingerprint_overlap)} fingerprint overlap(s)")
        return "leakage: " + "; ".join(parts)


def _extract_field(task: Mapping[str, Any], key: str) -> str | None:
    val = task.get(key)
    if val is None:
        return None
    if isinstance(val, (list, tuple, set)):
        return ",".join(sorted(map(str, val)))
    return str(val)


def detect_leakage(
    splits: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    prompt_field: str = "specification",
    family_field: str = "generator_family",
    template_field: str = "template_id",
) -> LeakageReport:
    """Scan a mapping of ``{split_name: [tasks]}`` for cross-split leakage.

    A task is identified by its ``task_id``. Leaks are reported as tuples
    ``(task_id_a, task_id_b, detail)`` where the two task ids come from
    different splits.
    """
    split_names = list(splits.keys())
    split_sizes = {name: len(tasks) for name, tasks in splits.items()}

    # Index per split.
    by_exact: dict[str, dict[str, str]] = {}  # split -> {hash: task_id}
    by_norm: dict[str, dict[str, str]] = {}
    by_family: dict[str, dict[str, str]] = {}
    by_template: dict[str, dict[str, str]] = {}
    by_fp: dict[str, dict[str, str]] = {}

    for name, tasks in splits.items():
        by_exact[name] = {}
        by_norm[name] = {}
        by_family[name] = {}
        by_template[name] = {}
        by_fp[name] = {}
        for task in tasks:
            tid = str(task.get("task_id", ""))
            prompt = str(task.get(prompt_field, ""))
            by_exact[name][prompt_hash(prompt)] = tid
            by_norm[name][prompt_hash(normalize_prompt(prompt))] = tid
            fam = _extract_field(task, family_field)
            if fam:
                by_family[name][fam] = tid
            tmpl = _extract_field(task, template_field)
            if tmpl:
                by_template[name][tmpl] = tid
            by_fp[name][fingerprint_task(task)] = tid

    exact_dups: list[tuple[str, str, str]] = []
    norm_dups: list[tuple[str, str, str]] = []
    fam_overlap: list[tuple[str, str, str]] = []
    tmpl_overlap: list[tuple[str, str, str]] = []
    fp_overlap: list[tuple[str, str, str]] = []

    for i, a in enumerate(split_names):
        for b in split_names[i + 1:]:
            for h, tid_a in by_exact[a].items():
                if h in by_exact[b]:
                    exact_dups.append((tid_a, by_exact[b][h], f"{a}/{b}"))
            for h, tid_a in by_norm[a].items():
                if h in by_norm[b]:
                    norm_dups.append((tid_a, by_norm[b][h], f"{a}/{b}"))
            for fam, tid_a in by_family[a].items():
                if fam in by_family[b]:
                    fam_overlap.append((tid_a, by_family[b][fam], f"{a}/{b}:{fam}"))
            for tmpl, tid_a in by_template[a].items():
                if tmpl in by_template[b]:
                    tmpl_overlap.append((tid_a, by_template[b][tmpl], f"{a}/{b}:{tmpl}"))
            for fp, tid_a in by_fp[a].items():
                if fp in by_fp[b]:
                    fp_overlap.append((tid_a, by_fp[b][fp], f"{a}/{b}"))

    has_leak = bool(exact_dups or norm_dups or fam_overlap or tmpl_overlap or fp_overlap)
    return LeakageReport(
        has_leak=has_leak,
        exact_duplicates=exact_dups,
        normalized_duplicates=norm_dups,
        family_overlap=fam_overlap,
        template_overlap=tmpl_overlap,
        fingerprint_overlap=fp_overlap,
        split_sizes=split_sizes,
    )


def family_aware_split(
    tasks: Sequence[Mapping[str, Any]],
    *,
    family_field: str = "generator_family",
    train_fraction: float = 0.7,
    val_fraction: float = 0.15,
    seed: int = 42,
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    """Split tasks so each family appears in exactly one split.

    Families are assigned to train/val/test whole. This guarantees no
    family overlap. The split is deterministic given ``seed``.
    """
    import random as _r
    rng = _r.Random(seed)
    families: dict[str, list[Mapping[str, Any]]] = {}
    for task in tasks:
        fam = _extract_field(task, family_field) or "__no_family__"
        families.setdefault(fam, []).append(task)
    fam_names = sorted(families.keys())
    rng.shuffle(fam_names)
    n = len(fam_names)
    n_train = int(round(n * train_fraction))
    n_val = int(round(n * val_fraction))
    train_fams = fam_names[:n_train]
    val_fams = fam_names[n_train:n_train + n_val]
    test_fams = fam_names[n_train + n_val:]
    train = [t for f in train_fams for t in families[f]]
    val = [t for f in val_fams for t in families[f]]
    test = [t for f in test_fams for t in families[f]]
    return train, val, test


__all__ = [
    "LeakageReport",
    "detect_leakage",
    "family_aware_split",
    "normalize_prompt",
    "prompt_hash",
]

from __future__ import annotations
import random
from dataclasses import dataclass
from ..state.schema import MathExample

# =============================================================================
# Hard Negative Sampling for L_functional
#
# Categories (per the corrected plan, section 7.2):
#   1. same operation, different numbers
#   2. similar expression structure (same depth, different operators)
#   3. same answer, different derivation
#   4. same operands, different operator
#   5. numerically close problems (operands within 10%)
#
# K=3 negatives per step are sampled uniformly from these 5 categories.
# =============================================================================

@dataclass
class NegativePair:
    anchor:MathExample
    negative:MathExample
    category:str

def _extract_operands(question:str)->list[int]:
    import re
    return [int(x) for x in re.findall(r"-?\d+",question)]

def _extract_answer(answer:str)->int|None:
    import re
    nums=re.findall(r"-?\d+",answer)
    return int(nums[-1]) if nums else None

def _same_operation(a:MathExample,b:MathExample)->bool:
    return a.state.operation==b.state.operation

def _same_operands(a:MathExample,b:MathExample)->bool:
    return _extract_operands(a.question)==_extract_operands(b.question)

def _same_answer(a:MathExample,b:MathExample)->bool:
    ai,bi=_extract_answer(a.answer),_extract_answer(b.answer)
    return ai is not None and bi is not None and ai==bi

def _same_depth(a:MathExample,b:MathExample)->bool:
    return a.difficulty==b.difficulty

def _numerically_close(a:MathExample,b:MathExample,tolerance:float=0.1)->bool:
    oa,ob=_extract_operands(a.question),_extract_operands(b.question)
    if not oa or not ob or len(oa)!=len(ob): return False
    for x,y in zip(oa,ob):
        if x==0 or y==0:
            if abs(x-y)>5: return False
        elif abs(x-y)/max(abs(x),abs(y))>tolerance: return False
    return True

def _different_numbers(a:MathExample,b:MathExample)->bool:
    return not _same_operands(a,b)

def _different_operator(a:MathExample,b:MathExample)->bool:
    return a.state.operation!=b.state.operation

def sample_negatives(
    anchor:MathExample,
    pool:list[MathExample],
    k:int=3,
    seed:int=1337,
)->list[NegativePair]:
    """Sample K hard negatives for a given anchor example.

    Uniformly samples from 5 categories. If a category has no candidates,
    falls back to random negatives from the pool (excluding the anchor).
    """
    rng=random.Random(seed)
    categories=["same_op_diff_nums","same_depth_diff_op","same_answer_diff_derivation","same_operands_diff_op","numerically_close"]
    pairs:list[NegativePair]=[]
    chosen_cats=rng.sample(categories,min(k,len(categories)))
    results:list[NegativePair]=[]
    for cat in chosen_cats:
        candidates:list[MathExample]=[]
        for ex in pool:
            if ex.example_id==anchor.example_id: continue
            if cat=="same_op_diff_nums" and _same_operation(anchor,ex) and _different_numbers(anchor,ex): candidates.append(ex)
            elif cat=="same_depth_diff_op" and _same_depth(anchor,ex) and _different_operator(anchor,ex): candidates.append(ex)
            elif cat=="same_answer_diff_derivation" and _same_answer(anchor,ex) and not _same_operands(anchor,ex): candidates.append(ex)
            elif cat=="same_operands_diff_op" and _same_operands(anchor,ex) and _different_operator(anchor,ex): candidates.append(ex)
            elif cat=="numerically_close" and _numerically_close(anchor,ex) and not _same_operands(anchor,ex): candidates.append(ex)
        if candidates:
            neg=rng.choice(candidates)
            results.append(NegativePair(anchor,neg,cat))
        else:
            # Fallback: random non-anchor example
            fallback_pool=[ex for ex in pool if ex.example_id!=anchor.example_id]
            if fallback_pool:
                neg=rng.choice(fallback_pool)
                results.append(NegativePair(anchor,neg,"random_fallback"))
    return results[:k]

def sample_same_skill_pairs(
    examples:list[MathExample],
    seed:int=1337,
)->list[tuple[MathExample,MathExample]]:
    """Sample pairs of examples with the same skill label for L_invariance."""
    rng=random.Random(seed)
    by_skill:dict[str,list[MathExample]]={}
    for ex in examples:
        label=ex.skill_label or ex.state.operation
        by_skill.setdefault(label,[]).append(ex)
    pairs:list[tuple[MathExample,MathExample]]=[]
    for label,group in by_skill.items():
        if len(group)<2: continue
        shuffled=list(group);rng.shuffle(shuffled)
        for i in range(0,len(shuffled)-1,2):
            pairs.append((shuffled[i],shuffled[i+1]))
    return pairs

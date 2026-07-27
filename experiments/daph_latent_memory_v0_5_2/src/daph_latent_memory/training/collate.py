from __future__ import annotations
from ..state.schema import MathExample

def tokenize_batch(examples:list[MathExample],tokenizer,max_prompt:int,max_answer:int):
    prompts=[f"Question:\n{x.question}\nAnswer:" for x in examples];states=[x.state.process_text() for x in examples];answers=[x.answer for x in examples]
    p=tokenizer(prompts,return_tensors="pt",padding=True,truncation=True,max_length=max_prompt);s=tokenizer(states,return_tensors="pt",padding=True,truncation=True,max_length=max_prompt);a=tokenizer(answers,return_tensors="pt",padding=True,truncation=True,max_length=max_answer,add_special_tokens=False);return p,s,a

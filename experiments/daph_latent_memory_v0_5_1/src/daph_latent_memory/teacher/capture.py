from __future__ import annotations
import torch

def select_layer_index(num_hidden_states:int,fraction:float)->int:
    if not 0.0<fraction<=1.0: raise ValueError("fraction must be in (0, 1]")
    return max(1,min(num_hidden_states-1,round((num_hidden_states-1)*fraction)))

@torch.no_grad()
def capture_teacher_state(model,tokenizer,question:str,process_text:str,layer_fraction:float,device:torch.device)->torch.Tensor:
    text=f"Question:\n{question}\n\nKnown process state:\n{process_text}\n\nContinue solving the problem."
    tok=tokenizer(text,return_tensors="pt",truncation=True);tok={k:v.to(device) for k,v in tok.items()};out=model(**tok,output_hidden_states=True,use_cache=False);idx=select_layer_index(len(out.hidden_states),layer_fraction);h=out.hidden_states[idx];mask=tok["attention_mask"].unsqueeze(-1);return (h*mask).sum(dim=1)/mask.sum(dim=1).clamp_min(1)

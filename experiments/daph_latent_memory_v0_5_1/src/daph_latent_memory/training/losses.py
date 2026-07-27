from __future__ import annotations
import torch
from torch.nn import functional as F

def hidden_alignment_loss(student:torch.Tensor,teacher:torch.Tensor)->torch.Tensor:
    student=F.normalize(student.float(),dim=-1);teacher=F.normalize(teacher.float(),dim=-1);return 1.0-(student*teacher).sum(dim=-1).mean()

def total_loss(answer_loss:torch.Tensor,align_loss:torch.Tensor,answer_weight:float,alignment_weight:float)->torch.Tensor:return answer_weight*answer_loss+alignment_weight*align_loss

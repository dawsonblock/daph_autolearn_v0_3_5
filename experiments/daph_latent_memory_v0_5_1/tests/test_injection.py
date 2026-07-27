import torch
from torch import nn
from daph_latent_memory.latent.injection import LatentInjector

class DummyModel(nn.Module):
    def __init__(self):super().__init__();self.emb=nn.Embedding(32,8).to(dtype=torch.float16)
    def get_input_embeddings(self):return self.emb

def test_injector_casts_latents_and_masks_answer_padding():
    model=DummyModel();injector=LatentInjector(model);prompt_ids=torch.tensor([[1,2,3]]);prompt_mask=torch.tensor([[1,1,1]]);answer_ids=torch.tensor([[4,5,0]]);answer_mask=torch.tensor([[1,1,0]]);latents=torch.randn(1,2,8,dtype=torch.float32);batch=injector.build(prompt_ids,prompt_mask,answer_ids,answer_mask,latents);assert batch.inputs_embeds.dtype==torch.float16;assert batch.labels.tolist()==[[-100,-100,-100,-100,-100,4,5,-100]];assert batch.attention_mask.tolist()==[[1,1,1,1,1,1,1,0]]

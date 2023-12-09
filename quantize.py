"""
Draft of symmetric simulated quantization for pretrained GPT2 Model
Embedding:
  (wte): Embedding(50257, 768)
  (wpe): Embedding(1024, 768)

Structure of GPT2 Block:
12 x GPT2Block(
      (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
      (attn): GPT2Attention(
        (c_attn): Conv1D()
        (c_proj): Conv1D()
        (attn_dropout): Dropout(p=0.1, inplace=False)
        (resid_dropout): Dropout(p=0.1, inplace=False)
      )
      (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
      (mlp): GPT2MLP(
        (c_fc): Conv1D()
        (c_proj): Conv1D()
        (act): NewGELUActivation()
        (dropout): Dropout(p=0.1, inplace=False)
      )

Approach for each GPT2 Model Block:
- obtain input from previous block and dequantize to FP32 (unless we are in first GPT2 Model Block)
- conduct all computations within block in FP32
- quantize output for next block 
"""

from transformers import GPT2Model
from transformers.models.gpt2.modeling_gpt2 import GPT2Block
import torch
import torch.nn as nn
import os

from model import GPTConfig, GPT

def print_model_size(mdl):
    torch.save(mdl.state_dict(), "./tmp.pt")
    print("%.2f MB" %(os.path.getsize("./tmp.pt")/1e6))
    os.remove('tmp.pt')


def symm_quantize(tensor, scale):
  return torch.clip(torch.round(tensor / scale), -127, 127).to(torch.int8)

def symm_dequantize(tensor, scale):
    return tensor.float() * scale

class CustomGPT2Model(GPT):
    def __init__(self, config, state_dict):
        super().__init__(config)
        self.scales = {}
        self.orig_quant_weights = {}
        self.load_state_dict(state_dict)

        with torch.no_grad():
          for name, param in self.named_parameters():
             if 'h.' in name and name.endswith('.weight'):
                param.requires_grad = False 
                scale = param.abs().max() / 127
                self.scales[name] = scale
                param.data = symm_quantize(param, scale)
                self.orig_quant_weights[name] = param.data.clone().to("cuda")

    def forward(self, *inputs, **kwargs):
        # Dequantize the weights before the forward pass
        with torch.no_grad():
            for name, param in self.named_parameters():
                if 'h.' in name and name.endswith('.weight'):
                    scale = self.scales[name]
                    param.data = symm_dequantize(param, scale)

        outputs = super().forward(*inputs, **kwargs)

        with torch.no_grad():
            for name, param in self.named_parameters():
                if 'h.' in name and name.endswith('.weight'):
                    param.data = self.orig_quant_weights[name]

        return outputs
      





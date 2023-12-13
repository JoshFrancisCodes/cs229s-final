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

def print_model_size(mdl):
    torch.save(mdl.state_dict(), "./tmp.pt")
    print("%.2f MB" %(os.path.getsize("./tmp.pt")/1e6))
    os.remove('tmp.pt')


model = GPT2Model.from_pretrained('gpt2')
print("SIZE: ", print_model_size(model))


def symm_quantize(tensor, scale):
  return torch.clip(torch.round(tensor / scale), -127, 127).to(torch.int8)

def symm_dequantize(tensor, scale):
    return tensor.float() * scale

scales = {}
with torch.no_grad():
  for name, param in model.named_parameters():
    if name.endswith('.weight'):
      param.requires_grad = False 
      scale = param.abs().max() / 127
      scales[name] = scale
      param.data = symm_quantize(param, scale)

class CustomGPT2Model(GPT2Model):
    def __init__(self, config, scales):
        super().__init__(config)
        self.scales = scales

    def forward(self, *inputs, **kwargs):
        # Dequantize the weights before the forward pass
        with torch.no_grad():
            for name, param in self.named_parameters():
                if name.endswith('.weight'):
                    scale = self.scales[name]
                    dequantized_param = symm_dequantize(param, scale)
                    param.data = dequantized_param

        return super().forward(*inputs, **kwargs)

with torch.no_grad():
  for name, param in model.named_parameters():
    print(name, param)
print("SIZE: ", print_model_size(CustomGPT2Model(model.config, scales)))


raise Exception("TEST")

quantized_weights_path = "quantized_weights.pt"
torch.save(quantized_weights, quantized_weights_path)

class QuantizedInferenceModel(GPT2Model):
    def __init__(self, config, quantized_weights_path):
        super().__init__(config)
        self.quantized_weights = torch.load(quantized_weights_path)

    def dequantize_weight(self, name):
        quantized_weight = self.quantized_weights[name]
        scale = self.quantized_weights[f"{name}_scale"]
        return symm_dequantize(quantized_weight, scale)  # Convert back to float32

    def forward(self, *input, **kwargs):
        # Replace weights in self with dequantized weights
        with torch.no_grad():
            for name, _ in self.named_parameters():
                if name in self.quantized_weights:
                    dequantized_weight = self.dequantize_weight(name)
                    param = dict(self.named_parameters())[name]
                    param.data.copy_(dequantized_weight)

        return super().forward(*input, **kwargs)

quant_model = QuantizedInferenceModel(model.config, quantized_weights_path)
with torch.no_grad():
  for name, param in model.named_parameters():
    print(name, param)
quant_model.eval()

raise Exception("Test")


class QuantizedEmbedding(nn.Embedding):
  def __init__(self, embedding_layer):
    super(QuantizedEmbedding, self).__init__(num_embeddings=embedding_layer.num_embeddings, embedding_dim=embedding_layer.embedding_dim, padding_idx=embedding_layer.padding_idx)
    weight = embedding_layer.weight.detach()
    self.scale = weight.abs().max() / 127
    self.quantized_weight = symm_quantize(weight, embedding_layer.scale)

  def forward(self, input_ids):
    dequantized_weight = symm_dequantize(self.quantized_weight, self.scale)
    return nn.functional.embedding(input_ids, dequantized_weight, self.padding_idx)


class QuantizedGPT2Block(GPT2Block):
  def __init__(self, config, is_first=False):
    super().__init__(config)
    self.is_first = is_first # tracking if we are at the first GPTBlock to det if we need to quantize or not

  def forward(self, hidden_states, layer_past=None, attention_mask=None, head_mask=None, use_cache=True, output_attentions=False):
    if not self.is_first:
      scale = hidden_states.abs().max() / 127
      hidden_states = symm_dequantize(hidden_states, scale)
    
    # self.attn derived from https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py 
    attn_outputs = self.attn(
      self.ln_1(hidden_states),
      layer_past=layer_past,
      attention_mask=attention_mask,
      head_mask=head_mask,
      use_cache=use_cache,
      output_attentions=output_attentions
    )

    attn = attn_outputs[0]
    mlp = self.mlp(self.ln_2(attn))

    output_scale = mlp.abs().max() / 127
    mlp_quantized = symm_quantize(mlp, output_scale)
    return [mlp_quantized] + attn_outputs[1:]

quant_wte = QuantizedEmbedding(model.wte)
print(quant_wte)
print(quant_wte.weight)



for i, block in enumerate(model.h):
  is_first = (i == 0)
  quantized_block = QuantizedGPT2Block(model.config, is_first)
  quantized_block.load_state_dict(block.state_dict())
  model.h[i] = quantized_block

print(model)



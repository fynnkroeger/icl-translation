from llm_transparency_tool.models.tlens_model import TransformerLensTransparentLlm
import llm_transparency_tool.routes.contributions as contributions
from torch import bfloat16, no_grad
import transformers

model_name = "mistralai/Mistral-7B-Instruct-v0.1"
hf_model = transformers.AutoModel.from_pretrained(
    model_name, device_map="auto", torch_dtype=bfloat16
)
hf_tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
model = TransformerLensTransparentLlm(
    model_name, hf_model=hf_model, tokenizer=hf_tokenizer, device="gpu"
)
model.run(["When Mary and John went to the store, John gave a drink to"])  # No instruct format
print(model.tokens())  # always loads checkpoint twice?
print(tokenizer.batch_decode(model.tokens()))
exit()
batch_i = 0
n_layers = 32
n_tokens = model.tokens()[batch_i].shape[0]
layer = 15
c_attn, c_resid_attn = contributions.get_attention_contributions(
    resid_pre=model.residual_in(layer)[batch_i].unsqueeze(0),
    resid_mid=model.residual_after_attn(layer)[batch_i].unsqueeze(0),
    decomposed_attn=model.decomposed_attn(batch_i, layer).unsqueeze(0),
)
print(c_attn, c_resid_attn)
exit()
"""
build_full_graph
        c_attn, c_resid_attn = contributions.get_attention_contributions(
            resid_pre=model.residual_in(layer)[batch_i].unsqueeze(0),
            resid_mid=model.residual_after_attn(layer)[batch_i].unsqueeze(0),
            decomposed_attn=model.decomposed_attn(batch_i, layer).unsqueeze(0),
        )

"""

import transformer_lens
from torch import bfloat16, no_grad
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "mistralai/Mistral-7B-Instruct-v0.1"

# model = AutoModelForCausalLM.from_pretrained(
#     model_name, device_map="auto", torch_dtype=bfloat16, attn_implementation="eager"
# )


tlens_model = transformer_lens.HookedTransformer.from_pretrained(
    model_name=model_name,
    # hf_model=model,
    fold_ln=False,  # Keep layer norm where it is.
    center_writing_weights=False,
    center_unembed=False,
    device="cuda",
    # n_devices=n_devices,
    dtype=bfloat16,
)
tlens_model.eval()

tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
tlens_model.set_tokenizer(tokenizer, default_padding_side="left")
tlens_model.set_use_attn_result(True)
tlens_model.set_use_attn_in(False)
tlens_model.set_use_split_qkv_input(False)

input_text = "Write a story about horses."
model_inputs = tokenizer.apply_chat_template(
    [{"role": "user", "content": input_text}],
    padding=True,
    return_tensors="pt",
    tokenize=True,
    add_generation_prompt=True,
)
print(tokenizer.batch_decode(model_inputs[0]))


model_inputs.to(tlens_model.cfg.device)

with no_grad():
    logits, cache = tlens_model.run_with_cache(model_inputs)

output_tokens = logits.argmax(dim=-1)
# really bad resluts, half garbage tokens, some make kind of sense
# Decode the output tokens
output_text = tokenizer.batch_decode(output_tokens[0], skip_special_tokens=False)

print("Input:", input_text)
print("Output:", output_text)

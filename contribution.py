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

input_text = "Translate the following English text to French: 'Hello, how are you?'"
model_inputs = tokenizer.apply_chat_template(
    [{"role": "user", "content": input_text}],
    padding=True,
    return_tensors="pt",
    tokenize=True,
    add_generation_prompt=True,
)
model_inputs.to(tlens_model.cfg.device)

with no_grad():
    logits, cache = tlens_model.run_with_cache(model_inputs)

output_tokens = logits.argmax(dim=-1)

# Decode the output tokens
output_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)

print("Input:", input_text)
print("Output:", output_text)

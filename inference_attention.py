import numpy as np
from ALMA.utils.utils import LANG_TABLE
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch import bfloat16, float16
import pickle

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
model_name = "mistralai/Mistral-7B-Instruct-v0.2"
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
tokenizer.pad_token_id = tokenizer.eos_token_id
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=bfloat16, attn_implementation="eager")
batch_size = 1
max_batches = 4

n_shots = 1
lang_pair = "de-en"

source_lang, target_lang = lang_pair.split("-")
prompt = f"Translate this from {LANG_TABLE[source_lang]} to {LANG_TABLE[target_lang]}:\n"
with open(f"wmtdatasets/wmt23_{lang_pair}.json") as f:
    few_shot_dataset = json.load(f)

few_shot_prompt = []
for i in range(n_shots):
    sample = few_shot_dataset[i]
    few_shot_prompt.append(dict(role="user", content=prompt+sample["source"]))
    few_shot_prompt.append(dict(role="assistant", content=sample["target"]))

with open(f"wmtdatasets/wmt22_{lang_pair}.json") as f:
    test_dataset = json.load(f)

output = []
attentions = []
tokens = []
for i in range(0, len(test_dataset), batch_size):
    if i/batch_size >= max_batches:
        break
    batch = test_dataset[i:i + batch_size]
    messages2d = []
    for sample in batch:
        sample_prompt = [{"role": "user", "content": prompt+sample["source"]}]
        messages2d.append(few_shot_prompt+sample_prompt)
    model_inputs = tokenizer.apply_chat_template(messages2d, padding=True, return_tensors="pt", tokenize=True, add_generation_prompt=True)
    input_seq_len = model_inputs.shape[1]
    print(input_seq_len)
    generation = model.generate(model_inputs.to("cuda"), max_new_tokens=300, pad_token_id=tokenizer.eos_token_id, output_attentions=True, return_dict_in_generate=True)
    batch_size_actual, seq_len = generation.sequences.shape
    print(seq_len)
    new_tokens = seq_len-input_seq_len
    num_heads = 32
    layers = len(generation.attentions[0])
    unified_attention = np.zeros((layers, batch_size_actual, num_heads, seq_len, seq_len))

    for layer_num, layer in enumerate(generation.attentions[0]):
        layer = layer.detach().cpu().to(float16).squeeze().numpy()
        unified_attention[layer_num, :, :, :input_seq_len, :input_seq_len] = layer

    for batch_index in range(batch_size_actual):
        for token_index, token in enumerate(generation.attentions[1:]):
            for layer_num, layer in enumerate(token):
                layer = layer.detach().cpu().to(float16).numpy().squeeze()
                if batch_size_actual > 1:
                    layer = layer[batch_index]
                unified_attention[layer_num, batch_index, :, token_index+input_seq_len, :token_index+input_seq_len+1] = layer
    # np.save(f"outputs/{lang_pair}_{n_shots}shot_attention2.npy", unified_attention)
    attentions.append(unified_attention)
    tokens_BL = [[] for _ in range(batch_size_actual)]
    for s in range(seq_len):
        tokens_B = tokenizer.batch_decode(generation.sequences[:, s])
        for b, x in enumerate(tokens_B):
            tokens_BL[b].append(x)
    tokens.extend(tokens_BL)

    print(tokenizer.batch_decode(generation.sequences))
    assert len(set(len(outer) for outer in generation.attentions)) == 1  # all of len num_layers
    for outer_index, outer in enumerate(generation.attentions):
        assert len(set(tuple(inner.shape) for inner in outer)) == 1
        # print(outer_index, outer[0].shape)

    decoded = tokenizer.batch_decode(generation.sequences, skip_special_tokens=True)
    translations = [x.split("[/INST]")[-1].lstrip() for x in decoded]
    # todo better cutting
    for translation, sample in zip(translations, batch):
        output.append(dict(source=sample["source"], target=sample["target"], translation=translation))

    # write here so we dont loose progress
    with open(f"outputs/{lang_pair}_{n_shots}shot_handwritten.json", "w") as f:
        json.dump(output, f, indent=1)
    with open(f"outputs/{lang_pair}_{n_shots}shot_tokens.json", "w") as f:
        json.dump(tokens, f, indent=1)
    with open(f"outputs/{lang_pair}_{n_shots}shot_attention.pickle", "wb") as f:
        pickle.dump(attentions, f)

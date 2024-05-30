# from translate_instruct import translate_batch
from ALMA.utils.utils import LANG_TABLE
import json
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch import bfloat16

model_name = "mistralai/Mistral-7B-Instruct-v0.2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token_id = tokenizer.eos_token_id
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=bfloat16)
batch_size = 128
max_batches = 1000

n_shots = 1
lang_pair = "de-en"

source_lang, target_lang = lang_pair.split("-")
prompt = f"Translate this from {LANG_TABLE[source_lang]} to {LANG_TABLE[target_lang]}:\n"
with open(f"wmtdatasets/handwritten_{lang_pair}.json") as f:
    few_shot_dataset = json.load(f)

few_shot_prompt = []
for i in range(n_shots):
    sample = few_shot_dataset[i]
    few_shot_prompt.append(dict(role="user", content=prompt+sample["source"]))
    few_shot_prompt.append(dict(role="assistant", content=sample["target"]))

with open(f"wmtdatasets/wmt22_{lang_pair}.json") as f:
    test_dataset = json.load(f)

output = []
for i in tqdm(range(0, len(test_dataset), batch_size)):
    if i/batch_size >= max_batches:
        break
    batch = test_dataset[i:i + batch_size]
    messages2d = []
    for sample in batch:
        sample_prompt = [{"role": "user", "content": prompt+sample["source"]}]
        messages2d.append(few_shot_prompt+sample_prompt)
    model_inputs = tokenizer.apply_chat_template(messages2d, padding=True, return_tensors="pt", tokenize=True, add_generation_prompt=True)
    # print(model_inputs)
    generation = model.generate(model_inputs.to("cuda"), max_new_tokens=512, pad_token_id=tokenizer.eos_token_id, return_dict_in_generate=True)
    # tuple generation["attentions"]
    decoded = tokenizer.batch_decode(generation.sequences, skip_special_tokens=True,)
    translations = [x.split("[/INST]")[-1].lstrip() for x in decoded]
    for translation, sample in zip(translations, batch):
        output.append(dict(source=sample["source"], target=sample["target"], translation=translation))

    with open(f"outputs/{lang_pair}_{n_shots}shot_handwritten.json", "w") as f:
        json.dump(output, f, indent=1)
# use different folder
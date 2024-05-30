from utils import LANG_TABLE
import json
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch import bfloat16
from pathlib import Path

model_name = "mistralai/Mistral-7B-Instruct-v0.2"

tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
tokenizer.pad_token_id = tokenizer.eos_token_id
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=bfloat16)
batch_size = 32

n_shots = 1
lang_pair = "de-en"
few_shot_dataset_name = "wmt21"
test_dataset_name = "wmt22"
out_path = Path(f"outputs/{test_dataset_name}_{lang_pair}_{few_shot_dataset_name}_{n_shots}shot.json")
out_path.parent.mkdir(exist_ok=True)

source_lang, target_lang = lang_pair.split("-")
prompt = f"Translate this from {LANG_TABLE[source_lang]} to {LANG_TABLE[target_lang]}:\n"
with open(f"datasets/{few_shot_dataset_name}_{lang_pair}.json") as f:
    few_shot_dataset = json.load(f)

few_shot_prompt = []
for i in range(n_shots):
    sample = few_shot_dataset[i]
    few_shot_prompt.append(dict(role="user", content=prompt+sample["source"]))
    few_shot_prompt.append(dict(role="assistant", content=sample["target"]))

with open(f"datasets/{test_dataset_name}_{lang_pair}.json") as f:
    test_dataset = json.load(f)

output = []
for i in tqdm(range(0, len(test_dataset), batch_size)):
    batch = test_dataset[i:i + batch_size]
    messages2d = []
    for sample in batch:
        sample_prompt = [{"role": "user", "content": prompt+sample["source"]}]
        messages2d.append(few_shot_prompt+sample_prompt)
    model_inputs = tokenizer.apply_chat_template(messages2d, padding=True, return_tensors="pt", tokenize=True, add_generation_prompt=True)
    input_sequence_len = model_inputs.shape[1]

    generation = model.generate(model_inputs.to("cuda"), max_new_tokens=512, pad_token_id=tokenizer.eos_token_id, return_dict_in_generate=True)
    new_tokens = generation.sequences[:, input_sequence_len:]
    decoded = tokenizer.batch_decode(new_tokens, skip_special_tokens=True,)
    for translation, sample in zip(decoded, batch):
        output.append(dict(source=sample["source"], target=sample["target"], translation=translation))

    with open(out_path, "w") as f:
        json.dump(output, f, indent=1)

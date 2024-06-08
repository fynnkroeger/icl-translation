from utils import LANG_TABLE
import json
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch import bfloat16
from pathlib import Path
import random
import string


def translate(
    test,
    lang_pair,
    n_shots,
    prompt_formatter,
    few_shot_dataset_name,
    test_dataset_name,
    out_dir,
    run_name,
    model,
    tokenizer,
    batch_size,
    n_batches=None,
    attention_processor=None,
):
    source_lang, target_lang = lang_pair.split("-")
    source_lang, target_lang = LANG_TABLE[source_lang], LANG_TABLE[target_lang]

    if few_shot_dataset_name and n_shots:
        with open(f"datasets/{few_shot_dataset_name}_{lang_pair}.json") as f:
            few_shot_dataset = json.load(f)

        random.seed(42)
        random.shuffle(few_shot_dataset)
        few_shot_examples = few_shot_dataset[:n_shots]

        out_path = (
            out_dir
            / f"{test_dataset_name}_{lang_pair}_{n_shots:02d}shot_{few_shot_dataset_name}_{run_name}.json"
        )
    else:
        few_shot_examples = []
        out_path = out_dir / f"{test_dataset_name}_{lang_pair}_00shot_{run_name}.json"
    print(f"writing output to {out_path} at the end")
    out_dir.mkdir(exist_ok=True)

    with open(f"datasets/{test_dataset_name}_{lang_pair}.json") as f:
        test_dataset = json.load(f)
    output = []
    for i in tqdm(range(0, len(test_dataset), batch_size)):
        if n_batches is not None and i >= n_batches:
            test = True
            break
        batch = test_dataset[i : i + batch_size]
        messages2d = []
        for sample in batch:
            formatted = prompt_formatter(
                few_shot_examples, sample["source"], source_lang, target_lang
            )
            messages2d.append(formatted)
        if i == 0:
            print(prompt_log := messages2d[0])
        model_inputs = tokenizer.apply_chat_template(
            messages2d,
            padding=True,
            return_tensors="pt",
            tokenize=True,
            add_generation_prompt=True,
        )
        input_sequence_len = model_inputs.shape[1]
        generation = model.generate(
            model_inputs.to("cuda"),
            max_new_tokens=200,
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_attentions=attention_processor is not None,
        )
        if attention_processor is not None:
            assert batch_size == 1
            seq_len = generation.sequences.shape[1]
            input_seq_len = model_inputs.shape[1]
            tokens = []
            for s in range(seq_len):
                tokens.append(tokenizer.batch_decode(generation.sequences[:, s])[0])
            name = f"{out_path.stem}_{i:02d}"
            attention_processor(generation.attentions, input_seq_len, seq_len, tokens, name)

        new_tokens = generation.sequences[:, input_sequence_len:]
        decoded = tokenizer.batch_decode(
            new_tokens,
            skip_special_tokens=True,
        )
        for translation, sample in zip(decoded, batch):
            output.append(
                dict(
                    source=sample["source"],
                    target=sample["target"],
                    translation=translation,
                )
            )
        if test:
            break

    if not test:
        with open(out_path, "w") as f:
            json.dump(output, f, indent=1)

        # todo what to log, what into filename
        logs = {}
        if (log_file := Path("outputs/logs.json")).exists():
            logs = json.loads(log_file.read_text())
        logs[out_path.name] = dict(prompt=prompt_log)
        with open(log_file, "w") as f:
            sorted_logs = dict(sorted([(k, v) for k, v in logs.items()]))
            json.dump(sorted_logs, f, indent=1)


def format_multi_message(few_shot_examples, source, source_lang, target_lang):
    prompt = (
        f"Translate this from {source_lang} to {target_lang}. Respond only with the translation.\n"
    )
    messages = []
    for sample in few_shot_examples:
        messages.append(dict(role="user", content=prompt + sample["source"]))
        messages.append(dict(role="assistant", content=sample["target"]))
    return messages + [{"role": "user", "content": prompt + source}]


def format_single_message_prompt_arrow(few_shot_examples, source, source_lang, target_lang):
    if few_shot_examples:
        few_prompt = f"Example translations from {source_lang} to {target_lang}:\n"
        for sample in few_shot_examples:
            few_prompt += f"{sample['source']} -> {sample['target']}\n"
        few_prompt += "\n"
    else:
        few_prompt = ""
    # consider / using examples, ...
    prompt = (
        f"Translate this from {source_lang} to {target_lang}. Respond only with the translation.\n"
    )
    return [{"role": "user", "content": few_prompt + prompt + source}]


def format_single_message_arrow(few_shot_examples, source, source_lang, target_lang):
    if not few_shot_examples:
        raise NotImplementedError()
    few_prompt = ""
    for sample in few_shot_examples:
        few_prompt += f"{sample['source']} -> {sample['target']}\n"
    return [{"role": "user", "content": few_prompt + f"{source} -> "}]


def format_single_message_labeled(few_shot_examples, source, source_lang, target_lang):
    few_prompt = ""
    for sample in few_shot_examples:
        few_prompt += f"{LANG_TABLE[source_lang]}: {sample['source']}\n{LANG_TABLE[target_lang]}: {sample['target']}\n\n"
    prompt = f"{LANG_TABLE[source_lang]}: {source}\n{LANG_TABLE[target_lang]}: "
    return [{"role": "user", "content": few_prompt + prompt}]


if __name__ == "__main__":
    model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    tokenizer.pad_token_id = tokenizer.eos_token_id
    print("finished tokenizer init, on to model")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", torch_dtype=bfloat16, attn_implementation="sdpa"
    )
    print("instatiated model")
    for formatter in [format_multi_message]:
        run_name = "".join(random.choices(string.ascii_uppercase, k=4))
        print("starting run", run_name)

        for n_shots in [1, 4]:
            translate(
                test=False,
                lang_pair="en-de",
                n_shots=n_shots,
                prompt_formatter=formatter,
                few_shot_dataset_name="wmt21",
                test_dataset_name="wmt22",
                out_dir=Path("outputs"),
                run_name=run_name,
                model=model,
                tokenizer=tokenizer,
                batch_size=8,
            )

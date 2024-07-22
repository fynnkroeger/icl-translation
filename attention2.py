from inference import *
import numpy as np
from torch import bfloat16, float16
from functools import partial

folder = Path("Mistral-7B-v0.1/attention")
folder.mkdir(exist_ok=True, parents=True)


def print_attention_len(attentions, *args):
    print(len(attentions))


def save_attention_heatmap(
    attentions, input_seq_len, seq_len, tokens, output_prefix, num_heads, pooling_method
):
    layers = len(attentions[0])
    attention_matrix = np.zeros((layers, num_heads, seq_len, seq_len))
    for layer_num, layer in enumerate(attentions[0]):
        layer = layer.detach().cpu().to(float16).squeeze().numpy()
        attention_matrix[layer_num, :, :input_seq_len, :input_seq_len] = layer

    for token_index, token in enumerate(attentions[1:]):
        for layer_num, layer in enumerate(token):
            layer = layer.detach().cpu().to(float16).numpy().squeeze()
            attention_matrix[
                layer_num,
                :,
                token_index + input_seq_len,
                : token_index + input_seq_len + 1,
            ] = layer

    for index, token in enumerate(tokens[input_seq_len:], input_seq_len):
        if "\n" in token or "</s>" == token or "###" == token:
            attention_matrix = attention_matrix[:, :, :index, :index]
            tokens = tokens[:index]
            break

    if pooling_method == "max":
        pooled = np.max(attention_matrix, axis=1)
    elif pooling_method == "avg":
        pooled = np.average(attention_matrix, axis=1)
    else:
        raise NotImplementedError()
    out_file = folder / f"{output_prefix}_{pooling_method}.npy"
    out_file.parent.mkdir(exist_ok=True, parents=True)  # output prefix contains /
    np.save(out_file, pooled)
    with open(folder / f"{output_prefix}.json", "w") as f:
        json.dump(tokens, f)


if __name__ == "__main__":
    model_name = "mistralai/Mistral-7B-v0.1"
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    tokenizer.pad_token_id = tokenizer.eos_token_id
    print("finished tokenizer init, on to model")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", torch_dtype=bfloat16, attn_implementation="eager"
    )
    print("instatiated model")
    for formatter in [
        format_single_message_arrow_title,
        format_single_message_arrow,
        format_single_message_arrow_oneline,
    ]:
        run_name = formatter.__name__
        no_0shot = run_name in [
            "format_single_message_arrow_oneline",
            "format_single_message_arrow",
        ]
        print("starting run", run_name)
        for lang_pair in ["de-en", "en-de"]:
            for n_shots in [4]:
                if no_0shot and n_shots == 0:
                    continue  # need a few shot example as we have no label
                print(f"starting {run_name} {lang_pair} {n_shots:=} ")
                translate(
                    test=True,  # dont overwrite logs
                    instruct=False,
                    lang_pair=lang_pair,
                    n_shots=n_shots,
                    prompt_formatter=formatter,
                    few_shot_dataset_name="wmt21",
                    test_dataset_name="wmt22",
                    out_dir=Path("Mistral-7B-v0.1"),
                    model=model,
                    tokenizer=tokenizer,
                    batch_size=1,
                    n_batches=100,
                    attention_processor=partial(
                        save_attention_heatmap, num_heads=32, pooling_method="avg"
                    ),
                    shuffle_seed=999,
                )

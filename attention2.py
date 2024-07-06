from inference import *
import numpy as np
from torch import bfloat16, float16

folder = Path("Mistral-7B-v0.1/attention")
folder.mkdir(exist_ok=True, parents=True)


def print_attention_len(attentions, *args):
    print(len(attentions))


def save_attention_heatmap(attentions, input_seq_len, seq_len, tokens, name):
    layers = len(attentions[0])
    num_heads = 32
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
        if "\n" in token or "</s>" == token or "###" == token:  # add cutting at <END> and [
            attention_matrix = attention_matrix[:, :, :index, :index]
            tokens = tokens[:index]
            # print("cutting off at", index)
            break

    max_pooled = np.max(attention_matrix, axis=1)  # head dimension
    avg_pooled = np.average(attention_matrix, axis=1)

    # because name contains a slash
    (folder / f"{name}_max.npy").parent.mkdir(exist_ok=True, parents=True)
    # np.save(folder / f"{name}_max.npy", max_pooled)
    np.save(folder / f"{name}_avg.npy", avg_pooled)
    with open(folder / f"{name}.json", "w") as f:
        json.dump(tokens, f)
    # print(max_pooled.shape)
    # print(tokens)


if __name__ == "__main__":
    # model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    model_name = "mistralai/Mistral-7B-v0.1"
    # model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    tokenizer.pad_token_id = tokenizer.eos_token_id
    print("finished tokenizer init, on to model")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", torch_dtype=bfloat16, attn_implementation="eager"
    )
    print("instatiated model")
    for formatter in [
        format_single_message_arrow_title,
        # format_single_message_labeled,
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
                    test=True,  # dont overwrite logs, log somewhere else?
                    instruct=False,
                    lang_pair=lang_pair,  # because works, but not with nonewline -> why
                    n_shots=n_shots,
                    prompt_formatter=formatter,
                    few_shot_dataset_name="wmt21",
                    test_dataset_name="wmt22",
                    out_dir=Path("Mistral-7B-v0.1"),
                    model=model,
                    tokenizer=tokenizer,
                    batch_size=1,
                    n_batches=100,
                    attention_processor=save_attention_heatmap,  # maybe clean up with higher order function
                    shuffle_seed=999,
                )

from inference import *
import numpy as np
from torch import bfloat16, float16


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
        if "\n" in token or "</s>" == token:  # add cutting at <END> and [
            attention_matrix = attention_matrix[:, :index, :index]
            tokens = tokens[:index]
            print("cutting off at", index)
            break

    max_pooled = np.max(attention_matrix, axis=1)  # head dimension
    avg_pooled = np.average(attention_matrix, axis=1)

    np.save(f"attentions/{name}_max.npy", max_pooled)
    np.save(f"attentions/{name}_avg.npy", avg_pooled)
    with open(f"attentions/{name}.json", "w") as f:
        json.dump(tokens, f)
    print(max_pooled.shape)
    print(tokens)


if __name__ == "__main__":
    Path("attentions").mkdir(exist_ok=True)
    model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    # model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    tokenizer.pad_token_id = tokenizer.eos_token_id
    print("finished tokenizer init, on to model")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", torch_dtype=bfloat16, attn_implementation="eager"
    )
    print("instatiated model")
    for formatter in [format_single_message_arrow_oneline]:
        print("starting run", formatter.__name__)

        for n_shots in [4]:
            translate(
                test=True,  # dont overwrite files
                lang_pair="de-en",
                n_shots=n_shots,
                prompt_formatter=formatter,
                few_shot_dataset_name="wmt21",
                test_dataset_name="wmt22",
                out_dir=Path("outputs"),
                model=model,
                tokenizer=tokenizer,
                batch_size=1,
                n_batches=1,
                attention_processor=save_attention_heatmap,
            )

import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path

n_shots = 4
with open(f"outputs/de-en_{n_shots}shot_tokens.json") as f:
    tokens_NL = json.load(f)

with open(f"outputs/de-en_{n_shots}shot_attention.pickle", "rb") as f:
    attentions = pickle.load(f)

for i, (attention_DBHLL, tokens_L) in enumerate(zip(attentions, tokens_NL, strict=True)):
    attention_DBLL = np.max(attention_DBHLL, axis=2)

    for index, token in enumerate(tokens_L):
        if "INST" in token:
            inst_index = index
    for index, token in enumerate(tokens_L[inst_index:], inst_index):
        if "\n" in token or "</s>" == token:
            attention_DBLL = attention_DBLL[:, :, :index, :index]
            tokens_L = tokens_L[:index]
            break

    depth, batch_size, seqlen, _ = attention_DBLL.shape
    assert batch_size == 1
    labelsize = {0: 10, 1: 7, 2: 6, 4: 6}[n_shots]
    figsize = {0: 10, 1: 10, 2: 32, 4: 50}[n_shots]
    layers = [0, 1, 2, 14, 15, 16, 29, 30, 31]
    layers = list(range(32))
    for layer_index in tqdm(layers):
        fig, ax = plt.subplots(figsize=(figsize, figsize))
        ax.set_title(f"Layer {layer_index}")
        transformed = attention_DBLL[layer_index, 0]
        # multiply by line size? so that we compare to "equal" attention
        # transformed = np.log2(transformed+(2**-40)) vmin=-5, vmax=0
        # transformed *= np.arange(1, 1+transformed.shape[0]).reshape((-1, 1)) vmax=2
        ax.imshow(transformed, cmap="viridis")
        ax.set_xticks(np.arange(seqlen))
        ax.set_yticks(np.arange(seqlen))
        ax.set_xticklabels(tokens_L)
        ax.set_yticklabels(tokens_L)
        ax.tick_params(axis='x', labelrotation=90)
        ax.tick_params(labelsize=labelsize)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # fig.colorbar(im, ax=ax)
        path = Path(f"heatmaps/max_{n_shots}shot_{i}/layer{layer_index}.png")
        path.parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
    break

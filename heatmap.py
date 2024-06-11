import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from multiprocessing import Pool


def process_layer(args):
    path, layer_index, tokens, attention, labelsize, figsize = args
    seqlen = attention.shape[-1]
    fig, ax = plt.subplots(figsize=(figsize, figsize))
    ax.set_title(f"Layer {layer_index}")
    ax.imshow(attention[layer_index], cmap="viridis")
    ax.set_xticks(np.arange(seqlen))
    ax.set_yticks(np.arange(seqlen))
    ax.set_xticklabels(tokens)
    ax.set_yticklabels(tokens)
    ax.tick_params(axis="x", labelrotation=90)
    ax.tick_params(labelsize=labelsize)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # fig.colorbar(im, ax=ax)

    out_path = Path(f"attentions/heatmaps/{path.stem}/layer{layer_index:02d}.png")
    out_path.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def process_file(path, n_shots):
    attention = np.load(path)
    with open(path.with_suffix(".json")) as f:
        tokens = json.load(f)
    labelsize = {0: 10, 1: 7, 2: 6, 4: 6}[n_shots]
    figsize = {0: 10, 1: 10, 2: 32, 4: 40}[n_shots]
    layers = list(range(32))
    args = [(path, layer_index, tokens, attention, labelsize, figsize) for layer_index in layers]

    with Pool(32) as pool:
        list(tqdm(pool.imap_unordered(process_layer, args), total=len(layers)))


if __name__ == "__main__":
    n_shots = 4
    for path in sorted(Path("attentions").iterdir()):
        if path.suffix != ".npy" or "SJ" not in path.stem:
            continue
        process_file(path, n_shots)

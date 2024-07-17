import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from multiprocessing import Pool
from PIL import Image


def process_layer(args):
    path, layer_index, tokens, attention, labelsize, figsize = args
    seqlen = min(attention.shape[-1], len(tokens))  # likely from cutting newline
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

    out_path = Path(f"Mistral-7B-v0.1/heatmaps/{path.stem}/layer{layer_index:02d}.png")
    out_path.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def process_file(path, index, n_shots, use_max=True):
    assert index == 0, "need to implement writing into different path"
    attention = np.load(path / f"{index:04d}_{'max' if use_max else 'avg'}.npy")
    with open(path / f"{index:04d}.json") as f:
        tokens = json.load(f)
    labelsize = {0: 10, 1: 7, 2: 6, 4: 6}[n_shots]
    figsize = {0: 10, 1: 10, 2: 32, 4: 40}[n_shots]
    layers = list(range(32))
    args = [(path, layer_index, tokens, attention, labelsize, figsize) for layer_index in layers]

    with Pool(16) as pool:
        list(tqdm(pool.imap_unordered(process_layer, args), total=len(layers)))


def make_image_plot_false_color(format, n_shot, index, layer, use_max=False):
    n = f"Mistral-7B-v0.1/attention/wmt22_de-en_{n_shot:02d}shot_wmt21_format_single_message_{format}"
    path = Path(n)
    attention = np.load(path / f"{index:04d}_{'max' if use_max else 'avg'}.npy")
    matrix = attention[layer]
    img_arr = np.zeros((matrix.shape[0], matrix.shape[1], 3))
    cmap = plt.get_cmap("viridis")
    boundaries = [0, 0.005, 0.01, 0.1, 1]
    colors = [cmap(0)[:3]] + [cmap(x)[:3] for x in np.linspace(0.5, 1, len(boundaries) - 2)]
    for i, (a, b) in enumerate(zip(boundaries, boundaries[1:])):
        mask = (a < matrix) & (matrix <= b)
        img_arr[mask, :] = colors[i]
    img = Image.fromarray((img_arr * 255).astype(np.uint8))
    img = img.resize((img.width * 3, img.height * 3), Image.NEAREST)
    img.save(f"Mistral-7B-v0.1/heatmaps/{format}_{n_shot:02d}_{layer:02d}.png")

    with open(path / f"{index:04d}.json") as f:
        tokens = np.array(json.load(f))
    stripe = np.zeros((matrix.shape[0], 20, 3), dtype=np.uint8)
    stripe[tokens == "###", :] = [30, 70, 255]
    stripe[tokens == "\n", :] = [30, 70, 255]
    stripe[tokens == "->", :] = [200, 200, 30]
    img = Image.fromarray(stripe)
    img = img.resize((img.width * 3, img.height * 3), Image.NEAREST)
    img.save(f"Mistral-7B-v0.1/heatmaps/{format}_{n_shot:02d}_{layer:02d}_stripe.png")


if __name__ == "__main__":
    make_image_plot_false_color("arrow_oneline", 4, 0, 17)  # translation
    make_image_plot_false_color("arrow_oneline", 4, 0, 11)  # induction
    make_image_plot_false_color("arrow_oneline", 4, 0, 26)  # special
    make_image_plot_false_color("arrow_title", 4, 0, 25)  # instruction
    make_image_plot_false_color("arrow_oneline", 4, 0, 31)  # example
    make_image_plot_false_color("arrow", 4, 0, 25)  # example

    exit()
    process_file(
        Path("Mistral-7B-v0.1/attention/wmt22_de-en_04shot_wmt21_format_single_message_arrow"),
        index=0,
        n_shots=2,
        use_max=False,
    )
    exit()
    for path in Path("Mistral-7B-v0.1/attention").iterdir():
        print(path)
        n_shots = int(path.name.split("_")[2][:2])
        if n_shots == 4:
            n_shots = 2
        process_file(path, index=0, n_shots=n_shots)

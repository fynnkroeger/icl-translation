from pathlib import Path
import numpy as np
import json
import utils
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import utils
import random
from PIL import Image, ImageDraw

# for all formats
# given a format, go though all attention maps, for each
# split the prompt, calculate attention flows
# then average them and output plots


def calculate_average_flow_and_plot(path: Path, n, average_over_coordinates=True):
    _, langpair, shot, _, *name = path.name.split("_")
    good_name = utils.prompt_names["_".join(name)]
    assert good_name == "arrow oneline", "format not implemented yet"
    file_name = f'{langpair}_{shot}_{good_name.replace(" ", "-")}_{n:04d}'
    print(file_name)
    n_shots = int(shot[:2])

    flows = {}
    colors = list(mcolors.XKCD_COLORS.values())
    random.seed(1)
    random.shuffle(colors)
    colors = ["#000000"] + list(mcolors.TABLEAU_COLORS.values()) + colors

    for i in range(n):
        matrix = np.load(path / f"{i:04d}_avg.npy")
        with open(path / f"{i:04d}.json", "r") as f:
            tokens = json.load(f)
        print(len(tokens), matrix.shape)
        # get indices of seperators
        # then ends, then rest is examples
        sep = "###"
        joiner = "->"
        if sep == tokens[-1]:
            tokens = tokens[:-1]
        # print(tokens)
        sep_indices = [i for i, t in enumerate(tokens) if t == sep]
        if len(sep_indices) != 4:
            print(f"wrong number seperators {i:04d}")
            continue
        end_target = [utils.extend_left_non_alpha(tokens, i) for i in sep_indices]
        print(end_target)
        end_source_all = []
        error = False
        for a, b in zip([[0]] + end_target, end_target + [[len(tokens)]]):
            start = a[-1] + 1
            example = tokens[start : b[0]]
            join_index = [i + start for i, t in enumerate(example) if t == joiner]
            if len(join_index) != 1:
                print(f"wrong number joiners {i:04d}")
                error = True
                break
            end_source_all.append(utils.extend_left_non_alpha(tokens, join_index[0]))
        if error:
            continue
        print(["".join(tokens[x] for x in i) for i in end_source_all])

        source_all = []
        for a, b in zip([[0]] + end_target, end_source_all):
            source_all.append(list(range(a[-1] + 1, b[0])))
        target_all = []
        for a, b in zip(end_source_all, end_target + [[len(tokens)]]):
            target_all.append(list(range(a[-1] + 1, b[0])))
        task_target = target_all[-1]
        task_source = source_all[-1]
        task_end_source = end_source_all[-1]
        n_shots = len(end_target)
        source = source_all[:n_shots]
        end_source = end_source_all[:n_shots]
        target = target_all[:n_shots]
        # calculate flows
        coordinates = {
            # does this make sense if we dont have end_source anywhere?
            "translation": utils.coords_multi(utils.append_pointwise(source, end_source), target),
            "translation task": utils.coords(task_source + task_end_source, task_target),
            "induction": utils.coords_multi(target, target)
            + utils.coords_multi(source_all, source_all),
            "induction task": utils.coords(task_target, task_target),
            "summarize source": utils.coords_multi(source_all, end_source_all)
            + utils.coords_multi(end_source_all, end_source_all),
            "summarize example": utils.coords_multi(
                utils.append_pointwise(source, end_source, target), end_target
            ),
            "summary attention": utils.coords(utils.flat(end_target + end_source), task_target),
            "example attention": utils.coords(utils.flat(source + target), task_target),
        }
        everything = []
        for v in coordinates.values():
            everything += v
        set_everything = set(everything)
        assert len(everything) == len(
            set_everything
        ), "matix value assinged to multiple categories!"
        coordinates["rest"] = [
            (i, j)
            for i in range(len(tokens))
            for j in range(len(tokens))
            if 0 < i < j <= task_target[-1] and (i, j) not in set_everything
        ]
        # del coordinates["rest"]
        for key, c in coordinates.items():
            res = sum(matrix[:, j, i] for i, j in c) / (len(c) * n)
            if key in flows:
                flows[key] += res
            else:
                flows[key] = res

        if i == 0:
            img = Image.new("RGB", (matrix.shape[1], matrix.shape[2]))
            draw = ImageDraw.Draw(img)
            for (name, coords), col in zip(coordinates.items(), colors[1:]):
                print(name, col)
                for y, x in coords:
                    draw.point((y, x), fill=col)
            img = img.resize((img.width * 3, img.height * 3), Image.NEAREST)
            img.save(f"Mistral-7B-v0.1/plots/{file_name}_matrix.png")

    fig, ax = plt.subplots()
    for idx, (k, v) in enumerate(flows.items()):
        ax.plot(v, label=k, color=colors[idx + 1])
    ax.legend()
    plt.title(file_name)
    plt.savefig(f"Mistral-7B-v0.1/plots/{file_name}_flow.png", dpi=300)
    print("done out")


# todo somehow get the different segments with the matrix visually
# do the matrix just as a normal image?
if __name__ == "__main__":
    Path("Mistral-7B-v0.1/plots").mkdir(exist_ok=True)
    for lang in "en-de", "de-en":
        path = Path(
            f"Mistral-7B-v0.1/attention/wmt22_{lang}_04shot_wmt21_format_single_message_arrow_oneline"
        )
        n = len([p for p in path.iterdir() if "max" not in p.name]) // 2
        print(n, path.name)
        calculate_average_flow_and_plot(path, n)

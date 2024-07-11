from pathlib import Path
import numpy as np
import json
import utils
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.scale as mscale
import utils
import random
from PIL import Image, ImageDraw
from tqdm import tqdm
from collections import defaultdict

# for all formats
# given a format, go though all attention maps, for each
# split the prompt, calculate attention flows
# then average them and output plots


def calculate_average_flow_and_plot(path: Path, n, puctuation_summary=False):
    _, langpair, shot, _, *name = path.name.split("_")
    good_name = utils.prompt_names["_".join(name)]
    assert good_name in ["arrow oneline", "arrow", "title arrow"], "format not implemented yet"
    sep = {"arrow oneline": "###", "arrow": "\n", "title arrow": "\n"}[good_name]
    joiner = "->"

    file_name = f'{langpair}_{shot}_{good_name.replace(" ", "-")}_{n:04d}'
    n_shots = int(shot[:2])

    flows = defaultdict(list)
    colors = list(mcolors.XKCD_COLORS.values())
    random.seed(1)
    random.shuffle(colors)
    colors = ["#000000"] + list(mcolors.TABLEAU_COLORS.values()) + colors
    for i in tqdm(range(n)):
        matrix = np.load(path / f"{i:04d}_avg.npy")
        with open(path / f"{i:04d}.json", "r") as f:
            tokens = json.load(f)
        # print(len(tokens), matrix.shape)
        # get indices of seperators
        # then ends, then rest is examples
        if sep == tokens[-1]:
            tokens = tokens[:-1]
        # print(tokens)
        sep_indices = [i for i, t in enumerate(tokens) if t == sep]
        if good_name == "title arrow":
            instruction_end, *sep_indices = sep_indices
            instruction = list(range(1, instruction_end + 1))  # ob1
        else:
            instruction_end = 0
        if len(sep_indices) != 4:
            print(f"wrong number seperators {i:04d}")
            continue
        end_target = [utils.extend_left_non_alpha(tokens, i) for i in sep_indices]
        end_source_all = []
        error = False
        for a, b in zip([[instruction_end]] + end_target, end_target + [[len(tokens)]]):
            start = a[-1] + 1
            example = tokens[start : b[0]]
            join_index = [i + start for i, t in enumerate(example) if t == joiner]
            second_is_last = len(join_index) == 2 and join_index[1] == b[0] - 1
            # just ignore, is bad generations
            if len(join_index) != 1 and not second_is_last:
                print(f"wrong number joiners {i:04d}")
                error = True
                break
            end_source_all.append(utils.extend_left_non_alpha(tokens, join_index[0]))
        if error:
            continue
        # print([" ".join(tokens[x] for x in i) for i in end_source_all])

        source_all = []
        for a, b in zip([[instruction_end]] + end_target, end_source_all):
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

        joiner_before = []
        for s in range(n_shots):
            joiner_before.extend(
                utils.coords(
                    end_target[s],
                    utils.flat(
                        source_all[s + 1 :]
                        + target[s + 1 :]
                        + end_source_all[s + 1 :]
                        + end_target[s + 1 :]
                    ),
                )
            )

        coordinates = {
            "translation": utils.coords_multi(utils.append_pointwise(source, end_source), target),
            "translation task": utils.coords(task_source + task_end_source, task_target),
            "induction": utils.coords_multi(target, target)
            + utils.coords_multi(source_all, source_all),
            "induction task": utils.coords(task_target, task_target),
            "summarize example": utils.coords_multi(
                utils.append_pointwise(source, end_source, target), end_target
            ),
            "joiner attention": joiner_before,
            "joiner attention task": utils.coords(utils.flat(end_target), task_target),
            # todo i need a plot for this?
            # "joiner attention task 0": utils.coords(utils.flat([end_target[0]]), task_target),  # basically all in arrow
            # "joiner attention task": utils.coords(utils.flat(end_target[1:]), task_target),
            # low ones
            # "example attention": utils.coords(utils.flat(source + target), task_target),
            # "summarize source": utils.coords_multi(source_all, end_source_all),
            # "divider attention": utils.coords(utils.flat(end_source), task_target),
        }
        # if good_name == "arrow oneline":
        #     coordinates.update(
        #         {
        #             "summarize source": utils.coords_multi(source_all, end_source_all),
        #             "divider attention": utils.coords(utils.flat(end_source), task_target),
        #         }
        #     )
        if good_name == "title arrow":
            coordinates.update(
                {
                    "instruction summary": utils.coords(instruction, [instruction_end]),
                    "instruction attention": utils.coords(
                        [instruction_end],
                        utils.flat(source_all + target_all + end_source_all + end_target),
                    ),  # todo decompose into task and non task?
                }
            )

        if puctuation_summary:
            end_source_punct = [a[:-1] for a in end_source_all]
            end_source_sep = [[a[-1]] for a in end_source_all]
            end_target_punct = [a[:-1] for a in end_target]
            end_target_sep = [[a[-1]] for a in end_target]

            coordinates = {
                "summarize source": utils.coords_multi(source_all, end_source_all)
                + utils.coords_multi(end_source_all, end_source_all),
                "summarize source punct": utils.coords_multi(source_all, end_source_punct),
                "summarize source sep": utils.coords_multi(source_all, end_source_sep)
                + utils.coords_multi(end_source_punct, end_source_sep),
                "summarize example": utils.coords_multi(
                    utils.append_pointwise(source, end_source, target), end_target
                ),
                "summarize example punct": utils.coords_multi(
                    utils.append_pointwise(source, end_source, target), end_target_punct
                ),
                "summarize example sep": utils.coords_multi(
                    utils.append_pointwise(source, end_source, target), end_target_sep
                ),
                "summary attention": utils.coords(utils.flat(end_target + end_source), task_target),
                "summary attention punct": utils.coords(
                    utils.flat(end_target_punct + end_source_punct[:-1]), task_target
                ),
                "summary attention sep": utils.coords(
                    utils.flat(end_target_sep + end_source_sep[:-1]), task_target
                ),
            }
        everything = []
        for v in coordinates.values():
            everything += v
        set_everything = set(everything)
        if not puctuation_summary:
            assert len(everything) == len(
                set_everything
            ), "matix value assinged to multiple categories!"
            coordinates["rest"] = [
                (i, j)
                for i in range(len(tokens))
                for j in range(len(tokens))
                if 0 < i < j <= task_target[-1] and (i, j) not in set_everything
            ]
        for key, c in coordinates.items():
            relevant = [matrix[:, j, i] for i, j in c]
            flows[key].append(np.average(relevant, axis=0))

        if i == 0:
            if puctuation_summary:
                for (name, coords), col in zip(coordinates.items(), colors[1:]):
                    img = Image.new("RGB", (matrix.shape[1], matrix.shape[2]))
                    draw = ImageDraw.Draw(img)
                    for y, x in coords:
                        draw.point((y, x), fill=col)
                    img = img.resize((img.width * 3, img.height * 3), Image.NEAREST)
                    p = Path(f"Mistral-7B-v0.1/plots/{file_name}_matrix/{name}.png")
                    p.parent.mkdir(exist_ok=True)
                    img.save(p)

            img = Image.new("RGB", (matrix.shape[1], matrix.shape[2]))
            draw = ImageDraw.Draw(img)
            for (name, coords), col in zip(coordinates.items(), colors[1:]):
                for y, x in coords:
                    draw.point((y, x), fill=col)
            img = img.resize((img.width * 3, img.height * 3), Image.NEAREST)
            img.save(f"Mistral-7B-v0.1/plots/matrix/{file_name}.png")
    mscale.register_scale(utils.SegmentedScale)
    fig, ax = plt.subplots()
    x = np.arange(1, 32 + 1)
    if good_name == "title arrow":
        b = 0.03
        plt.yscale("segmented", breakpoint=b, scale_ratio=10)
        ax.plot(x, [b] * 32, "--", color="black")
    for idx, (k, v) in enumerate(flows.items()):
        arr = np.array(v)
        # todo make two plots, another without clamping
        ax.plot(x, np.median(arr, axis=0), label=k, color=colors[idx + 1], alpha=0.8)
        # pretty much the same
        # ax.plot(x, np.clip(np.average(arr, 0), 0, c), ".", color=colors[idx + 1], alpha=0.8)
        # never anything unexpected -> dont crowd unnecessary
        # ax.fill_between(
        #     x,
        #     np.quantile(arr, 0.25, axis=0),
        #     np.quantile(arr, 0.75, axis=0),
        #     color=colors[idx + 1],
        #     alpha=0.1,
        # )
    ax.legend()
    plt.title(file_name)
    plt.savefig(f"Mistral-7B-v0.1/plots/{file_name}_flow.png", dpi=300)
    print(len(list(flows.values())[0]), "valid examples", file_name)


if __name__ == "__main__":
    Path("Mistral-7B-v0.1/plots/matrix").mkdir(exist_ok=True, parents=True)
    for mode in ["arrow_title", "arrow", "arrow_oneline"]:
        for lang in ["de-en", "en-de"]:
            path = Path(
                f"Mistral-7B-v0.1/attention/wmt22_{lang}_04shot_wmt21_format_single_message_{mode}"  # _title # _oneline
            )
            n = len([p for p in path.iterdir() if "max" not in p.name]) // 2
            # print(n, path.name)
            calculate_average_flow_and_plot(path, n)

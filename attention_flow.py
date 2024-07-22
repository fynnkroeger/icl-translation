from pathlib import Path
import numpy as np
import json
import utils
import matplotlib.pyplot as plt
import matplotlib.scale as mscale
import matplotlib.patches as mpatches
import utils
from PIL import Image, ImageDraw
from tqdm import tqdm
from collections import defaultdict

# for all formats
# given a format, go though all attention maps, for each
# split the prompt, calculate attention flows
# then average them and output plots

colors_dict = {
    "translation": "#FFAA6D",
    "translation divider": "#FF6D6D",
    "translation task": "#FF6900",
    "translation divider task": "#FF0000",
    "induction": "#fdb915",
    "summarize source": "#E514FA",
    "summarize example": "#8714B7",
    "joiner attention": "#7DE977",
    "joiner attention task": "#149A0D",
    "instruction summary": "#021E72",
    "instruction attention": "#0441F7",
    "instruction attention task": "#14EEFC",
    "joiner-joiner flow": "#6DC1F2",
    "divider-divider flow": "#00C49A",
    "joiner-divider flow": "#757500",
    "divider-joiner flow": "#FFA199",
    "rest": "#333",
}


def calculate_average_flow_and_plot(path: Path, n, puctuation_summary=False, group_matrix=False):
    _, langpair, shot, _, *name = path.name.split("_")
    good_name = utils.prompt_names["_".join(name)]
    assert good_name in ["arrow oneline", "arrow", "title arrow"], "format not implemented yet"
    joiner_token = {"arrow oneline": "###", "arrow": "\n", "title arrow": "\n"}[good_name]
    divier_token = "->"

    file_name = f'{langpair}_{shot}_{good_name.replace(" ", "-")}_{n:04d}'
    n_shots = int(shot[:2])

    flows = defaultdict(list)
    for i in tqdm(range(n)):
        matrix = np.load(path / f"{i:04d}_avg.npy")
        with open(path / f"{i:04d}.json", "r") as f:
            tokens = json.load(f)
        if joiner_token == tokens[-1]:
            tokens = tokens[:-1]
        joiner_indices = [i for i, t in enumerate(tokens) if t == joiner_token]
        if good_name == "title arrow":
            instruction_end, *joiner_indices = joiner_indices
            instruction = list(range(1, instruction_end + 1))  # ob1
        else:
            instruction_end = 0
        if len(joiner_indices) != n_shots:
            print(f"wrong number joiners {i:04d}")
            continue
        joiners = [utils.extend_left_non_alpha(tokens, i) for i in joiner_indices]
        divider_all = []
        error = False
        for a, b in zip([[instruction_end]] + joiners, joiners + [[len(tokens)]]):
            start = a[-1] + 1
            example = tokens[start : b[0]]
            divider_index = [i + start for i, t in enumerate(example) if t == divier_token]
            second_is_last = len(divider_index) == 2 and divider_index[1] == b[0] - 1
            # just use the first, is due to bad generations
            if len(divider_index) != 1 and not second_is_last:
                print(f"wrong number dividers {i:04d}")
                error = True
                break
            divider_all.append(utils.extend_left_non_alpha(tokens, divider_index[0]))
        if error:
            continue

        source_all = []
        for a, b in zip([[instruction_end]] + joiners, divider_all):
            source_all.append(list(range(a[-1] + 1, b[0])))
        target_all = []
        for a, b in zip(divider_all, joiners + [[len(tokens)]]):
            target_all.append(list(range(a[-1] + 1, b[0])))
        task_target = target_all[-1]
        task_source = source_all[-1]
        task_divider = divider_all[-1]
        n_shots = len(joiners)
        source = source_all[:n_shots]
        divider = divider_all[:n_shots]
        target = target_all[:n_shots]

        joiner_before = []
        joiner_joiner = []
        div_div = []
        joiner_div = []
        div_joiner = []

        for s in range(n_shots):
            joiner_joiner.extend(utils.coords([joiners[s][-1]], [j[-1] for j in joiners]))
            joiner_div.extend(utils.coords([joiners[s][-1]], [j[-1] for j in divider_all]))
            # no need for divider_all on left as can flow nowhere
            div_div.extend(utils.coords([divider[s][-1]], [d[-1] for d in divider_all]))
            # dont overlap with summary attention, only consider flow to later joiners
            div_joiner.extend(utils.coords([divider[s][-1]], [d[-1] for d in joiners[s + 1 :]]))
            c = utils.coords(joiners[s], utils.flat(source_all + target + divider_all + joiners))
            joiner_before.extend([p for p in c if p not in joiner_joiner and p not in joiner_div])

        coordinates = {
            "translation": utils.coords_multi(source, target),
            "translation task": utils.coords(task_source, task_target),
            "translation divider": utils.coords_multi(divider, target),
            "translation divider task": utils.coords(task_divider, task_target),
            "induction": utils.coords_multi(target, target)
            + utils.coords_multi(source_all, source_all)
            + utils.coords(task_target, task_target),
            "summarize source": utils.coords_multi(source_all, divider_all),
            "summarize example": utils.coords_multi(
                utils.append_pointwise(divider, target), joiners
            ),
            "joiner attention": joiner_before,
            "joiner attention task": utils.coords(utils.flat(joiners), task_target),
            "joiner-joiner flow": joiner_joiner,
            "joiner-divider flow": joiner_div,
            "divider-divider flow": div_div,
            "divider-joiner flow": div_joiner,
            # todo i need a plot for this?
            # "joiner attention task 0": utils.coords(utils.flat([end_target[0]]), task_target),  # basically all in arrow
            # "joiner attention task": utils.coords(utils.flat(end_target[1:]), task_target),
            # low ones
            # "example attention": utils.coords(utils.flat(source + target), task_target),
            # "divider attention": utils.coords(utils.flat(end_source), task_target),
        }
        groups = {
            "translation": [
                "translation",
                "translation task",
                "translation divider",
                "translation divider task",
            ],
            "induction": ["induction", "induction task"],
            "example": [
                "summarize source",
                "summarize example",
                "joiner attention",
                "joiner attention task",
            ],
            "special": [
                "joiner-joiner flow",
                "divider-divider flow",
                "joiner-divider flow",
                "divider-joiner flow",
            ],
        }
        right_plot = groups["special"] + ["summarize source", "summarize example"]
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
                        utils.flat(source_all + target + divider_all + joiners) + task_target,
                    ),
                }
            )
            groups = {
                "instruction": [
                    "instruction summary",
                    "instruction attention",
                ]
            }

        if puctuation_summary:
            end_source_punct = [a[:-1] for a in divider_all]
            end_source_sep = [[a[-1]] for a in divider_all]
            end_target_punct = [a[:-1] for a in joiners]
            end_target_sep = [[a[-1]] for a in joiners]

            coordinates = {
                "summarize source": utils.coords_multi(source_all, divider_all)
                + utils.coords_multi(divider_all, divider_all),
                "summarize source punct": utils.coords_multi(source_all, end_source_punct),
                "summarize source sep": utils.coords_multi(source_all, end_source_sep)
                + utils.coords_multi(end_source_punct, end_source_sep),
                "summarize example": utils.coords_multi(
                    utils.append_pointwise(source, divider, target), joiners
                ),
                "summarize example punct": utils.coords_multi(
                    utils.append_pointwise(source, divider, target), end_target_punct
                ),
                "summarize example sep": utils.coords_multi(
                    utils.append_pointwise(source, divider, target), end_target_sep
                ),
                "summary attention": utils.coords(utils.flat(joiners + divider), task_target),
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
        left = {k: v for k, v in coordinates.items() if k not in right_plot}
        right = {k: v for k, v in coordinates.items() if k in right_plot}
        coordinates = dict(**left, **right)
        for key, c in coordinates.items():
            relevant = [matrix[:, j, i] for i, j in c]
            flows[key].append(np.average(relevant, axis=0))

        if i == 0:
            if puctuation_summary:
                for name, coords in coordinates.items():
                    img = Image.new("RGB", (matrix.shape[1], matrix.shape[2]))
                    draw = ImageDraw.Draw(img)
                    for y, x in coords:
                        draw.point((y, x), fill=colors_dict[name])
                    img = img.resize((img.width * 3, img.height * 3), Image.NEAREST)
                    p = Path(f"Mistral-7B-v0.1/plots/{file_name}_matrix/{name}.png")
                    p.parent.mkdir(exist_ok=True)
                    img.save(p)
            if group_matrix:
                for name, group in groups.items():
                    img = Image.new("RGB", (matrix.shape[1], matrix.shape[2]))
                    draw = ImageDraw.Draw(img)
                    for name2, coords in coordinates.items():
                        for y, x in coords:
                            draw.point(
                                (y, x), fill=colors_dict[name2] if name2 in group else "#333"
                            )
                    img = img.resize((img.width * 3, img.height * 3), Image.NEAREST)
                    img.save(f"Mistral-7B-v0.1/plots/matrix/{file_name}_{name}.png")
                    print("matrix", name)
            else:
                img = Image.new("RGB", (matrix.shape[1], matrix.shape[2]))
                draw = ImageDraw.Draw(img)
                for name, coords in coordinates.items():
                    for y, x in coords:
                        draw.point((y, x), fill=colors_dict[name])
                img = img.resize((img.width * 3, img.height * 3), Image.NEAREST)
                img.save(f"Mistral-7B-v0.1/plots/matrix/{file_name}.png")
    mscale.register_scale(utils.SegmentedScale)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    patches = []
    x = np.arange(1, 32 + 1)
    for idx, (k, v) in enumerate(flows.items()):
        arr = np.array(v)
        alpha1 = 0.05 if k in right_plot else 1
        alpha2 = 0.9 if k in right_plot else 0.05
        ax1.plot(x, np.median(arr, axis=0), label=k, color=colors_dict[k], alpha=alpha1)
        ax2.plot(x, np.median(arr, axis=0), label=k, color=colors_dict[k], alpha=alpha2)
        patches.append(mpatches.Patch(color=colors_dict[k], label=k))
    if good_name == "title arrow" or good_name == "arrow":
        b = 0.03
        ax1.set_yscale("segmented", breakpoint=b, scale_ratio=10 if good_name == "arrow" else 20)
        ax2.set_yscale("segmented", breakpoint=b, scale_ratio=10 if good_name == "arrow" else 20)
        _, top = plt.ylim()
        top = round(top / 0.1, 0) * 0.1
        ticks = np.concatenate(
            [np.linspace(0, b, 5, endpoint=False), np.linspace(b, top + b, 5, endpoint=False)]
        )
        ax1.set_yticks(ticks)
        ax2.set_yticks(ticks)
        ax1.plot(x, [b] * 32, "--", color="black")
        ax2.plot(x, [b] * 32, "--", color="black")
    plt.title(file_name)
    ax2.legend(handles=patches, bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(f"Mistral-7B-v0.1/plots/{file_name}_flow.png", dpi=300)
    print(len(list(flows.values())[0]), "valid examples", file_name)


if __name__ == "__main__":
    Path("Mistral-7B-v0.1/plots/matrix").mkdir(exist_ok=True, parents=True)

    p = Path(f"Mistral-7B-v0.1/attention/wmt22_de-en_04shot_wmt21_format_single_message_arrow")
    calculate_average_flow_and_plot(p, 1, group_matrix=True)
    p2 = Path(
        f"Mistral-7B-v0.1/attention/wmt22_de-en_04shot_wmt21_format_single_message_arrow_title"
    )
    calculate_average_flow_and_plot(p2, 1, group_matrix=True)
    for mode in ["arrow_title", "arrow", "arrow_oneline"]:
        for lang in ["de-en", "en-de"]:
            path = Path(
                f"Mistral-7B-v0.1/attention/wmt22_{lang}_04shot_wmt21_format_single_message_{mode}"
            )
            n = len([p for p in path.iterdir() if "max" not in p.name]) // 2
            calculate_average_flow_and_plot(path, n)

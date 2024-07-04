from pathlib import Path
import numpy as np
import json
import utils
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import utils

# for all formats
# given a format, go though all attention maps, for each
# split the prompt, calculate attention flows
# then average them and output plots


def calculate_average_flow_and_plot(path: Path, n, average_over_coordinates=True):
    _, langpair, shot, _, *name = path.name.split("_")
    good_name = utils.prompt_names["_".join(name)]
    file_name = f'{langpair}_{shot}_{good_name.replace(" ", "-")}_{n:04d}'
    print(file_name)
    n_shots = int(shot[:2])

    flows = {}
    colors = ["#000000"] + list(mcolors.TABLEAU_COLORS.values())

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
        for a, b in zip([[0]] + end_target, end_target + [[len(tokens)]]):
            start = a[-1] + 1
            example = tokens[start : b[0]]
            join_index = [i + start for i, t in enumerate(example) if t == joiner]
            if len(join_index) != 1:
                print(f"wrong number joiners {i:04d}")
                continue
            end_source_all.append(utils.extend_left_non_alpha(tokens, join_index[0]))
        print(["".join(tokens[x] for x in i) for i in end_source_all])

        source_all = []
        for a, b in zip([[0]] + end_target, end_source_all):
            source_all.append(list(range(a[-1] + 1, b[0])))
        target_all = []
        for a, b in zip(end_source_all, end_target + [[len(tokens)]]):
            target_all.append(list(range(a[-1] + 1, b[0])))
        task_target = target_all[-1]
        n_shots = len(end_target)
        source = source_all[:n_shots]
        end_source = end_source_all[:n_shots]
        target = target_all[:n_shots]
        # calculate flows
        coordinates = {
            # does this make sense if we dont have end_source anywhere?
            "translation": utils.coords_multi(
                utils.append_pointwise(source_all, end_source_all), target_all
            ),
            "induction": utils.coords_multi(target_all, target_all)
            + utils.coords_multi(source_all, source_all),
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
            colors = colors
            # Create a triangular matrix
            tri_matrix = np.zeros((matrix.shape[1], matrix.shape[2]))
            color_map = {k: colors[idx + 1] for idx, k in enumerate(coordinates.keys())}
            for index, (key, c) in enumerate(coordinates.items()):
                for i, j in c:
                    tri_matrix[i, j] = index + 1

            # Plot the triangular matrix
            fig, ax = plt.subplots()
            cmap = mcolors.ListedColormap(colors)
            bounds = np.linspace(0, len(colors), len(colors) + 1)
            norm = mcolors.BoundaryNorm(bounds, cmap.N)

            # masked_matrix = np.ma.masked_where(tri_matrix == 0, tri_matrix)
            cax = ax.matshow(tri_matrix.T, cmap=cmap, norm=norm)

            # Create custom legend
            handles = [mpatches.Patch(color=color_map[key], label=key) for key in color_map]
            ax.legend(handles=handles, loc="upper right")
            plt.tight_layout()
            plt.axis("off")

            plt.savefig(f"Mistral-7B-v0.1/plots/{file_name}_matrix.png", dpi=300)
            print("matrix")

    fig, ax = plt.subplots()
    for idx, (k, v) in enumerate(flows.items()):
        ax.plot(v, label=k, color=colors[idx + 1])
    ax.legend()
    plt.savefig(f"Mistral-7B-v0.1/plots/{file_name}_flow.png", dpi=300)
    print("done out")


if __name__ == "__main__":
    path = Path(
        "Mistral-7B-v0.1/attention/wmt22_de-en_04shot_wmt21_format_single_message_arrow_oneline"
    )
    calculate_average_flow_and_plot(path, 1)

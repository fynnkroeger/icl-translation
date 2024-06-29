from pathlib import Path
import numpy as np
import json
import utils

# for all formats
# given a format, go though all attention maps, for each
# split the prompt, calculate attention flows
# then average them and output plots

path = Path(
    "Mistral-7B-v0.1/attention/wmt22_en-de_04shot_wmt21_format_single_message_arrow_oneline"
)
print(path)
n = 4
n_shots = 4
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
    print(end_source_all)

    source_all = []
    for a, b in zip([[0]] + end_target, end_source_all):
        source_all.append(list(range(a[-1] + 1, b[0])))
    target_all = []
    for a, b in zip(end_source_all, end_target + [[len(tokens)]]):
        target_all.append(list(range(a[-1] + 1, b[0])))
    print(source_all)
    print(target_all)
    task_target = target_all[-1]
    n_shots = len(end_target)
    source = source_all[:n_shots]
    end_source = end_source_all[:n_shots]
    target = target_all[:n_shots]
    # calculate flows
    coordinates = {
        # does this make sense if we dont have end_source anywhere?
        "summarize source": utils.coords_multi(source_all, end_source_all),
        "summarize example": utils.coords_multi(
            utils.append_pointwise(source, end_source, target), end_target
        ),
        "example attention": utils.coords(utils.flat(source + target), task_target),
        "summary attention": utils.coords(utils.flat(end_target + end_source), task_target),
    }

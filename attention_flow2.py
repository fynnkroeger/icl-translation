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
    end_source = []
    for a, b in zip([[0]] + end_target, end_target):
        start = a[-1] + 1
        example = tokens[start : b[0]]
        join_index = [i + start for i, t in enumerate(example) if t == joiner]
        if len(join_index) != 1:
            print(f"wrong number joiners {i:04d}")
            continue
        end_source.append(utils.extend_left_non_alpha(tokens, join_index[0]))
    print(end_source)

    continue

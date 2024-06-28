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
    if sep == tokens[-1]:
        tokens = tokens[:-1]
    # print(tokens)
    sep_indices = [i for i, t in enumerate(tokens) if t == sep]
    if len(sep_indices) != 4:
        print(f"wrong n seperators {i:04d}")
        continue
    end_target = []
    for index in sep_indices:
        stop_indices = []
        while not tokens[index].isalpha():
            stop_indices.append(index)
            index -= 1
        stop_indices.reverse()
        end_target.append(stop_indices)
    print(sep_indices)
    print(end_target)
    examples = []
    example = []
    print([[tokens[a] for a in x] for x in end_target])
    continue
    s = utils.split_list(tokens, sep)
    for x in s:
        print(x)

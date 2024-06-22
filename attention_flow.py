import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt


def flat(a):
    out = []
    for x in a:
        out += x
    return out


def coords(from_tokens, to_tokens):
    return [(i, j) for i in from_tokens for j in to_tokens if i < j]


def coords_multi(from_tokens, to_tokens):
    out = []
    for a, b in zip(from_tokens, to_tokens):
        out += coords(a, b)
    return out


# todo use avg pooling here
# todo change naming for attentions (format, index), put all in folder if we do more
path = Path("attentions/wmt22_de-en_04shot_wmt21_format_single_message_arrow_oneline_00_avg.npy")
attention = np.load(path)
with open(path.with_stem(path.stem[:-4]).with_suffix(".json")) as f:
    tokens = json.load(f)

print(attention.shape)
# for i, t in enumerate(tokens):
#     print(i, t)

start_and_inst = [0, 1, 2, 3]

source = [list(range(4, 24)), list(range(45, 62)), list(range(76, 91)), list(range(105, 129))]

end_source = [[24, 25], [62], [91, 92], [129, 130]]

target = [list(range(26, 43)), list(range(63, 74)), list(range(93, 103)), list(range(131, 146))]

end_target = [[43, 44], [74, 75], [103, 104], [146, 147]]

task_source = list(range(148, 160))
task_source_end = [160, 161]  # . ->
inst = [162, 163, 164, 165, 166]
task_target = list(range(167, 181))
"""
kinds of tokens:
- instruction
- ex_source
- joiner
- ex_target
- seperator (consider end of sentence too)
- instruction
^-- always the same so we can annotate once 
- source
- target (output)
- INST
# here find inst, this splits into the source and target
# before: split by ###

uninteresting patterns
- self, previous, first attention
- within target
- within examples

interesting patterns
- direct flow from ex to target
- ex to summary
- summary to target
- source to target
- between summary tokens
"""


coordinates = {}
# seperate examples and task??
# coordinates["source-source_end"] = coords_multi(source, end_source)
# coordinates["source-target"] = coords_multi(source, target)
# coordinates["source_end-target"] = coords_multi(end_source, target)
# coordinates["source_end-target_end"] = coords_multi(end_source, end_target)
coordinates["source_target"] = coords(task_source, task_target)
# coordinates["source_source-end"] = coords(task_source, task_source_end)
coordinates["source_inst"] = coords(task_source, inst)
coordinates["source-end_inst"] = coords(task_source_end, inst)

# coordinates["end_target-inst"] = coords(flat(end_target), inst)

coordinates["inst-task_target"] = coords(inst, task_target)

everything = []
for v in coordinates.values():
    everything += v
coordinates["rest"] = [
    (i, j)
    for i in range(len(tokens))
    for j in range(len(tokens))
    if i < j and (i, j) not in everything
]
del coordinates["rest"]
flows = {}
for key, c in coordinates.items():
    flows[key] = sum(attention[:, j, i] for i, j in c)  # / len(c)

for k, v in flows.items():
    plt.plot(v, label=k)
plt.legend()
plt.savefig("out.png", dpi=300)

print(flows)

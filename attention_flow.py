import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt

# todo use avg pooling here
# todo change naming for attentions (format, index), put all in folder if we do more
path = Path("attentions/wmt22_de-en_04shot_wmt21_IDIS_00.npy")
attention = np.load(path)
with open(path.with_suffix(".json")) as f:
    tokens = json.load(f)

print(attention.shape)
# for i, t in enumerate(tokens):
#     print(i, t)

start_and_inst = [0, 1, 2, 3]

source1 = list(range(4, 24))
end_source1 = [24, 25]  # . ->
target1 = list(range(26, 43))
end_target1 = [43, 44]  # . ###

source2 = list(range(45, 62))
end_source2 = [62]  # ->
target2 = list(range(63, 74))
end_target2 = [74, 75]  # . ###

# clean " ?
source3 = list(range(76, 91))
end_source3 = [91, 92]  # . ->
target3 = list(range(93, 103))
end_target3 = [103, 104]  # . ###

source4 = list(range(105, 129))
end_source4 = [129, 130]  # . ->
target4 = list(range(131, 146))
end_target4 = [146, 147]  # . ###

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


def coords(from_tokens, to_tokens):
    return [(i, j) for i in from_tokens for j in to_tokens if i < j]


sources = source1 + source2 + source3 + source4
targets = target1 + target2 + target3 + target4
end_sources = end_source1 + end_source2 + end_source3 + end_source4
end_targets = end_target1 + end_target2 + end_target3 + end_target4
examples = sources + targets + end_sources + end_targets
coordinates = {}
# seperate examples and task??
coordinates["source-source_end"] = coords(sources, end_sources)
coordinates["source-target"] = coords(sources + end_sources, targets)
coordinates["end_targets-inst"] = coords(end_targets, inst)

coordinates["inst-task_target"] = coords(inst, task_target)
coordinates["rest"] = [(i, j) for i in range(len(tokens)) for j in range(len(tokens)) if i < j]

flows = {}
for key, c in coordinates.items():
    flows[key] = sum(attention[:, j, i] for i, j in c) / len(c)

for k, v in flows.items():
    plt.plot(v, label=k)
plt.legend()
plt.savefig("out.png")

print(flows)

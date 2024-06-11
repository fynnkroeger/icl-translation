import pandas as pd
import json
from collections import defaultdict

# dimensions: lang_pair, n_shots, format
# not considered now: model, metric (only kiwi)
lang_pairs = ["de-en", "en-de"]

with open("sorted_out/evals.json") as f:
    evals = json.load(f)
print(evals.values())
print()
print()
lang_to_other = defaultdict(list)
for result in evals.values():
    lang_to_other[result["lang_pair"]].append(
        dict(prompt=result["run_name"], n_shots=result["n_shots"], kiwi22=result["kiwi22"])
    )
for l, v in lang_to_other.items():
    print(
        pd.DataFrame(v)
        .pivot(index="prompt", columns="n_shots", values="kiwi22")
        .reset_index()
        .to_latex(index=False, float_format="%.4f")
    )

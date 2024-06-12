import pandas as pd
import json
from collections import defaultdict

# dimensions: lang_pair, n_shots, format
# not considered now: model, metric (only kiwi)
metric = "bleu"
lang_pairs = ["de-en", "en-de"]

with open("sorted_out/evals.json") as f:
    evals = json.load(f)
# print(evals.values())
print()
print()
lang_to_other = defaultdict(list)
for result in evals.values():
    a = result
    a["run_name"] = a["run_name"].replace("_", " ").replace("format ", "").replace("message", "")
    lang_to_other[result["lang_pair"]].append(a)
    # dict(prompt=result["run_name"], n_shots=result["n_shots"], kiwi22=result["kiwi22"])

for l, v in lang_to_other.items():
    print(
        "\\begin{table}[]\n"
        + pd.DataFrame(v)
        .pivot(index="run_name", columns="n_shots", values=metric)
        .reset_index()
        .rename(columns={"run_name": "format"})
        .to_latex(index=False, float_format="%.4f" if metric == "kiwi22" else "%.2f")
        + "\\caption{"
        + f"{metric} evaluation for {l} translation"
        + "}\n\label{tab:label1}\n\end{table}\n\n"
    )

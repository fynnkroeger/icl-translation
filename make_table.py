import pandas as pd
import json
from collections import defaultdict
import utils

metrics = ["kiwi22", "bleu", "chrf"]
lang_pairs = ["de-en", "en-de", "de-fr", "fr-de"]

with open(f"Mistral-7B-v0.1/evals.json") as f:
    evals = json.load(f)

lang_to_other = defaultdict(list)
for result in evals.values():
    a = result
    a["run_name"] = utils.print_names[a["run_name"]]
    lang_to_other[result["lang_pair"]].append(a)


s = ""
for metric in metrics:
    s += "\\begin{table*}[htbp]\n\\centering\n"
    for lang_pair in lang_pairs:
        s += (
            pd.DataFrame(lang_to_other[lang_pair])
            .pivot(index="run_name", columns="n_shots", values=metric)
            .reset_index()
            .rename(columns={"run_name": lang_pair, 0: "0-shot", 1: "1-shot", 4: "4-shot"})
            .set_index(lang_pair)
            .reindex(["arrow oneline", "arrow", "arrow title", "label title"])
            .reset_index()
            .to_latex(index=False, float_format="%.4f" if metric == "kiwi22" else "%.2f")
            .replace("NaN", "–")
        )
    s += (
        "\\caption{"
        + f"{metric} evaluation"
        + "}\n\label{tab:"
        + f"{metric}"
        + "}\n\end{table*}\n\n"
    )

with open("Mistral-7B-v0.1/tables.tex", "w") as f:
    f.write(s)

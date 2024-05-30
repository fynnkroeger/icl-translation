import subprocess
import json
from pathlib import Path

overwrite = False
Path("datasets").mkdir(exist_ok=True)

for dataset in ["wmt21", "wmt22", "wmt23"]:
    for lang_pair in ["en-de", "de-en"]:
        out_path = Path(f"datasets/{dataset}_{lang_pair}.json")
        if out_path.exists() and not overwrite:
            continue
        output = []
        src = subprocess.run(f"sacrebleu -t {dataset} -l {lang_pair} --echo src", shell=True, capture_output=True)
        trgt = subprocess.run(f"sacrebleu -t {dataset} -l {lang_pair} --echo ref", shell=True, capture_output=True)
        src_lines = src.stdout.decode().splitlines()
        trgt_lines = trgt.stdout.decode().splitlines()
        for source, target in zip(src_lines, trgt_lines, strict=True):
            output.append(dict(source=source, target=target))
        with open(out_path, "w") as f:
            json.dump(output, f, indent=1)
        print(dataset, lang_pair, len(output))

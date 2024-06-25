from pathlib import Path
import json
import sacrebleu
from comet import download_model, load_from_checkpoint
import re

model_path = download_model("Unbabel/wmt22-cometkiwi-da")
comet_model = load_from_checkpoint(model_path)
model_path = Path("Mistral-7B-v0.1")
(model_path / "sorted").mkdir(exist_ok=True)
save = True
# hierarchichal structure for different datasets/evals?
eval_output = {}
if (eval_file := (model_path / "evals.json")).exists():
    eval_output = json.loads(eval_file.read_text())
new_eval = False
for path in sorted((model_path / "outputs").iterdir()):
    if path.name in eval_output or "logs" in path.name:
        continue

    output = json.loads(path.read_text())
    references = [[d["target"] for d in output]]
    translations = []
    for sample in output:
        pattern = r"\n|###" if "[" in sample["source"] else r"\n|###|\["
        cut = re.split(pattern, sample["translation"], maxsplit=1)[0]
        translations.append(cut.strip())
    sources = [d["source"] for d in output]
    source_lang, target_lang = path.stem.split("_")[1].split("-")
    bleu = sacrebleu.metrics.BLEU(trg_lang=target_lang)
    chrf = sacrebleu.metrics.CHRF(word_order=2)
    comet_score = comet_model.predict([dict(src=s, mt=t) for s, t in zip(sources, translations)])

    print(path.name, len(translations))
    print(bs := bleu.corpus_score(translations, references))
    print(cs := chrf.corpus_score(translations, references))
    print(f"comet kiwi22 = {comet_score.system_score:.3f}")
    print()
    if not save:
        continue
    new_eval = True
    logs = json.loads((model_path / "logs.json").read_text())
    eval_output[path.name] = dict(
        **logs[path.name],
        kiwi22=round(comet_score.system_score, 4),
        chrf=round(cs.score, 2),
        bleu=round(bs.score, 2),
    )

    bleu1 = sacrebleu.metrics.BLEU(trg_lang=target_lang, effective_order=True)
    chrf1 = sacrebleu.metrics.CHRF(word_order=2)
    scored = []
    for i, (ref, trans, out) in enumerate(zip(references[0], translations, output)):
        chrf = round(chrf1.sentence_score(trans, [ref]).score, 2)
        bleu = round(bleu1.sentence_score(trans, [ref]).score, 2)
        kiwi22 = round(comet_score.scores[i], 4)
        out.update(translation=trans, kiwi22=kiwi22, chrf=chrf, bleu=bleu, index=i)
        scored.append(out)
    scored.sort(key=lambda d: d["kiwi22"])

    with open(model_path / f"sorted/{path.name}", "w") as f:
        json.dump(scored, f, indent=1)

if new_eval:
    with open(eval_file, "w") as f:
        sorted_evals = dict(sorted([(k, v) for k, v in eval_output.items()]))
        json.dump(sorted_evals, f, indent=1)

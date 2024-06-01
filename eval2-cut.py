from pathlib import Path
import json
import sacrebleu

Path("sorted_out").mkdir(exist_ok=True)

# hierarchichal structure for different datasets/evals?
eval_output = {}
if (eval_file := Path("sorted_out/evals.json")).exists():
    eval_output = json.loads(eval_file.read_text())
new_eval = False
for path in sorted(Path("outputs").iterdir()):
    if path.name in eval_output:
        continue
    new_eval = True

    output = json.loads(path.read_text())
    references = [[d["target"] for d in output]]
    translations = [d["translation"].split("\n")[0] for d in output]

    source_lang, target_lang = path.stem.split("_")[1].split("-")
    bleu = sacrebleu.metrics.BLEU(trg_lang=target_lang)
    chrf = sacrebleu.metrics.CHRF(word_order=2)
    print(path.name, len(translations))
    print(bs := bleu.corpus_score(translations, references))
    print(cs := chrf.corpus_score(translations, references))
    print()
    eval_output[path.name] = dict(chrf=cs.score, bleu=bs.score)

    bleu1 = sacrebleu.metrics.BLEU(trg_lang=target_lang, effective_order=True)
    chrf1 = sacrebleu.metrics.CHRF(word_order=2)
    scored = []
    for i, (ref, trans, out) in enumerate(zip(references[0], translations, output)):
        chrf = chrf1.sentence_score(trans, [ref]).score
        bleu = bleu1.sentence_score(trans, [ref]).score
        out.update(translation=trans, chrf=chrf, bleu=bleu, index=i)
        scored.append(out)
    scored.sort(key=lambda d: d["chrf"])

    with open(f"sorted_out/{path.name}", "w") as f:
        json.dump(scored, f, indent=1)

if new_eval:
    with open(eval_file, "w") as f:
        sorted_evals = dict(sorted([(k, v) for k, v in eval_output.items()]))
        json.dump(sorted_evals, f, indent=1)

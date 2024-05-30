from pathlib import Path
import json
import sacrebleu
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

Path("sorted_out").mkdir(exist_ok=True)

plot_data = {}
for path in sorted(Path("outputs").iterdir()):
    output = json.loads(path.read_text())
    references = [[d["target"] for d in output]]
    translations = [d["translation"].split("\n")[0] for d in output]

    source_lang, target_lang = path.stem.split("_")[1].split("-")
    bleu = sacrebleu.metrics.BLEU(trg_lang=target_lang)
    chrf = sacrebleu.metrics.CHRF(word_order=2)
    print(path.name, len(translations))
    print(bleu.corpus_score(translations, references))
    print(chrf.corpus_score(translations, references))
    print()

    bleu1 = sacrebleu.metrics.BLEU(trg_lang=target_lang, effective_order=True)
    chrf1 = sacrebleu.metrics.CHRF(word_order=2)
    scored = []
    for i, (ref, trans, out) in enumerate(zip(references[0], translations, output)):
        chrf = chrf1.sentence_score(trans, [ref]).score
        bleu = bleu1.sentence_score(trans, [ref]).score
        out.update(translation=trans, chrf=chrf, bleu=bleu, index=i)
        scored.append(out)
    scored.sort(key=lambda d: d["chrf"])
    plot_data[path.stem] = np.array([s["chrf"] for s in scored])

    with open(f"sorted_out/{path.name}", "w") as f:
        json.dump(scored, f, indent=1)


Path("plots").mkdir(exist_ok=True)
sns.kdeplot(plot_data)
plt.savefig("plots/kde.png", dpi=300)
plt.figure()
sns.kdeplot(plot_data, cumulative=True)
plt.savefig("plots/kde_cumulative.png", dpi=300)

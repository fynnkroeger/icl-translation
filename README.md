# In Context Learning for Text Generation

Here is a quick rundown of the relevant files to reproduce the results:

## Installing requirements
Install conda or miniconda, then run the following command. This might take some minutes due to installing pytorch.
```
conda env create -f environment.yaml
conda activate icl-translation
```
Then you can run all the files with `python file.py`.

## download_datasets.py
Running this file downloads the en-de, de-en, de-fr and fr-de language pairs for the wmt22 and wmt23 datasets using sacrebleu and saves them into `datasets/`.

## inference.py
Running this file runs the downloaded datasets through `Mistral-7B-v0.1`, for each of the four prompt formats and the different shot numbers.
This requires a GPU and can take a up to several hours.
The `batch_size` at the very bottom of the file can be changed to adapt the memory requirements. 
The generations are saved to `Mistral-7B-v0.1/outputs` in json format.

The function to generate the few shot prompts are also found in this file.

## evaluation.py
Running this file evaluates the outputs of the previous step and saves the results to `Mistral-7B-v0.1/evals.json`.
COMET, BLEU and chrf2++ are used to evaluate translations.
The outputs of the model are cut off before a newline or the character that is used to join the few shot examples, as the model sometimes continued generation.

## make_table.py
This file can be used to generate LaTeX tables from the `evals.json` file.

## inference_attention.py 
This file can be used to create attention matrices that are used for the main analysis.
It runs inference on the model but with batch size 1 and saves the attention maps to `Mistral-7B-v0.1/attention`.

## attention_flow.py
Uses the attention matrices from above to create the main analysis line plots and the matrix plots visualizing the different patterns.

## heatmap.py
Uses the attention matrices to compute heatmaps for exploration and visualization.

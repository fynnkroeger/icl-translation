# In Context Learning for Text Generation

Here is a quick rundown of the relevant files to reproduce the results:

## Installing requirements

## download_datasets.py
Running this file downloads the en-de, de-en, de-fr and fr-de language pairs for the wmt22 and wmt23 datasets using sacrebleu and saves them into `datasets/`.

## inference.py
Running this file runs the downloaded datasets through `Mistral-7B-v0.1`, for each of the four prompt formats and the different shot numbers.
This requires a GPU and can take a up to several hours.
The `batch_size` at the very bottom of the file can be changed to adapt the memory requirements. 
The generations are saved to `Mistral-7B-v0.1/outputs` in json format.
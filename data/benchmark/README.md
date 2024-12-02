# RAGChecker Benchmark

Please take the following steps to get the benchmark dataset.

### Download raw data

#### BioASQ

Please login to [BioASQ](http://participants-area.bioasq.org/datasets/), then do the following:

- In `Datasets for task a`, download `allMeSH_2022.zip` throuth the entry of `Training v.2022 (txt)` in the table. Unzip it to `raw_data/bioasq/allMeSH_2022.json`. Note that this JSON file is of 27G large, please make sure you have enough disk space.

- In `Datasets for task a`, download files throuth the links column `Test data` in the table from 2014 to 2023. Unzip the files and put the JSON files into the folder `raw_data/bioasq`:
    - `{2~9}B{1~5}_golden.json`
    - `10B{1~6}_golden.json`
    - `11B{1~4}_golden.json`


#### LoTTE

Download the LoTTE corpus here: https://downloads.cs.stanford.edu/nlp/data/colbert/colbertv2/lotte.tar.gz and unzip to folder `raw_data`.

#### NovelQA

Get access to NovelQA dataset: https://huggingface.co/datasets/NovelQA/NovelQA . Then login your huggingface account:
```bash
pip install huggingface_hub
huggingface-cli login
```

### Run data processing script

Run the following script, the benchmark dataset will be processed to the folder `processed_data`: 

```bash
sh data_process.sh
```
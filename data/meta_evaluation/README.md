## Meta Evaluation of RAGChecker

You can use the code and data in this folder to reproduce the meta evaluation results of RAGChecker.

### Data

The human labeled data of meta evaluation can be found at `human_labeled_data.json`. We have a total of 280 data instances for meta evaluation, each of them is labeled by two annotators, so this file contains 560 data instances. Please refer to our paper for details of meta evaluation data construction.

The evaluation output of baseline RAG evaluation systems are in the rest 5 json files:
- `baseline_trulens_llama3.json`, evaluation output of Trulens.
- `baseline_ares_llama3.json`, evaluation output of ARES.
- `baseline_ragas_llama3.json`, evaluation output of RAGAS.
- `baseline_crud_llama3.json`, evaluation output of CRUD-RAG.
- `baseline_ragchecker.json`, evaluation output of RAGChecker.

### Script

Execute the following command to run the meta evaluation script:
```bash
python meta_eval.py
```

You will get the meta evaluation results in `meta_eval_results.json` corresponding to the Table 2 and Table 5 in our paper. Also, the script will output three violin plot files corresponding to the meta evaluation results of appendix C in our paper.
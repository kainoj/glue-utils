# Utils to quickly eval many 🤗models on the GLUE tasks

Setup
```bash
wget https://raw.githubusercontent.com/huggingface/transformers/master/examples/pytorch/text-classification/run_glue.py
conda env create -f environment.yaml
conda activate glue-utils
```

Eval a model on every glue task:
```bash
python runner.py path/to/model
```

This will store results in `path/to/model/glue/[task_name]/`.

To quickly gather all results, run:
```bash
python gather_scores.py path/to/model/glue/
```
# Utils to quickly eval many 🤗models on the GLUE tasks

Setup
```bash
wget https://raw.githubusercontent.com/huggingface/transformers/master/examples/pytorch/text-classification/run_glue.py
conda env create -f environment.yaml
conda activate glue-utils
```

Eval multiple models on every glue task:
```bash
python runner.py \
    path/to/model_1 \
    path/to/model_2 \
    path/to/model_3 \
    --gpu_idx 1
```

This will store results in `path/to/model_N/glue/[task_name]/`, for each model, for each task.

To quickly gather all results, run:
```bash
python gather_scores.py path/to/model/glue/
```
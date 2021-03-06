# Utils to quickly evaluate many 🤗models on the GLUE tasks

Setup
```bash
wget https://raw.githubusercontent.com/huggingface/transformers/master/examples/pytorch/text-classification/run_glue.py
conda env create -f environment.yaml
conda activate glue-utils
```

Evaluate multiple models on every glue task:
```bash
python runner.py \
    path/to/model_1/ \
    path/to/model_2/ \
    path/to/model_3/ \
    --gpu_idx 1
```

This will store results in `path/to/model_N/glue/[task_name]/`, for each model, for each task.

To quickly gather all results, run:
```bash
python gather_scores.py \
    path/to/model_1/glue/ \
    path/to/model_2/glue/ \
    path/to/model_3/glue/
```


### Reproduce the environment
If something goes wrong with the setup, follow these steps:

```bash
conda create --name py35 python=3.5
conda activate glue
conda install pytorch torchvision cudatoolkit=11.3 -c pytorch
pip install git+https://github.com/huggingface/transformers
pip install "datasets>=1.8.0" scipy scikit-learn
conda env export > environment.yaml 
```

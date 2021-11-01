import os
from pathlib import Path
from run_glue import main as glue


# sorted by execution time
TASKS = ['stsb', 'mrpc', 'cola', 'wnli', 'sst2', 'qnli', 'rte', 'qqp', 'mnli',]


def main():
    pretrained = None
    pretrained = Path(pretrained)

    if not pretrained.exists():
        raise ValueError("Model not found!")

    print(f'Evaluating {pretrained}.')

    for task in TASKS:
        print(f"Task: {task}")

        output_dir = pretrained / "glue" / task

        cmd = f"""\
        CUDA_VISIBLE_DEVICES=1 python run_glue.py \
            --model_name_or_path  {pretrained} \
            --task_name {task} \
            --do_train \
            --do_eval \
            --max_seq_length 128 \
            --per_device_train_batch_size 32 \
            --learning_rate 2e-5 \
            --num_train_epochs 3 \
            --output_dir {output_dir} \
            --fp16
        """
        os.system(cmd)


if __name__ == '__main__':
    main()

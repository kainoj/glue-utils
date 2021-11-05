import os
import argparse

from pathlib import Path

# sorted by execution time
TASKS = ['stsb', 'mrpc', 'cola', 'wnli', 'sst2', 'qnli', 'rte', 'qqp', 'mnli']


def run_glues(model_path, gpu_idx: str = '0'):
    model_path = Path(model_path)

    if not model_path.exists():
        raise ValueError("Model not found!")

    print(f'Evaluating {model_path}.')

    for task in TASKS:
        print(f"Task: {task}")

        output_dir = model_path / "glue" / task

        cmd = f"""\
        CUDA_VISIBLE_DEVICES={gpu_idx} python run_glue.py \
            --model_name_or_path  {model_path} \
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', help='Path to pre-trained model.')
    parser.add_argument(
        '--gpu_idx',
        help='GPU index, as used in CUDA_VISIBLE_DEVICES.'
    )

    args = parser.parse_args()
    run_glues(args.model_path, gpu_idx=args.gpu_idx)

    print(f"Finished:\n{args.model_name}/glue")


if __name__ == '__main__':
    main()

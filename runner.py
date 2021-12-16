import os
import argparse

from pathlib import Path
from gather_scores import gather_scores, pretty_print


# sorted by execution time
TASKS = ['stsb', 'mrpc', 'cola', 'wnli', 'sst2', 'qnli', 'rte', 'qqp', 'mnli']


def run_glue(model_path, task, output_dir_task, gpu_idx):
    cmd = f"""\
        CUDA_VISIBLE_DEVICES={gpu_idx} python run_glue.py \
            --model_name_or_path {model_path} \
            --task_name {task} \
            --do_train \
            --do_eval \
            --max_seq_length 128 \
            --per_device_train_batch_size 32 \
            --learning_rate 2e-5 \
            --num_train_epochs 3 \
            --output_dir {output_dir_task} \
            --fp16
        """
    os.system(cmd)


def run_glues(model_path, gpu_idx: str = '0'):
    model_path = Path(model_path)

    if not model_path.exists():
        raise ValueError("Model not found!")

    print(f'Evaluating {model_path}.')

    output_dir = model_path / "glue"

    for task in TASKS:
        print(f"Task: {task}")

        output_dir_task = output_dir / task
        run_glue(model_path, task, output_dir_task, gpu_idx)

    return output_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'model_paths',
        nargs='+',
        help='Path(s) to pre-trained model.'
    )
    parser.add_argument(
        '--gpu_idx',
        default='0',
        help='GPU index, as used in CUDA_VISIBLE_DEVICES.'
    )

    args = parser.parse_args()

    glue_out_paths = []

    for model_path in args.model_paths:
        out_path = run_glues(model_path, gpu_idx=args.gpu_idx)
        glue_out_paths.append(out_path)

    print("\n========== Finished ==========\n")

    for path in glue_out_paths:
        scores = gather_scores(path)
        print(f"\n\nModel:\n{path}")
        pretty_print(scores)


if __name__ == '__main__':
    main()

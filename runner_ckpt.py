import argparse

from runner import run_glue, TASKS
from pathlib import Path
from gather_scores import gather_scores, pretty_print


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'model_paths',
        nargs='+',
        help='Path(s) to pre-trained model checkpoints.'
    )
    parser.add_argument(
        '--gpu_idx',
        default='0',
        help='GPU index, as used in CUDA_VISIBLE_DEVICES.'
    )
    parser.add_argument(
        '--task',
        choices=TASKS
    )

    args = parser.parse_args()

    glue_out_paths = []

    for model_path in args.model_paths:

        output_dir = Path(model_path) / 'glue'

        run_glue(
            model_path=model_path,
            task=args.task,
            output_dir_task=output_dir / args.task,
            gpu_idx=args.gpu_idx,
            do_train=False
        )

        glue_out_paths.append(output_dir)

    print("\n========== Finished ==========\n")

    for path in glue_out_paths:
        print(f"\n\nModel:\n{path}")
        scores = gather_scores(path)
        pretty_print(scores)


if __name__ == '__main__':
    main()

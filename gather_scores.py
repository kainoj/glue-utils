import argparse
import json
import logging
from pathlib import Path


logger = logging.getLogger(__name__)


# sorted by execution time
TASKS = ['cola', 'sst2', 'mrpc', 'stsb', 'qqp', 'mnli', 'qnli', 'rte', 'wnli']

METRIC_MAP = {
    'cola': 'eval_matthews_correlation',
    'sst2': 'eval_accuracy',
    'mrpc': 'eval_combined_score',
    'stsb': 'eval_combined_score',
    'qqp':  'eval_combined_score',
    'mnli': 'eval_accuracy',
    'qnli': 'eval_accuracy',
    'rte':  'eval_accuracy',
    'wnli': 'eval_accuracy',
}


def get_score(results_json_file, score_key_name):
    with open(results_json_file) as f:
        data = json.load(f)
        return data[score_key_name]


def pretty_print(scores, sep='\t'):
    for task in TASKS:
        print(task.upper(), end=sep)

    print()

    for task in TASKS:
        print(scores[task], end=sep)


def gather_scores(path_to_glue_output):
    scores = {}

    for task in TASKS:
        results_file = path_to_glue_output / task / 'all_results.json'
        if results_file.exists():
            y = get_score(results_file, METRIC_MAP[task])
            scores[task] = y
        else:
            logging.warning(f" all_results.json for task '{task}' not found.")
            scores[task] = float("NaN")

    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'path_to_glue_output',
        help='GLUE path contains folders named by tasks names, each must '
             'contain `all_results.json`'
        )
    args = parser.parse_args()

    path_to_glue_output = Path(args.path_to_glue_output)

    scores = gather_scores(path_to_glue_output)
    pretty_print(scores)


if __name__ == '__main__':
    main()

# main.py

import argparse
from evaluations.boolq_eval import evaluate_boolq
from evaluations.hellaswag_eval import evaluate_hellaswag
from evaluations.winogrande_eval import evaluate_winogrande
from evaluations.rte_eval import evaluate_rte
from evaluations.piqa_eval import evaluate_piqa
from evaluations.commonsenseqa_eval import evaluate_commonsenseqa
from evaluations.multirc_eval import evaluate_multirc
from evaluations.arc_eval import evaluate_arc
from evaluations.cb_eval import evaluate_cb
from models.model_loader import ModelWrapper

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Model Evaluation Suite")
    parser.add_argument('--model', type=str, required=True, help='Model name to evaluate')
    parser.add_argument('--evaluations', nargs='+', default=['boolq', 'hellaswag', 'winogrande', 'rte', 'piqa', 'commonsenseqa', 'multirc', 'arc', 'cb'], help='List of evaluations to run')
    parser.add_argument('--sample-size', type=int, default=None, help='Number of samples to evaluate from each dataset')
    args = parser.parse_args()

    # Load the model
    model = ModelWrapper(args.model)

    # Dictionary mapping evaluation names to functions
    evaluation_functions = {
        'boolq': evaluate_boolq,
        'hellaswag': evaluate_hellaswag,
        'winogrande': evaluate_winogrande,
        'rte': evaluate_rte,
        'piqa': evaluate_piqa,
        'commonsenseqa': evaluate_commonsenseqa,
        'multirc': evaluate_multirc,
        'arc': evaluate_arc,
        'cb': evaluate_cb,
    }

    # Run evaluations
    for eval_name in args.evaluations:
        if eval_name in evaluation_functions:
            print(f"Starting evaluation: {eval_name}")
            accuracy = evaluation_functions[eval_name](model, sample_size=args.sample_size)
            print(f"{eval_name} Accuracy: {accuracy:.2f}%\n")
        else:
            print(f"Evaluation {eval_name} not found.")

if __name__ == '__main__':
    main()

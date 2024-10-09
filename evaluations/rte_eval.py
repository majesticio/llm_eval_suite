# evaluations/rte_eval.py

from datasets import load_dataset
from tqdm import tqdm

def evaluate_rte(model, sample_size=None):
    """
    Evaluate the model on the RTE (Recognizing Textual Entailment) dataset.

    Args:
        model: The model object with a `chat` method.
        sample_size: Number of samples to use for evaluation (default is entire dataset).
    """
    rte_dataset = load_dataset("super_glue", "rte", split="validation")

    # If sample_size is provided, select a subset of the dataset
    if sample_size:
        rte_dataset = rte_dataset.select(range(min(sample_size, len(rte_dataset))))

    correct = 0
    total = 0

    for idx, example in enumerate(tqdm(rte_dataset, desc="Evaluating RTE")):
        premise = example['premise']
        hypothesis = example['hypothesis']
        correct_answer = bool(example['label'])  # 1 for entailment, 0 for contradiction

        prompt = f"""
Premise: {premise}
Hypothesis: {hypothesis}
Is the hypothesis entailed by the premise? Please answer 'Yes' or 'No'.

Answer:""".strip()

        try:
            response = model.chat(prompt)
            predicted_answer = extract_rte_answer(response)

            if predicted_answer == correct_answer:
                correct += 1
            total += 1

        except Exception as e:
            print(f"Error processing example {idx}: {e}")
            continue

    accuracy = correct / total * 100 if total > 0 else 0
    print(f"RTE Accuracy: {accuracy:.2f}%")
    return accuracy

def extract_rte_answer(response):
    response = response.strip().lower()
    if 'yes' in response:
        return True
    elif 'no' in response:
        return False
    else:
        return None

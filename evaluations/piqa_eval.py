# evaluations/piqa_eval.py

from datasets import load_dataset
from tqdm import tqdm

def evaluate_piqa(model, sample_size=None):
    """
    Evaluate the model on the PIQA dataset.

    Args:
        model: The model object with a `chat` method.
        sample_size: Number of samples to use for evaluation (default is entire dataset).
    """
    piqa_dataset = load_dataset("piqa", split="validation")

    # If sample_size is provided, select a subset of the dataset
    if sample_size:
        piqa_dataset = piqa_dataset.select(range(min(sample_size, len(piqa_dataset))))

    correct = 0
    total = 0

    for idx, example in enumerate(tqdm(piqa_dataset, desc="Evaluating PIQA")):
        goal = example['goal']
        solution1 = example['sol1']
        solution2 = example['sol2']
        correct_answer = example['label']

        prompt = f"""
Goal: {goal}
Solution 1: {solution1}
Solution 2: {solution2}
Which solution is more plausible, Solution 1 or Solution 2?

Answer:""".strip()

        try:
            response = model.chat(prompt)
            predicted_answer = extract_piqa_answer(response)

            if predicted_answer == correct_answer:
                correct += 1
            total += 1

        except Exception as e:
            print(f"Error processing example {idx}: {e}")
            continue

    accuracy = correct / total * 100 if total > 0 else 0
    print(f"PIQA Accuracy: {accuracy:.2f}%")
    return accuracy

def extract_piqa_answer(response):
    response = response.strip().lower()
    if 'solution 1' in response:
        return 0
    elif 'solution 2' in response:
        return 1
    else:
        return None

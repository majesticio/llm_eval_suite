# evaluations/cb_eval.py

from datasets import load_dataset
from tqdm import tqdm

def evaluate_cb(model, sample_size=None):
    """
    Evaluate the model on the CB dataset (CommitmentBank).

    Args:
        model: The model object with a `chat` method.
        sample_size: Number of samples to use for evaluation (default is entire dataset).
    """
    cb_dataset = load_dataset("super_glue", "cb", split="validation")

    # If sample_size is provided, select a subset of the dataset
    if sample_size:
        cb_dataset = cb_dataset.select(range(min(sample_size, len(cb_dataset))))

    correct = 0
    total = 0

    for idx, example in enumerate(tqdm(cb_dataset, desc="Evaluating CB")):
        premise = example['premise']
        hypothesis = example['hypothesis']
        correct_answer = example['label']  # 0: entailment, 1: contradiction, 2: neutral

        prompt = create_cb_prompt(premise, hypothesis)

        try:
            response = model.chat(prompt)
            predicted_answer = extract_cb_answer(response)

            if predicted_answer == correct_answer:
                correct += 1
            total += 1

        except Exception as e:
            print(f"Error processing example {idx}: {e}")
            continue

    accuracy = correct / total * 100 if total > 0 else 0
    print(f"CB Accuracy: {accuracy:.2f}%")
    return accuracy

def create_cb_prompt(premise, hypothesis):
    prompt = f"""
Premise: {premise}
Hypothesis: {hypothesis}
Is the hypothesis entailed by the premise, contradicted by the premise, or neutral with respect to the premise?

Answer:""".strip()
    return prompt

def extract_cb_answer(response):
    response = response.strip().lower()
    if 'entailed' in response or 'entailment' in response:
        return 0
    elif 'contradicted' in response or 'contradiction' in response:
        return 1
    elif 'neutral' in response:
        return 2
    else:
        return None

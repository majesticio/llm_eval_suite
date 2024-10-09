# evaluations/boolq_eval.py

from datasets import load_dataset
from tqdm import tqdm
import re

def evaluate_boolq(model, sample_size=None):
    """
    Evaluate the model on the BoolQ dataset.

    Args:
        model: The model object with a `chat` method.
        sample_size: Number of samples to use for evaluation (default is entire dataset).
    """
    # Load the BoolQ dataset
    boolq_dataset = load_dataset("super_glue", "boolq", split="validation")

    # If sample_size is provided, select a subset of the dataset
    if sample_size:
        boolq_dataset = boolq_dataset.select(range(min(sample_size, len(boolq_dataset))))

    correct = 0
    total = 0

    for idx, example in enumerate(tqdm(boolq_dataset, desc="Evaluating BoolQ")):
        passage = example['passage']
        question = example['question']
        correct_answer = bool(example['label'])

        prompt = f"""
Passage:
{passage}

Question: {question}
Please answer 'Yes' or 'No' based on the information provided in the passage.

Answer:""".strip()

        try:
            response = model.chat(prompt)
            predicted_answer = extract_boolq_answer(response)

            if predicted_answer is not None:
                if predicted_answer == correct_answer:
                    correct += 1
                total += 1
            else:
                print(f"Warning: Invalid response from model: {response}")
                total += 1  # Count as incorrect

        except Exception as e:
            print(f"Error processing example {idx}: {e}")
            continue

    accuracy = correct / total * 100 if total > 0 else 0
    print(f"BoolQ Accuracy: {accuracy:.2f}%")
    return accuracy

def extract_boolq_answer(response):
    response = response.strip().lower()
    response = response.strip('.,"\' ')
    if any(word in response for word in ['yes', 'yeah', 'yep', 'correct', 'true', 'affirmative']):
        return True
    elif any(word in response for word in ['no', 'nope', 'nah', 'false', 'negative']):
        return False
    else:
        return None

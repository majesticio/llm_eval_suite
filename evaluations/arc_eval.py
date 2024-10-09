# evaluations/arc_eval.py

from datasets import load_dataset
from tqdm import tqdm
import re

def evaluate_arc(model, sample_size=None):
    """
    Evaluate the model on the AI2 ARC dataset.

    Args:
        model: The model object with a `chat` method.
        sample_size: Number of samples to use for evaluation (default is entire dataset).
    """
    arc_dataset = load_dataset("ai2_arc", "ARC-Challenge", split="validation")

    # If sample_size is provided, select a subset of the dataset
    if sample_size:
        arc_dataset = arc_dataset.select(range(min(sample_size, len(arc_dataset))))

    correct = 0
    total = 0

    for idx, example in enumerate(tqdm(arc_dataset, desc="Evaluating ARC")):
        question = example['question']
        choices_labels = example['choices']['label']  # ['A', 'B', 'C', 'D']
        choices_texts = example['choices']['text']    # List of choice texts
        choices = dict(zip(choices_labels, choices_texts))  # Map labels to texts

        correct_answer = example['answerKey']  # 'A', 'B', 'C', 'D', or 'E'

        prompt = create_arc_prompt(question, choices)

        try:
            response = model.chat(prompt)
            predicted_answer = extract_arc_answer(response)

            if predicted_answer == correct_answer:
                correct += 1
            total += 1

        except Exception as e:
            print(f"Error processing example {idx}: {e}")
            continue

    accuracy = correct / total * 100 if total > 0 else 0
    print(f"ARC Accuracy: {accuracy:.2f}%")
    return accuracy

def create_arc_prompt(question, choices):
    choices_str = '\n'.join([f"{label}. {text}" for label, text in choices.items()])
    prompt = f"""
Question: {question}
Choices:
{choices_str}
Please select the correct choice (A, B, C, D, or E).

Answer:""".strip()
    return prompt

def extract_arc_answer(response):
    response = response.strip().upper()
    match = re.search(r'\b([ABCDE])\b', response)
    if match:
        return match.group(1)
    else:
        return None

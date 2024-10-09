# evaluations/commonsenseqa_eval.py

from datasets import load_dataset
from tqdm import tqdm
import re

def evaluate_commonsenseqa(model, sample_size=None):
    """
    Evaluate the model on the CommonSenseQA dataset.

    Args:
        model: The model object with a `chat` method.
        sample_size: Number of samples to use for evaluation (default is entire dataset).
    """
    commonsenseqa_dataset = load_dataset("commonsense_qa", split="validation")

    # If sample_size is provided, select a subset of the dataset
    if sample_size:
        commonsenseqa_dataset = commonsenseqa_dataset.select(range(min(sample_size, len(commonsenseqa_dataset))))

    correct = 0
    total = 0

    for idx, example in enumerate(tqdm(commonsenseqa_dataset, desc="Evaluating CommonSenseQA")):
        question = example['question']  # 'question' is a string
        choices_labels = example['choices']['label']  # List of labels
        choices_texts = example['choices']['text']    # List of texts
        choices = dict(zip(choices_labels, choices_texts))  # Create a dictionary mapping labels to texts

        correct_answer = example['answerKey']  # Should be 'A', 'B', 'C', 'D', or 'E'

        prompt = create_commonsenseqa_prompt(question, choices)

        try:
            response = model.chat(prompt)
            predicted_answer = extract_commonsenseqa_answer(response)

            if predicted_answer == correct_answer:
                correct += 1
            total += 1

        except Exception as e:
            print(f"Error processing example {idx}: {e}")
            continue

    accuracy = (correct / total) * 100 if total > 0 else 0
    print(f"CommonSenseQA Accuracy: {accuracy:.2f}%")
    return accuracy

def create_commonsenseqa_prompt(question, choices):
    choices_str = '\n'.join([f"{label}. {text}" for label, text in choices.items()])
    prompt = f"""
Question: {question}
Choices:
{choices_str}
Please select the correct choice (A, B, C, D, or E).

Answer:""".strip()
    return prompt

def extract_commonsenseqa_answer(response):
    response = response.strip().upper()
    match = re.search(r'\b([ABCDE])\b', response)
    if match:
        return match.group(1)
    else:
        return None

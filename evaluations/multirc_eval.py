# evaluations/multirc_eval.py

from datasets import load_dataset
from tqdm import tqdm
import re

def evaluate_multirc(model, sample_size=None):
    """
    Evaluate the model on the MultiRC dataset.

    Args:
        model: The model object with a `chat` method.
        sample_size: Number of samples to use for evaluation (default is the entire dataset).
    """
    multirc_dataset = load_dataset("super_glue", "multirc", split="validation")

    # If sample_size is provided, select a subset of the dataset
    if sample_size:
        multirc_dataset = multirc_dataset.select(range(min(sample_size, len(multirc_dataset))))

    correct = 0
    total = 0

    for idx, example in enumerate(tqdm(multirc_dataset, desc="Evaluating MultiRC")):
        paragraph = example['paragraph']
        question = example['question']
        answer = example['answer']
        correct_label = bool(example['label'])  # Convert label to boolean

        prompt = f"""
Paragraph:
{paragraph}

Question: {question}
Proposed Answer: {answer}

Is the proposed answer correct based on the paragraph? Please answer 'Yes' or 'No'.

Answer:""".strip()

        try:
            response = model.chat(prompt)
            predicted_label = extract_multirc_answer(response)

            if predicted_label is not None:
                if predicted_label == correct_label:
                    correct += 1
                total += 1
            else:
                print(f"Warning: Invalid response from model: {response}")
                total += 1  # Count as incorrect

        except Exception as e:
            print(f"Error processing example {idx}: {e}")
            continue

    accuracy = correct / total * 100 if total > 0 else 0
    print(f"MultiRC Accuracy: {accuracy:.2f}%")
    return accuracy

def extract_multirc_answer(response):
    response = response.strip().lower()
    # Remove punctuation
    response = response.strip('.,"\' ')
    # Check for affirmative or negative responses
    if any(word in response for word in ['yes', 'yeah', 'yep', 'correct', 'true', 'affirmative']):
        return True
    elif any(word in response for word in ['no', 'nope', 'nah', 'false', 'negative', 'incorrect']):
        return False
    else:
        return None

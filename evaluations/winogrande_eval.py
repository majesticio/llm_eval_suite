# evaluations/winogrande_eval.py

from datasets import load_dataset
from tqdm import tqdm
import re

def evaluate_winogrande(model, sample_size=None):
    """
    Evaluate the model on the WinoGrande dataset.

    Args:
        model: The model object with a `chat` method.
        sample_size: Number of samples to use for evaluation (default is entire dataset).
    """
    winogrande_dataset = load_dataset("winogrande", "winogrande_xl", split="validation")

    # If sample_size is provided, select a subset of the dataset
    if sample_size:
        winogrande_dataset = winogrande_dataset.select(range(min(sample_size, len(winogrande_dataset))))

    correct = 0
    total = 0

    for idx, example in enumerate(tqdm(winogrande_dataset, desc="Evaluating WinoGrande")):
        sentence = example['sentence']
        option1 = example['option1']
        option2 = example['option2']
        correct_answer = example['answer']

        prompt = create_winogrande_prompt(sentence, option1, option2)

        try:
            response = model.chat(prompt)
            predicted_answer = extract_winogrande_answer(response)

            if predicted_answer == correct_answer:
                correct += 1
            total += 1

        except Exception as e:
            print(f"Error processing example {idx}: {e}")
            continue

    accuracy = correct / total * 100 if total > 0 else 0
    print(f"WinoGrande Accuracy: {accuracy:.2f}%")
    return accuracy

def create_winogrande_prompt(sentence, option1, option2):
    prompt = f"""
Sentence: {sentence}
Option 1: {option1}
Option 2: {option2}
Please select the correct option (Option 1 or Option 2).

Answer:""".strip()
    return prompt

def extract_winogrande_answer(response):
    response = response.strip().lower()
    if 'option 1' in response:
        return '1'
    elif 'option 2' in response:
        return '2'
    else:
        return None

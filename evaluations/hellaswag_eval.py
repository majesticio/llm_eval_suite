# evaluations/hellaswag_eval.py

from datasets import load_dataset
from tqdm import tqdm
import re

def evaluate_hellaswag(model, sample_size=None):
    """
    Evaluate the model on the HellaSwag dataset.

    Args:
        model: The model object with a `chat` method.
        sample_size: Number of samples to use for evaluation (default is entire dataset).
    """
    hellaswag_dataset = load_dataset("hellaswag", split="validation")
    label_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}

    # If sample_size is provided, select a subset of the dataset
    if sample_size:
        hellaswag_dataset = hellaswag_dataset.select(range(min(sample_size, len(hellaswag_dataset))))

    correct = 0
    total = 0

    for idx, example in enumerate(tqdm(hellaswag_dataset, desc="Evaluating HellaSwag")):
        context = example['ctx']
        endings = example['endings']
        correct_answer = label_map[int(example['label'])]

        prompt = create_hellaswag_prompt(context, endings)

        try:
            response = model.chat(prompt)
            predicted_answer = extract_hellaswag_answer(response, endings)

            if predicted_answer == correct_answer:
                correct += 1
            total += 1

        except Exception as e:
            print(f"Error processing example {idx}: {e}")
            continue

    accuracy = correct / total * 100 if total > 0 else 0
    print(f"HellaSwag Accuracy: {accuracy:.2f}%")
    return accuracy

def create_hellaswag_prompt(context, endings):
    choices_str = '\n'.join([f"{chr(65+i)}. {ending}" for i, ending in enumerate(endings)])
    prompt = f"""
Context:
{context}

Endings:
{choices_str}

Question: Which ending is the most plausible continuation of the context? Please answer with 'A', 'B', 'C', or 'D'.

Answer:""".strip()
    return prompt

def extract_hellaswag_answer(response, endings):
    response = response.strip().upper()
    match = re.search(r'\b([ABCD])\b', response)
    if match:
        return match.group(1)
    else:
        # Attempt to match the full ending text
        for i, ending in enumerate(endings):
            if ending.strip().lower() in response.lower():
                return chr(65 + i)  # Convert index to 'A', 'B', 'C', 'D'
        return None

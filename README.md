# **LLM Evaluation Suite**
----------
*Easily test open source LLMs using Ollama*

This project provides a comprehensive testing suite to evaluate large language models (LLMs) on a wide range of natural language understanding tasks. The suite includes evaluations that cover commonsense reasoning, reading comprehension, textual entailment, and other advanced language tasks.

## **Evaluations Included**

The suite includes the following evaluations:

### **1. BoolQ (Boolean Questions)**
- **Task**: BoolQ is a yes/no question-answering task where the model is given a passage and a question. The model must determine whether the question is true or false based on the passage.
- **Dataset Source**: [SuperGLUE BoolQ](https://super.gluebenchmark.com/tasks)
- **What it Tests**: Reading comprehension and fact-checking abilities.
- **Input**: Passage and question.
- **Output**: Yes/No.

### **2. WinoGrande**
- **Task**: WinoGrande is a commonsense reasoning dataset where the model is given a sentence with an ambiguous pronoun. The model must resolve the pronoun by selecting the most plausible antecedent from two options.
- **Dataset Source**: [WinoGrande](https://leaderboard.allenai.org/winogrande)
- **What it Tests**: Commonsense reasoning and pronoun resolution.
- **Input**: Sentence with two options for an ambiguous pronoun.
- **Output**: Option 1 or Option 2.

### **3. HellaSwag**
- **Task**: HellaSwag presents a model with a context and several plausible endings, requiring the model to select the most likely continuation of the context.
- **Dataset Source**: [HellaSwag](https://rowanzellers.com/hellaswag/)
- **What it Tests**: Commonsense reasoning and the ability to predict the next plausible action or event.
- **Input**: Context and multiple-choice endings.
- **Output**: The most plausible ending (A, B, C, or D).

### **4. RTE (Recognizing Textual Entailment)**
- **Task**: Given a premise and a hypothesis, the model must determine if the hypothesis is entailed, contradicted, or neutral with respect to the premise.
- **Dataset Source**: [SuperGLUE RTE](https://super.gluebenchmark.com/tasks)
- **What it Tests**: Textual entailment and natural language inference.
- **Input**: Premise and hypothesis.
- **Output**: Yes/No for entailment.

### **5. PIQA (Physical Interaction Question Answering)**
- **Task**: PIQA tests a model’s ability to reason about physical interactions. Given a goal and two possible solutions, the model must select the more plausible solution.
- **Dataset Source**: [PIQA](https://yonatanbisk.com/piqa/)
- **What it Tests**: Commonsense knowledge of physical interactions.
- **Input**: Goal and two solutions.
- **Output**: The more plausible solution (Solution 1 or Solution 2).

### **6. CommonSenseQA**
- **Task**: CommonSenseQA is a multiple-choice question-answering task that tests the model’s commonsense reasoning ability.
- **Dataset Source**: [CommonSenseQA](https://www.tau-nlp.org/commonsenseqa)
- **What it Tests**: Commonsense knowledge and reasoning.
- **Input**: Question with multiple-choice answers.
- **Output**: The correct choice (A, B, C, D, or E).

### **7. MultiRC (Multiple Sentence Reading Comprehension)**
- **Task**: MultiRC is a reading comprehension task where the model must read a passage and answer a question by selecting the most appropriate answer from a set of multiple choices.
- **Dataset Source**: [SuperGLUE MultiRC](https://super.gluebenchmark.com/tasks)
- **What it Tests**: Complex reading comprehension.
- **Input**: Passage, question, and multiple-choice answers.
- **Output**: The correct answer.

### **8. ARC (AI2 Reasoning Challenge)**
- **Task**: ARC is a multiple-choice science question-answering task that requires reasoning and scientific knowledge to answer.
- **Dataset Source**: [ARC](https://allenai.org/data/arc)
- **What it Tests**: Reasoning in scientific contexts.
- **Input**: Science question with multiple-choice answers.
- **Output**: The correct choice (A, B, C, D, or E).

### **9. CB (CommitmentBank)**
- **Task**: CB is a textual entailment task that requires the model to decide whether a hypothesis is entailed by, contradicted by, or neutral with respect to a given premise.
- **Dataset Source**: [SuperGLUE CB](https://super.gluebenchmark.com/tasks)
- **What it Tests**: Natural language inference and understanding.
- **Input**: Premise and hypothesis.
- **Output**: Entailed, Contradicted, or Neutral.

---

## **Usage**

### **1. Installation**

Install with `pipenv` and activate the virtual environment
```bash
pipenv install
pipenv shell
```

### **2. Running Evaluations**

To evaluate a model on multiple datasets, you can use the `main.py` script.

#### **Basic Usage**

```bash
python main.py --model <model-name> --evaluations <evaluation1> <evaluation2> --sample-size <number>
```

- **`--model`**: The name of the model to evaluate.
- **`--evaluations`**: The list of evaluations to run. You can include any combination of the following:
  - `boolq`
  - `winogrande`
  - `hellaswag`
  - `rte`
  - `piqa`
  - `commonsenseqa`
  - `multirc`
  - `arc`
  - `cb`
- **`--sample-size`**: (Optional) Limits the number of samples to evaluate from each dataset.

#### **Example Usage**

Evaluate the model `mixtral` on BoolQ, PIQA, and WinoGrande with 50 samples from each:

```bash
python main.py --model mixtral --evaluations boolq piqa winogrande --sample-size 50
```

Run the complete dataset on all evaluations and save the results:

```bash
python main.py --model mixtral > results/results.log
```

This will evaluate the model on the specified datasets and display the accuracy for each.

### **3. Adding More Evaluations**

To add a new evaluation, simply create a new evaluation script in the `evaluations/` folder, following the structure of existing evaluations, and add it to the `evaluation_functions` dictionary in `main.py`.

---

## **Contributing**

We welcome contributions to add new evaluations, improve existing ones, or optimize the framework! Please create a pull request or open an issue to suggest changes.

---

## **License**

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

---

## **Acknowledgements**

We would like to thank the creators of the datasets used in this project, as well as the contributors to the Hugging Face `datasets` library and `ollama` client used for model interaction.

---

This README provides a clear description of the suite’s functionality, the included evaluations, and how to use the testing suite. You can expand on it as needed or customize it to fit your project's style. Let me know if you need any adjustments!
```
testing_suite/
├── evaluations/
│   ├── boolq_eval.py
│   └── hellaswag_eval.py
|   ...
├── models/
│   └── model_loader.py
├── main.py
├── config.py
└── README.md
```

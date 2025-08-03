# Supervised Fine-Tuning of Language Models using Hugging Face & TRL

## Project Summary

This project demonstrates my implementation and understanding of **Supervised Fine-Tuning (SFT)** on causal language models using the Hugging Face Transformers ecosystem and the TRL library. The notebook walks through both training workflows and manual comparision based evaluation, structured in a clear and modular format to highlight essential components of the fine-tuning pipeline.

The focus of this project is not on large-scale deployment but on building **conceptual clarity** around SFT through hands-on experimentation, systematic output logging, and controlled testing on smaller models and datasets.

---

## 📁 Notebook Structure & Key Components

### 1. Environment Setup
Imported all required libraries, including `transformers`, `trl`, and `datasets`, as well as core dependencies like PyTorch and pandas.

### 2. Helper Functions for Inference and Display
I implemented utility functions to:

- Generate responses using a language model, formatted through structured `chat_template` inputs.
- Display training data with role-based message parsing.
- Test the model's performance on a set of predefined questions both before and after fine-tuning.

These functions modularize the notebook and isolate key inference logic for clarity and reusability.

### 3. Baseline Model Evaluation (Before SFT)
A base model (`Qwen3-0.6B-Base`) was loaded and evaluated on a small list of questions. This served as a performance benchmark **prior to fine-tuning**. Responses were printed clearly for comparison.

### 4. Evaluation of Pre-Trained SFT Model
To assess the effect of supervised fine-tuning, I loaded a previously fine-tuned model checkpoint (`Qwen3-0.6B-SFT`) and evaluated it on the same test questions. This allowed for a side-by-side comparison of model behavior before and after SFT.

### 5. Fine-Tuning a Small Model (End-to-End)
Due to resource constraints, I performed actual supervised fine-tuning on a lightweight model (`SmolLM2-135M`) and a reduced dataset.

- The dataset was loaded from Hugging Face (`banghua/DL-SFT-Dataset`), and truncated when running on CPU to reduce runtime.
- Model loading and tokenizer configuration were reused from the earlier setup.

### 6. Training Configuration with `SFTConfig`
A training configuration was defined using `trl.SFTConfig`, specifying key parameters such as:

- Learning rate
- Number of epochs
- Batch size
- Gradient accumulation steps
- Logging frequency

This section demonstrates my understanding of how training arguments influence model updates, memory usage, and training speed.

### 7. Training with `SFTTrainer`
The fine-tuning was executed using the `SFTTrainer` class from the TRL library, passing in the model, configuration, dataset, and tokenizer. The training loop was short and lightweight to ensure it could be completed locally without requiring GPUs.

### 8. Evaluation Post-Fine-Tuning
After training, the fine-tuned small model was evaluated based on manual comparision on the same set of questions. The results helped illustrate the behavioral changes introduced by SFT, even with a small dataset and compact model architecture.

---

## Interpretable Notebook:

This project intentionally avoids fine-tuning on large datasets or models. Instead, it focuses on **scaling down the full fine-tuning pipeline** to a level that can be executed and inspected locally, enabling:

- Close inspection through frequent `print()` statements
- Step-by-step understanding of prompt formatting and generation behavior
- Experimentation with training arguments and dataset sampling

The goal was to achieve **conceptual fluency and technical correctness**. All experiments were designed to be interpretable and reproducible on consumer-grade hardware.

---

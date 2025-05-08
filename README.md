# Chinese-Toxic-Comment-Detection

## Project Overview

This repository presents a comprehensive comparison of traditional machine learning, deep learning, encoder-based, and decoder-based large language models (LLMs) for Chinese toxic comment detection. The models are trained and evaluated using a binary classification task on two major datasets: COLD and TOCAB.

## Project Structure

```
├── data_preprocessing.ipynb         # Script for preprocessing and merging datasets
├── ml_toxic_text_classification.ipynb  # Traditional ML training and evaluation
├── cnn.ipynb                        # CNN deep learning model
├── lstm.ipynb                       # LSTM deep learning model
├── DistilBert.ipynb                # DistilBERT encoder-based model
├── BERT-transformered.ipynb        # BERT-based model
├── llm_finetune_entry.ipynb        # Script to fine-tune LLMs using LLaMA-Factory
├── llm_evaluation_metrics.ipynb    # Script to evaluate LLM-generated predictions
```

## Datasets

* COLD: Chinese Offensive Language Dataset with 37,480 comments (safe/toxic)
* TOCAB: 104,002 labeled abusive comments categorized into six subtypes
* All data is preprocessed into the format with 3 keys:

  * instruction: Prompt for classification
  * input: Text to classify
  * output: Label (safe or toxic)

## Preprocessing Steps

Implemented in `data_preprocessing.ipynb`:

* Format conversion (CSV → JSON)
* Label mapping (multi-class → binary)
* Key renaming: text → input, label → output
* Merging datasets and splitting into train/dev/test

## Models Implemented

### Traditional ML

* Logistic Regression
* Naïve Bayes
* SVM
* Decision Tree

### Deep Learning

* CNN
* LSTM

### Encoder-based Models

* DistilBERT
* BERT-base-uncased
* DeBERTa
* RoBERTa
* ModernBERT

### Decoder-based LLMs (via LoRA/QLoRA)

* Qwen1.5-0.5B / 1.8B / 4B / 7B (8-bit)
* LLaMA2-7B / 13B (8-bit)
* Falcon-7B (8-bit)

## Fine-Tuning

* Framework: LLaMA-Factory https://github.com/hiyouga/LLaMA-Factory
* Low-Rank Adaptation (LoRA) and Quantized LoRA (QLoRA) used
* Fine-tuned using HuggingFace Transformers on Google Colab (NVIDIA L4 GPU)

## Evaluation Metrics

Implemented in `llm_evaluation_metrics.ipynb`:

* Accuracy
* F1 Score
* Precision
* Recall

## Conclusion

Decoder-based LLMs (especially Qwen and LLaMA) show superior performance in detecting Chinese toxic comments, even under quantization. Encoder-based models offer a strong trade-off between performance and efficiency. Traditional ML and deep learning models perform reasonably but lack deep semantic understanding.

---

## How to Run

1. Run `data_preprocessing.ipynb` to generate the training, dev, and test sets
2. Run `ml_toxic_text_classification.ipynb`, `cnn.ipynb`, and `lstm.ipynb` for traditional/deep learning models
3. Run `DistilBert.ipynb` and `BERT-transformered.ipynb` for encoder-based models
4. Run `llm_finetune_entry.ipynb` to finetune LLMs with LLaMA-Factory
5. Run `llm_evaluation_metrics.ipynb` to evaluate LLM-generated predictions

---

## Reference

See the full reference list in the research paper: Chinese Toxic Comment Detection: A Comparative Study of Traditional ML, Deep Learning, Encoder-Based and Decoder-Based Models

---

For any issues, contact the author or create an issue in the repository.

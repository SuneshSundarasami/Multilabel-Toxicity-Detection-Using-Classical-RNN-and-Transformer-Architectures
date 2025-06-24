# 🚦 Multi-Label Toxic Comment Classifier

A modular, extensible **end-to-end NLP pipeline** for detecting and categorizing various types of toxicity in online comments. Built for the [Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge), this project provides robust data preprocessing, diverse model architectures, advanced evaluation, and experiment tracking—all in one place.

---

## 📑 Table of Contents

- [Features](#-features)
- [Project Structure](#-project-structure)
- [Pipeline Flowchart](#-pipeline-flowchart)
- [Dataset](#-dataset)
- [Preprocessing](#-preprocessing)
- [Model Architectures](#-model-architectures)
- [Playground Notebooks](#-playground-notebooks)
- [Installation](#-installation)
- [Usage](#-usage)
- [Results](#-results)
- [References](#-references)
- [Acknowledgements](#-acknowledgements)
- [Contributions](#-contributions)
- [Contact](#-contact)

---

## 🚀 Features

- **Multi-label classification**: toxic, severe toxic, obscene, threat, insult, identity hate  
- **Flexible pipelines** for classical (RNN, vectorizers) & deep learning (BERT, GPT, LoRA, etc.)
- **Comprehensive preprocessing**: advanced cleaning, tokenization, class balancing, feature extraction
- **Class balancing**: oversampling, adaptive focal loss, stratified splitting
- **Experimentation framework**: modular scripts and Jupyter notebooks for ablation, benchmarking, visualization
- **Extendable utilities**: add new models or plug in new data with ease

---

## 📁 Project Structure

The repository is organized for clarity and modularity, allowing easy extension and reproducibility:

- **`Utils/`**:  
  Core Python modules and subfolders for all main tasks:
  - `preprocessing/`: Text cleaning, tokenization, normalization, and feature extraction.
  - `rnn_models/`: Architectures and training scripts for RNN-based models (GRU, LSTM, BiLSTM+Attention).
  - `transformer_models/`: Code for loading, fine-tuning, and evaluating Transformer models (BERT, GPT, etc.).
  - `Vectorizers/`: Classical text representation (TF-IDF, CountVectorizer) and shallow model utilities.
  - `comment_generator/`: Tools for generating synthetic or adversarial comments.
  - `config/`: Configuration files for paths, hyperparameters, and settings.

- **`Playground/`**:  
  Jupyter notebooks for interactive exploration, EDA, model training/ablation, and prototyping.
  - See [Playground directory](https://github.com/SuneshSundarasami/Multi_Label_Toxic_Comment_Classifier/tree/main/Playground) for all files and scripts.

- **`pipeline/`**:  
  Project diagrams, including:
  - `flowchart.drawio`: Editable pipeline flowchart for [draw.io](https://app.diagrams.net/).
  - `flowchart.png`: High-res static image of the pipeline (included below).

- **`NLP Project Poster _ Multi-Class Toxic Comment Classification/`**:  
  Scientific poster, LaTeX sources, and supplementary materials.

---

## 🛠️ Pipeline Flowchart

A visual overview of the full pipeline:

![Pipeline Flowchart](pipeline/flowchart.png)

---

## 📊 Dataset

- **Source:** [Wikipedia Comments Dataset](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data)
- **Description:** 150,000+ comments, each annotated with one or more toxicity labels; highly imbalanced classes.

---

## 🧹 Preprocessing

- **Advanced normalization:**  
  Utilizes Ekphrasis to normalize URLs, emails, numbers, hashtags, allcaps, elongated/repeated/censored words, and more. HTML correction and Twitter-specific segmentation/correction included.

- **Robust cleaning:**  
  Truncates ultra-long comments and overly long words, and collapses extreme character repetitions (e.g., `"soooooooo"` → `"sooo"`). Falls back to simple tokenization for edge cases.

- **Semantic annotation:**  
  Converts Ekphrasis tags to meaningful tokens, e.g.:
  - `<user>` → `PERSON`
  - `<url>` → `WEBSITE`
  - `<allcaps>` → `CAPS`
  - `<elongated>` → `EMPHASIS`
  - `<repeated>` → `INTENSE`
  - `<hashtag>` → `TOPIC`

- **Configurable:**  
  All file paths and data locations are set via a YAML config.

For technical details, see [`preprocessor.py`](https://github.com/SuneshSundarasami/Multi_Label_Toxic_Comment_Classifier/blob/main/Utils/preprocessing/preprocessor.py).

---

## 🤖 Model Architectures

- **Classical models & vectorizers:**  
  TF-IDF, CountVectorizer, Logistic Regression, Random Forest, SVM, batch runners.

- **RNN-based models:**  
  GRU, LSTM, BiLSTM (+attention), GloVe embeddings, adaptive focal loss.

- **Transformer-based & LLMs:**  
  BERT, RoBERTa, GPT-2/3, FLAN-T5, LoRA fine-tuning.

---

## 📒 Playground Notebooks

- **EDA.ipynb**  
  Exploratory Data Analysis: Visualizes class distributions, label imbalance, text length stats, word clouds, and key examples to understand the dataset’s structure and challenge.

- **Preprocessing.ipynb**  
  Step-by-step demonstration of the text preprocessing pipeline, showing cleaning, normalization, tokenization, and annotation on real data samples.

- **Basic_model_trainer.ipynb**  
  Quick-start template for shallow models (Logistic Regression, SVM, Random Forest) using vectorized features; includes metrics and simple cross-validation.

- **Vectorizers_model.ipynb**  
  In-depth experiments with TF-IDF, CountVectorizer, and other classical feature extraction techniques, comparing their impact on different classifiers.

- **RNN_models.ipynb**  
  Trains and evaluates GRU, LSTM, and BiLSTM models (optionally with attention). Includes embeddings setup (e.g., GloVe), class imbalance handling, and analysis of sequence learning.

- **Transformer_based_models.ipynb**  
  Fine-tunes and benchmarks BERT, RoBERTa, and similar transformer models on the toxic comment dataset, tracking validation metrics and class-wise performance.

- **finetuning-gpt-1-full-training.ipynb**  
  Shows end-to-end fine-tuning of a GPT-style language model for multi-label toxicity detection, including data formatting and evaluation.

- **finetuning-flant5-base.ipynb**  
  Demonstrates fine-tuning of FLAN-T5 for the same task, with tips for prompt engineering and sequence-to-sequence approaches.

- **GPT_Transformer_Toxic_Classifier.ipynb**  
  Specialized experiments with GPT-based architectures for toxicity classification, including model adaptation and results.

- **Comment_generator.ipynb**  
  Utility notebook for generating synthetic, adversarial, or “hard” toxic comments to augment the training set.

- **Comment_context_generator.ipynb**  
  Generates contextualized comment examples for advanced augmentation or adversarial testing.

- **comparison.py**  
  Script for aggregating, comparing, and ranking results across all models and approaches; useful for ablation studies and leaderboard creation.

> *For more notebooks, result scripts, and data generators, see the full [Playground directory](https://github.com/SuneshSundarasami/Multi_Label_Toxic_Comment_Classifier/tree/main/Playground).*

---

## ⚡ Installation

```bash
git clone https://github.com/SuneshSundarasami/Multi_Label_Toxic_Comment_Classifier.git
cd Multi_Label_Toxic_Comment_Classifier

# Recommended: Create conda environment from environment.yml
conda env create -f environment.yml
conda activate toxic-comment-classification
```

**For RNN models:**  
Download [glove.840B.300d.zip](https://nlp.stanford.edu/data/glove.840B.300d.zip) (2GB) and place the extracted `.txt` in your `data/` directory.

---

## ▶️ Usage

- **Preprocessing:**  
  Use the script in `Utils/preprocessing/preprocessor.py`:
  ```bash
  python Utils/preprocessing/preprocessor.py
  ```
  Or import functions:
  ```python
  from Utils.preprocessing.preprocessor import preprocess_text, parallel_preprocess
  ```
- **EDA:**  
  `Playground/EDA.ipynb`

- **Classical models:**  
  `Playground/Vectorizers_model.ipynb` or `vectorizer_runner.py`

- **Deep learning:**  
  `Playground/RNN_models.ipynb` or `Transformer_based_models.ipynb`

- **LLM fine-tuning:**  
  `finetuning-gpt-1-full-training.ipynb` or `finetuning-flant5-base.ipynb`

- **Evaluation & comparison:**  
  `comparison.py`

---

## 📈 Results

- **Transformers (BERT, FLAN-T5, GPT)** outperform RNNs and classical models, especially on minority classes.
- **Class balancing + advanced preprocessing = robust detection.**
- See the scientific poster and EDA notebook for detailed results.

---

## 📚 References

- [Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge)
- [Wikipedia Comments Dataset](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data)
- [Scientific Poster (PDF)](NLP%20Project%20Poster%20_%20Multi-Class%20Toxic%20Comment%20Classification/hbrs-poster.pdf)

---

## 🙏 Acknowledgements

Thanks to Prof. Dr. Jörn Hees and Tim Metzler, M.Sc., for their guidance and support.

---

## 👥 Contributions

**Sunesh Praveen Raja Sundarasami**  
- Developed and implemented all classical, RNN-based, and transformer-based models (including BERT, GPT, and more)
- Designed and executed all data preprocessing, EDA, class balancing, and ablation/evaluation studies
- Created all experiment notebooks, scripts, and the scientific poster
- Contributed to project structure, code integration, and reproducibility

**Aaron Cuthinho**  
- Focused on transformer model training with LoRA (Low-Rank Adaptation)
- Co-created the scientific poster

---

## 📬 Contact

For questions, suggestions, or collaboration, open an issue or visit:  
[GitHub: Multi_Label_Toxic_Comment_Classifier](https://github.com/SuneshSundarasami/Multi_Label_Toxic_Comment_Classifier/)

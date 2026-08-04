# AI Sentiment Analysis using Machine Learning & Transformers

A Natural Language Processing (NLP) project that performs sentiment classification using traditional Machine Learning algorithms and modern Transformer-based deep learning models.

The project compares multiple models including Logistic Regression, Naive Bayes, BERT, DistilBERT, and RoBERTa on the SST-2 sentiment classification dataset. A real-time sentiment prediction application is deployed using Streamlit.

## 🚀 Live Demo

Streamlit App:  
(Add your Streamlit URL here)

---

## 📌 Project Overview

Sentiment Analysis is an NLP task that determines the emotional tone of a given text.

This project explores the performance difference between traditional machine learning approaches and transformer-based language models.

The implemented models classify text into:

- Positive Sentiment
- Negative Sentiment

The project includes model training, evaluation, comparison, and deployment as an interactive web application.

---

## 🧠 Models Implemented

### Traditional Machine Learning Models

| Model | Technique |
|---|---|
| Logistic Regression | TF-IDF based text classification |
| Multinomial Naive Bayes | Probabilistic text classification |

### Transformer-Based Models

| Model | Description |
|---|---|
| BERT | Bidirectional Transformer model |
| DistilBERT | Lightweight and faster version of BERT |
| RoBERTa | Optimized BERT architecture with improved training |

---

## 📊 Dataset

Dataset Used:

**Stanford Sentiment Treebank (SST-2)**

- Binary sentiment classification dataset
- Contains movie review sentences
- Labels:
  - Positive
  - Negative

Dataset Source:

https://huggingface.co/datasets/stanfordnlp/sst2

---

## ⚙️ Technologies Used

### Programming Language

- Python

### Machine Learning

- Scikit-learn
- TF-IDF Vectorization
- Logistic Regression
- Naive Bayes

### Deep Learning / NLP

- Hugging Face Transformers
- PyTorch
- BERT
- DistilBERT
- RoBERTa
- NLTK

### Deployment

- Streamlit
- GitHub

---

## 🏗️ Project Structure

```
Sentiment-Analysis-AI/

│
├── app.py
│   └── Streamlit application for real-time prediction
│
├── sentimental_analysis.ipynb
│   └── Model training, evaluation, and comparison
│
├── requirements.txt
│
├── README.md
│
└── model.py
```

---

## 🔄 Application Workflow

```
User Input Text
        |
        ↓
Text Processing
        |
        ↓
Transformer Model
(DistilBERT)
        |
        ↓
Sentiment Prediction
        |
        ↓
Confidence Score
```

---

## 📈 Evaluation Metrics

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score

The project compares:

- Speed and interpretability of traditional ML models
- Accuracy and contextual understanding of Transformer models

---

## 💻 Installation & Usage

### Clone Repository

```bash
git clone https://github.com/Santhosh1108/sentiment-analysis-ai.git
```

### Navigate to Project Folder

```bash
cd sentiment-analysis-ai
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Streamlit Application

```bash
streamlit run app.py
```

---

## 🎯 Key Features

✅ Sentiment classification using Transformer models  
✅ Comparison between traditional ML and Deep Learning approaches  
✅ Real-time prediction interface  
✅ Confidence score generation  
✅ Deployed as an interactive web application  

---

## 📚 Key Learnings

- Built an end-to-end NLP pipeline
- Worked with pretrained Transformer architectures
- Compared classical ML algorithms with modern language models
- Implemented model evaluation techniques
- Converted an ML research notebook into a deployable AI application

---

## 🔮 Future Improvements

- Add multi-class emotion detection
- Add user-uploaded datasets
- Add model selection option in the UI
- Deploy using a production API architecture
- Improve inference speed using optimized models

---

## 👨‍💻 Author

**Santhosh**

GitHub:
https://github.com/Santhosh1108

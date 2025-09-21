# InternIntelligence_NLPTextClassification

This repository contains my second internship project for **InternIntelligence** as part of my Data Science remote internship.

## 🚀 Project: News Article Classification

### 📌 Problem Definition
The goal of this project is to build and evaluate an NLP model that classifies BBC News articles into predefined categories.

### 📊 Dataset
- **Dataset**: BBC News Dataset  
- **Classes**: Business, Entertainment, Politics, Sports, Technology  
- **Features**: News articles (text)  
- **Target**: Category label  

### 🛠️ Steps Followed
1. Preprocessed text data (tokenization, stopword removal, TF-IDF).  
2. Trained traditional ML models (Naive Bayes, Logistic Regression, SVM, Random Forest).  
3. Fine-tuned **DistilBERT** using HuggingFace Transformers (achieved 97% accuracy).  
4. Deployed an interactive **Streamlit app** for real-time classification.  

### ⚙️ Files in this Repo
- `New_Article_Classification.ipynb` → Jupyter notebook with preprocessing, training & evaluation  
- `saved_model/` → Fine-tuned DistilBERT model & tokenizer  
- `app.py` → Streamlit app for predictions  
- `requirements.txt` → Dependencies  

### ▶️ How to Run Locally
1. Clone the repo:
   ```bash
   git clone https://github.com/sana-haroon/InternIntelligence_NLPTextClassification.git
   cd InternIntelligence_NLPTextClassification
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
3. Run the Streamlit app:
  ```bash
  streamlit run app.py
4. Enter a news article or headline and view predictions.

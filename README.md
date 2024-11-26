![logo_ironhack_blue 7](https://user-images.githubusercontent.com/23629340/40541063-a07a0a8a-601a-11e8-91b5-2f13e4e6b441.png)

# PROJECT - NLP Text Classification

### Project Overview:
This project focuses on classifying text data into predefined categories using Natural Language Processing (NLP) techniques and machine learning models. The process involves extensive text preprocessing, feature extraction, and the application of classifiers such as Support Vector Machines (SVM), XGBoost, Logistic Regression, and Multinomial Naive Bayes.

The goal is to create a robust text classification pipeline that can effectively process raw text, and classify it as either human translated text ('0') or machine translated ('1').

---

### Table of Contents
- Folder Structure
- Environment Setup
- Project Components
- Usage
- Future Enhancements

---

### Folder Structure:
- **`main.ipynb`**: Jupyter Notebook containing the end-to-end pipeline for text preprocessing, feature engineering, and classification.
- **`training_data_lowercase.csv`**: Sample training dataset used for model training.
- **`testing_data_lowercase_nolabels.csv`**: input testing dataset used for model evaluation.
- **`G5.csv`**: output dataset classifying which are human translated '0' and which are machine translated text '1'.
- **README.md**: Project documentation (this file).
- **requirements.txt**: Lists all dependencies required for the project.

---

### Project Components:
1. **Text Preprocessing**:
   - Lowercasing and cleaning raw text.
   - Tokenization using NLTK.
   - Stopword removal and lemmatization with POS tagging.

2. **Feature Engineering**:
   - Extracting features using TF-IDF and CountVectorizer.

3. **Model Training and Evaluation**:
   - Models used: 
     - Support Vector Machines (SVM).
     - XGBoost.
     - Logistic Regression.
     - Multinomial Naive Bayes.
   - Evaluation metrics include accuracy, confusion matrix, and classification reports.

4. **Model Tuning**:
   - Hyperparameter tuning using GridSearchCV to optimize performance.

---

### Usage:
Below is the sequence of steps to run the project:
1. **Preprocess the Text Data**:
   - Clean, tokenize, lemmatize, and prepare text data using `preprocess_text` functions.

2. **Feature Extraction**:
   - Generate TF-IDF and CountVectorizer matrices for the training and test sets.

3. **Model Training and Testing**:
   - Train machine learning models on the processed text data.
   - Evaluate models using confusion matrices and classification reports.

4. **Fine-tuning**:
   - Use GridSearchCV to optimize hyperparameters for better performance.

---

### Future Enhancements:
- Explore deep learning-based approaches (e.g., BERT or GPT models) for improved text classification.
- Incorporate additional features such as part-of-speech tags and word embeddings.


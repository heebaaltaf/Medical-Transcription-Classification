# Medical-Transcription-Classification 

This repository contains the code and resources for a medical transcription classification system, which leverages machine learning, deep learning, and transformers to automatically classify medical transcriptions into predefined categories. The system aims to assist healthcare professionals in efficiently organizing and analyzing large volumes of medical transcription data to predict the sub-speciality based on the transcription notes.

Table of Contents

  Introduction
  
  Dataset
  
  Data Cleaning and Preparation
  
  Models and Model Training
  
  Evaluation


## Introduction
Medical transcriptions often contain critical information that needs to be accurately analyzed and the information can be used to categorise the patient to a distinct subspeciality . Manually classifying these transcriptions can be time-consuming and error-prone. This project addresses this challenge by employing various machine learning techniques to automate the classification process.

The classification system employs a combination of traditional machine learning algorithms, deep learning models, and transformer-based models. By using these advanced techniques, the system can capture both low-level and high-level features present in medical transcriptions, leading to improved classification accuracy.

## Dataset:
The dataset used for the project was downloaded from Kaggel-Medical-Transcription-Classification and contains sample medical transcriptions for various medical specialties in csv format.

## Data Preparation:
 Steps for data preparation:
1. Import Libraries-
    Pandas,numpy,sklearn, matplotlib,seaborn,nltk,Spacy,Transformers
2. Load the dataset using Pandas.
3. Drop the irrelevant columns for the project ['Unnamed: 0','description','sample_name','keywords'] and keep the only two columns [transcription and medical speciality] required for our classification tasks.
4. Remove wrongly named subspecialities of medicine from the datasets [" Consult - History and Phy."," Consult - History and Phy.", " SOAP / Chart / Progress Notes"," Discharge Summary"," Emergency Room Reports"," Office Notes"," Letters"," IME-QME-Work Comp etc."]
5. EDA
6. Preprocessing text-Prepare the text.
Make the text lowercase
Remove text in square brackets
Remove punctuation
Remove words containing numbers
Remove the stop words.
Once we have done these cleaning operations we need to perform the following:
Preprocess using sciSpacy to extract the biomedical entities
Lemmatize the texts
7. For Machine Learning algorithms, Tfidf Transformer for tokenization,  Smote for imbalanced data and PCA for feature selection.
8. For Deep Learning Transformers based algorithms- Tokenizers for "BERT","Roberta","XLNet","BioBERT" from  Hugging Face.
9. Another reduced dataset was created after applying domain knowledge to improve the results. As the transcription text for a number of medical speciality is overlapping, we will make following changes:
As Neurology and Neurosurgery although two seperate branches, both deal with Brain and nervous system disorders, we can merge them into one category-"Neurology/Neurosurgery"
Similarly Urology and Nephrology although two seperate branches, both deal with kidney and urinary tract disorders, so we can merge into one category-"Nephrology/Urology"
Also, as Pain mangement and palliative treatment goes together mostly, we can merge them together as "pain and palliative"

10.Drop the categories with less than 50 counts to improve the results.
## Models Evaluated:
FULL Dataset:
1. Logistic Regression with and without PCA
2. Decision Tree with and without PCA
3. Random Forest with and without PCA
4. Naive Bayes.

Reduced Dataset: after applying domain knowledge:
1. Logistic Regression with and without PCA
2. Decision Tree with and without PCA
3. Random Forest with and without PCA
4. Naive Bayes
6. Transformers-Fine Tuning with BERT
7. Transformers-Fine Tuning with RoBERTa
8. Transformers-Fine Tuning with XLNET
9. Transformers-Fine Tuning with BIOBERT
## Evaluation Metrics:
The trained model is evaluated on a separate test set to assess its performance. Metrics like accuracy, precision, recall, and F1 score can be calculated to measure the model's effectiveness and choose the best Model.
![image](https://github.com/heebaaltaf/Medical-Transcription-Classification/assets/68467092/87bf7a37-7ddf-4adc-928b-ebe6f8cc0d6e)


## Inference: 
Once trained and choosing the best model,model can be used to predict the category of new, unseen medical transcriptions.


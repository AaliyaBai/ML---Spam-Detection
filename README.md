# ML-Spam-Detection

## Spam Detection using Naive Bayes Classifier

### Detecting an Email is Spam or Not

#### Project Overview

This project implements a spam detection model using a Naive Bayes classifier. The model classifies email messages as spam or ham (not spam) based on text features extracted using TF-IDF vectorization.

#### Technologies Used

Python

Pandas & NumPy – Data manipulation

Scikit-learn – Machine learning model & feature extraction

#### Dataset

The dataset used is mail_data.csv.

Contains email messages labeled as 'spam' or 'ham'.

#### Steps Involved

#### 1. Data Preprocessing

Load the dataset

Check for missing values and clean the data

Convert categorical labels ('spam', 'ham') to numerical format ('0' for spam, '1' for ham)

#### 2. Feature Engineering

Extract features from text using TF-IDF Vectorization

Use n-grams (1,2) and limit max features for efficiency

#### 3. Model Training

Train a Multinomial Naive Bayes classifier on the transformed text data

Split the dataset into training (75%) and testing (25%)

#### 4. Model Evaluation

Predict outcomes on the test set

Evaluate accuracy and performance using classification report

5. Prediction on New Data

Example email message is transformed using the trained vectorizer

Predicts whether the message is spam or ham

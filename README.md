# Naive Bayes Classifier for Spam Email Detection

This code is a basic implementation of a Naive Bayes classifier for spam email detection. It performs the following steps:

## Importing Libraries

The code imports `pandas` and `numpy` libraries, commonly used for data manipulation and numerical computing in Python.

## Loading the Dataset

Assuming there's a dataset named `emails.csv` containing email data, the code loads it into a Pandas DataFrame named `emails`.

## Checking Dataset Columns

It verifies that the loaded dataset contains required columns ('text' and 'spam'). If any of these columns are missing, it raises a ValueError.

## Defining `process_email` Function

This function processes the text of each email. It converts the text to lowercase and splits it into words, returning a list of unique words in the email.

## Applying `process_email` Function to Create a New Column

The `process_email` function is applied to the 'text' column of the DataFrame to create a new column named 'words'. This column contains a list of unique words in each email.

## Defining `calculate_priors` Function

This function calculates the prior probabilities of spam and ham emails in the dataset. It counts the number of spam and ham emails and divides them by the total number of emails to get the probabilities.

## Calculating Priors

It calculates the prior probabilities of spam and ham emails using the `calculate_priors` function.

## Defining `calculate_likelihoods` Function

This function calculates the likelihoods of each word given spam and ham emails in the dataset. It counts the occurrences of each word in spam and ham emails and divides by the total number of words in each category to get the probabilities.

## Calculating Likelihoods

It calculates the likelihoods of each word given spam and ham emails using the `calculate_likelihoods` function.

## Defining `calculate_posterior` Function

This function calculates the posterior probability of an email being spam given a specific word. It uses Bayes' theorem to combine the prior probabilities, likelihoods, and the probability of the word occurring in spam or ham emails.

## Calculating Posterior

It calculates the posterior probability of an email containing the word 'lottery' being spam using the `calculate_posterior` function.

## Defining `predict_naive_bayes` Function

This function predicts whether a new email is spam or ham using the Naive Bayes algorithm. It calculates the log probabilities of the email being spam or ham based on the presence of words in the email and compares them.

## Predicting Category of a New Email

It predicts the category of a new email ("You have won the lottery!") using the `predict_naive_bayes` function.

This code implements a simple spam filter using the Naive Bayes algorithm, learning from a dataset of labeled emails and predicting the category of new emails based on the presence of certain words.

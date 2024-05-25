import pandas as pd
import numpy as np

# Load the dataset
file_path = './emails.csv'
emails = pd.read_csv(file_path)

print(emails)

# Ensure the dataset has the correct columns
if 'text' not in emails.columns or 'spam' not in emails.columns:
    raise ValueError("The dataset must contain 'text' and 'spam' columns.")

print("hello world")

def process_email(text):
    text = text.lower()
    return list(set(text.split()))

emails["words"] = emails["text"].apply(process_email)
print("hello hello")
print(emails["words"])

def calculate_priors(emails):
    total_emails = len(emails)
    spam_emails = emails[emails["spam"] == 1]
    ham_emails = emails[emails["spam"] == 0]
    
    P_spam = len(spam_emails) / total_emails
    P_ham = len(ham_emails) / total_emails
    
    return P_spam, P_ham

P_spam, P_ham = calculate_priors(emails)

print("P_spam:", P_spam, P_ham)

def calculate_likelihoods(emails):
    spam_words = []
    ham_words = []
    
    for i, row in emails.iterrows():
        if row["spam"] == 1:
            spam_words.extend(row["words"])
        else:
            ham_words.extend(row["words"])
    
    spam_words_count = pd.Series(spam_words).value_counts()
    ham_words_count = pd.Series(ham_words).value_counts()
    
    total_spam_words = len(spam_words)
    total_ham_words = len(ham_words)
    
    spam_likelihoods = spam_words_count / total_spam_words
    ham_likelihoods = ham_words_count / total_ham_words
    
    return spam_likelihoods, ham_likelihoods

spam_likelihoods, ham_likelihoods = calculate_likelihoods(emails)

def calculate_posterior(emails, word):
    P_word_given_spam = spam_likelihoods.get(word, 1e-5)
    P_word_given_ham = ham_likelihoods.get(word, 1e-5)
    
    P_spam_given_word = (P_word_given_spam * P_spam) / ((P_word_given_spam * P_spam) + (P_word_given_ham * P_ham))
    
    return P_spam_given_word

# Example: Calculate the posterior probability of an email containing the word "lottery"
posterior_lottery = calculate_posterior(emails, "lottery")
print(f"The probability of an email being spam given that it contains the word 'lottery' is {posterior_lottery:.4f}")

def predict_naive_bayes(email_text):
    words = process_email(email_text)
    log_prob_spam = np.log(P_spam)
    log_prob_ham = np.log(P_ham)
    
    for word in words:
        if word in spam_likelihoods:
            log_prob_spam += np.log(spam_likelihoods[word])
        else:
            log_prob_spam += np.log(1e-5)
        
        if word in ham_likelihoods:
            log_prob_ham += np.log(ham_likelihoods[word])
        else:
            log_prob_ham += np.log(1e-5)
    
    if log_prob_spam > log_prob_ham:
        return "spam"
    else:
        return "ham"

# Example: Predict the category of a new email
new_email = "You have won the lottery!"
prediction = predict_naive_bayes(new_email)
print(f"The email '{new_email}' is classified as: {prediction}")

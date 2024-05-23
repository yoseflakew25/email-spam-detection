import pandas

# Read the dataset
emails = pandas.read_csv("emails.csv")


def process_email(text):
    text = text.lower()
    return list(set(text.split()))


def predict_naive_bayes(word):
    pass


def calculate_posterior(emails):
    """This function should return the probability if an email contains the
    word 'lottery.'
    """
    pass


emails["words"] = emails["text"].apply(process_email)

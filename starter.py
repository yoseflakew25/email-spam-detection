import pandas

# Read the dataset
emails = pandas.read_csv("emails.csv")


def process_email(text):
    text = text.lower()
    return list(set(text.split()))


emails["words"] = emails["text"].apply(process_email)

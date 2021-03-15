import nltk
from nltk.stem import WordNetLemmatizer


def convert_lowercase(text):
    return text.lower()


def tokenize(text):
    try:
        return nltk.word_tokenize(text)
    except LookupError:
        nltk.download("punkt")
        return nltk.word_tokenize(text)


def remove_token_whitespaces(tokens):
    for i in range(len(tokens)):
        tokens[i] = tokens[i].strip()
    return tokens


def lemmatize(tokens):
    try:
        lemmatizer = WordNetLemmatizer()

        for i in range(len(tokens)):
            tokens[i] = lemmatizer.lemmatize(tokens[i])
        return tokens
    except LookupError:
        nltk.download("wordnet")
        lemmatizer = WordNetLemmatizer()

        for i in range(len(tokens)):
            tokens[i] = lemmatizer.lemmatize(tokens[i])
        return tokens


def preprocess_sst(phrase):
    return " ".join(
        lemmatize(remove_token_whitespaces(tokenize(convert_lowercase(phrase))))
    )

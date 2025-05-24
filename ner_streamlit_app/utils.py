
import os
import re
from sklearn.model_selection import train_test_split
from sklearn_crfsuite import CRF, metrics
import joblib
def tokenize(text):
    return text.split()

def extract_features(tokens):
    return [
        {
            'word': token,
            'is_first': i == 0,
            'is_last': i == len(tokens) - 1,
            'is_capitalized': token[0].isupper(),
            'is_all_caps': token.isupper(),
            'is_all_lower': token.islower(),
            'prefix-1': token[0],
            'prefix-2': token[:2],
            'prefix-3': token[:3],
            'suffix-1': token[-1],
            'suffix-2': token[-2:],
            'suffix-3': token[-3:],
            'prev_word': '' if i == 0 else tokens[i - 1],
            'next_word': '' if i == len(tokens) - 1 else tokens[i + 1],
            'has_hyphen': '-' in token,
            'is_numeric': token.isdigit(),
            'capitals_inside': token[1:].lower() != token[1:]
        }
        for i, token in enumerate(tokens)
    ]

def predict_entities(text, model):
    tokens = tokenize(text)
    features = extract_features(tokens)
    predictions = model.predict([features])[0]
    entities = [(token, label) for token, label in zip(tokens, predictions) if label != 'O']
    return entities

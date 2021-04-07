
import pickle
import numpy as 
from sklearn.ensemble import RandomForestClassifier


def create_dataset(sentences):
    pass


if __name__ == '__main__':

    with open('../sentences_english_with_base_embeddings.pkl', 'rb') as f:
        sentences_english = pickle.load(f)


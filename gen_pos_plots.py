
import numpy as np
import matplotlib.pyplot as plt
import pickle
import argparse

MODEL_TYPE_TO_NUM_LAYERS = {'base': 13}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', default='base', type=str)
    parser.add_argument('--language', default='english', type=str)
    parser.add_argument('--tag', default='UPOS', type=str)
    args = parser.parse_args()

    model_type = args.model_type
    language = args.language
    tag = args.tag


    with open('POS_results/%s_%s_%s_model_accuracy_scores_by_layer.pkl' % (language, tag, model_type), 'rb') as f:
        accuracy_scores = pickle.load(f)

    with open('POS_results/%s_%s_%s_model_f1_micro_scores_by_layer.pkl' % (language, tag, model_type), 'rb') as f:
        f1_micro_scores = pickle.load(f)

    with open('POS_results/%s_%s_%s_model_f1_macro_scores_by_layer.pkl' % (language, tag, model_type), 'rb') as f:
        f1_macro_scores = pickle.load(f)

    with open('POS_results/%s_%s_%s_model_f1_weighted_scores_by_layer.pkl' % (language, tag, model_type), 'rb') as f:
        f1_weighted_scores = pickle.load(f)


    layers = np.arange(MODEL_TYPE_TO_NUM_LAYERS[model_type], dtype=int)
    plt.plot(layers, accuracy_scores, marker='d', label='accuracy')
    plt.plot(layers, f1_micro_scores, marker='d', label='f1_micro')
    plt.plot(layers, f1_macro_scores, marker='d', label='f1_macro')
    plt.plot(layers, f1_weighted_scores, marker='d', label='f1_weighted')
    plt.title('%s - %s - %s model' % (language, tag, model_type))
    plt.xticks(layers)
    plt.legend()
    plt.show()
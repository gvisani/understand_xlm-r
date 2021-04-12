
import numpy as np
import matplotlib.pyplot as plt
import pickle
import argparse

MODEL_TYPE_TO_NUM_LAYERS = {'base': 13}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', default='base', type=str)
    parser.add_argument('--language', default='english,italian,hindi', type=str)
    parser.add_argument('--tag', default='UPOS', type=str)
    args = parser.parse_args()

    model_type = args.model_type
    language = args.language
    tag = args.tag

    languages = language.split(',')

    if len(languages) == 1:

        with open('POS_results/%s_%s_%s_model_f1_micro_scores_by_layer.pkl' % (language, tag, model_type), 'rb') as f:
            f1_micro_scores = pickle.load(f)

        with open('POS_results/%s_%s_%s_model_f1_macro_scores_by_layer.pkl' % (language, tag, model_type), 'rb') as f:
            f1_macro_scores = pickle.load(f)

        layers = np.arange(MODEL_TYPE_TO_NUM_LAYERS[model_type], dtype=int)
        plt.plot(layers, f1_micro_scores, marker='d', label='F1 Micro')
        plt.plot(layers, f1_macro_scores, marker='d', label='F1 Macro')
        plt.grid(ls='--', color='gray', alpha=0.5)
        plt.title('%s - %s - %s model' % (language, tag, model_type))
        plt.xticks(layers)
        plt.legend()
        plt.show()

    else:

        f1_micro_scores_by_language = []
        f1_macro_scores_by_language = []
        for language in languages:
            with open('POS_results/%s_%s_%s_model_f1_micro_scores_by_layer.pkl' % (language, tag, model_type), 'rb') as f:
                f1_micro_scores_by_language.append(pickle.load(f))
            with open('POS_results/%s_%s_%s_model_f1_macro_scores_by_layer.pkl' % (language, tag, model_type), 'rb') as f:
                f1_macro_scores_by_language.append(pickle.load(f))


        layers = np.arange(MODEL_TYPE_TO_NUM_LAYERS[model_type], dtype=int)

        for ll, language in enumerate(languages):
            plt.plot(layers, f1_micro_scores_by_language[ll], marker='d', label=language)
        plt.grid(ls='--', color='gray', alpha=0.5)
        plt.title('F1 Micro - %s - %s model' % (tag, model_type))
        plt.xticks(layers)
        plt.ylim((0.49, 1.01))
        plt.legend()
        plt.show()
        plt.savefig('POS_results/f1_micro-%s-%s_model.pdf' % (tag, model_type))

        for ll, language in enumerate(languages):
            plt.plot(layers, f1_macro_scores_by_language[ll], marker='d', label=language)
        plt.grid(ls='--', color='gray', alpha=0.5)
        plt.title('F1 Macro - %s - %s model' % (tag, model_type))
        plt.xticks(layers)
        plt.ylim((0.49, 1.01))
        plt.legend()
        plt.show()
        plt.savefig('POS_results/f1_macro-%s-%s_model.pdf' % (tag, model_type))

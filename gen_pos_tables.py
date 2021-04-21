
import numpy as np
import matplotlib.pyplot as plt
import pickle
import argparse
from sklearn.metrics import f1_score
import pandas as pd
from scipy.stats import spearmanr

MODEL_TYPE_TO_NUM_LAYERS = {'base': 13}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', default='base', type=str)
    parser.add_argument('--language', default='english,italian,hindi,icelandic', type=str)
    parser.add_argument('--tag', default='UPOS', type=str)
    args = parser.parse_args()

    model_type = args.model_type
    language = args.language
    tag = args.tag

    languages = language.split(',')

    # generate csv files with tag-specific f1 scores by layer, and with tag counts
    # columns are tags, rows are f1 scores, and count is the last row
    for language in languages:

        with open('POS_results/%s_%s_%s_model_ground_truth_scores_by_layer.pkl' % (language, tag, model_type), 'rb') as f:
            all_test_by_layer = pickle.load(f)

        with open('POS_results/%s_%s_%s_model_predictions_scores_by_layer.pkl' % (language, tag, model_type), 'rb') as f:
            all_pred_by_layer = pickle.load(f)

        with open('POS_results/%s_%s_counts.pkl' % (language, tag), 'rb') as f:
            pos_counts = pickle.load(f)

        with open('POS_results/%s_%s_map.pkl' % (language, tag), 'rb') as f:
            pos_map = pickle.load(f)

        scores_table = {'index': ['count']}
        for ttag in pos_counts:
            scores_table[ttag] = [pos_counts[ttag]]

        for ll in range(MODEL_TYPE_TO_NUM_LAYERS[model_type]):
            test_all_tags = all_test_by_layer[ll]
            pred_all_tags = all_pred_by_layer[ll]

            f1_scores_by_tag = f1_score(test_all_tags, pred_all_tags, average=None, zero_division=0)
            
            scores_table['index'].append('Layer %d' % (ll))
            for tt, f1 in enumerate(f1_scores_by_tag):
                ttag = pos_map[tt]
                scores_table[ttag].append(f1)

        scores_table = pd.DataFrame(scores_table)

        correlations = [None]
        p_values = [None]
        for ll in range(MODEL_TYPE_TO_NUM_LAYERS[model_type]):
            corr, pvalue = spearmanr(scores_table.loc[scores_table['index'] == 'count', :].values[0][1:], scores_table.loc[scores_table['index'] == 'Layer %d' % (ll), :].values[0][1:])
            correlations.append(corr)
            p_values.append(pvalue)
        scores_table['Spearman Corr.'] = correlations
        scores_table['p-value'] = p_values

        scores_table.to_csv('POS_results/%s_%s_%s_model_f1_scores_by_tag_by_layer.csv' % (language, tag, model_type), index=False)
# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import argparse
from matplotlib.font_manager import FontProperties

MODEL_TYPE_TO_NUM_LAYERS = {'base': [12,12]} #[num layers, num heads]

def load_weights(sentences, model_type='base'):
    sentence_weights = []
    for sentence in sentences:
        sen_weight = np.zeros((len(sentence)-1, MODEL_TYPE_TO_NUM_LAYERS[model_type][0], 
                               MODEL_TYPE_TO_NUM_LAYERS[model_type][1], len(sentence) - 1)) #[sequence, layers, num heads, sequence]
        for word_i in range(1, len(sentence)):
            for l in range(MODEL_TYPE_TO_NUM_LAYERS[model_type][0]):
                layer_key = 'Attention Layer %d' % (l)
                sen_weight[word_i-1,l,:,:] = sentence[word_i][layer_key]
                
        sentence_weights.append(sen_weight)


    return sentence_weights

    print("here")

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', default='base', type=str)
    parser.add_argument('--language', default='hindi', type=str)
    parser.add_argument('--sentence', default=0, type=int)
    parser.add_argument('--word', default=26, type=int)

    args = parser.parse_args()

    model_type = args.model_type
    language = args.language
    sentence_num = args.sentence
    token_num = args.word

    print('-----------------------------------------------------')
    print('       Loading pickled sentences for %s model' % model_type)
    print('-----------------------------------------------------')
    with open('../sentences_%s_with_%s_embeddings.pkl' % (language, model_type), 'rb') as f:
        sentences = pickle.load(f)
    
    outfile = "weight_heatmaps/heatmap_%s_sentence_%d_word_%d.png" % (language, sentence_num, token_num)
    plttitle = "%s, word = %s" % (sentences[sentence_num][0]['text'], sentences[sentence_num][token_num]['token'])
    print('--------------------------------------------------------------------')
    print('          Creating attention weight info for %s model' % (model_type))
    print('--------------------------------------------------------------------')
    attention = load_weights(sentences, model_type)

    if token_num > len(sentences[sentence_num]) - 1:
        print("word does not exist")
        exit()
        
    print('---------------------------------------')
    print('sentence is = {}'.format(sentences[sentence_num][0]['text']))
    print('---------------------------------------')
    print('word is = {}'.format(sentences[sentence_num][token_num]['token']))
    tok_list = [sentences[sentence_num][idx]['token'] for idx in range(1, len(sentences[sentence_num]))]
    attentions_tok = attention[sentence_num][token_num]
    
    cols = 2
    rows = int(MODEL_TYPE_TO_NUM_LAYERS[model_type][1]/cols)
    
    if language == 'hindi':
        hindi_font = FontProperties(fname = 'Nirmala.ttf')
    fig, axes = plt.subplots( rows,cols, figsize = (20,30))

    axes = axes.flat
    print ("Attention weights for token = {}".format(sentences[sentence_num][token_num]['token']))
    
    for i,att in enumerate(attentions_tok):
    
        #im = axes[i].imshow(att, cmap='gray')
        #if language == 'hindi':
            #sns.heatmap(att,vmin = 0, vmax = 1,ax = axes[i], xticklabels = tok_list, cmap="coolwarm", fontproperties=hindi_font)
        #else:
        sns.heatmap(att,vmin = 0, vmax = 1,ax = axes[i], xticklabels = tok_list, cmap="coolwarm")
        axes[i].set_title(f'head - {i} ' )
        axes[i].set_ylabel('layers')    
        
    plt.suptitle(plttitle)
    #plt.show() 
    plt.savefig(outfile)
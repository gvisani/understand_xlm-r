import pickle
import torch
import transformers
from transformers import BertConfig,BertTokenizer,  BertModel
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
#test

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

    model_type = 'base'

    print('-----------------------------------------------------')
    print('       Loading pickled sentences for %s model' % model_type)
    print('-----------------------------------------------------')
    with open('sentences_english_with_%s_embeddings.pkl' % (model_type), 'rb') as f:
        sentences_english = pickle.load(f)
        
    print('--------------------------------------------------------------------')
    print('          Creating attention weight info for %s model' % (model_type))
    print('--------------------------------------------------------------------')
    attention = load_weights(sentences_english, model_type)

    sentence_num = 0
    token_num = 5
    if token_num > len(sentences_english[sentence_num]) - 1:
        print("word does not exist")
        exit()
        
    print('---------------------------------------')
    print('sentence is = {}'.format(sentences_english[sentence_num][0]['text']))
    print('---------------------------------------')
    print('word is = {}'.format(sentences_english[sentence_num][token_num]['token']))
    tok_list = [sentences_english[sentence_num][idx]['token'] for idx in range(1, len(sentences_english[sentence_num]))]
    attentions_tok = attention[sentence_num][token_num]
    
    cols = 2
    rows = int(MODEL_TYPE_TO_NUM_LAYERS[model_type][1]/cols)
    
    fig, axes = plt.subplots( rows,cols, figsize = (14,30))
    axes = axes.flat
    print ("Attention weights for token = {}".format(sentences_english[sentence_num][token_num]['token']))
    
    for i,att in enumerate(attentions_tok):
    
        #im = axes[i].imshow(att, cmap='gray')
        sns.heatmap(att,vmin = 0, vmax = 1,ax = axes[i], xticklabels = tok_list)
        axes[i].set_title(f'head - {i} ' )
        axes[i].set_ylabel('layers')    
        
    plt.show() 
import pickle
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import argparse
from matplotlib.font_manager import FontProperties
import os

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
    parser.add_argument('--language', default='english_italian_experiment', type=str)
    parser.add_argument('--sentence', default=6, type=int)
    parser.add_argument('--word', default=34, type=int)
    parser.add_argument('--by_head', default=False, type=bool)

    args = parser.parse_args()

    matplotlib.rcParams['font.sans-serif'] = ['Source Han Sans TW',
                                   'DejaVu Serif',
                                   'Arial Unicode MS',  # fc-list :lang=hi family
                                   'Nirmala UI'
                                   ]
    model_type = args.model_type
    language = args.language
    sentence_num = args.sentence
    token_num = args.word
    by_head = args.by_head
    
    print('-----------------------------------------------------')
    print('       Loading pickled sentences for %s model' % model_type)
    print('-----------------------------------------------------')
    with open('../sentences_%s_with_%s_embeddings.pkl' % (language, model_type), 'rb') as f:
        sentences = pickle.load(f)
    
    outfile = "weight_heatmaps/heatmap_%s_sentence_%d_word_%d.png" % (language, sentence_num, token_num)
    if os.path.exists(outfile):
      os.remove(outfile)
    plttitle = "%s, word = %s" % (sentences[sentence_num][0]['text'], sentences[sentence_num][token_num]['token'])
    print('--------------------------------------------------------------------')
    print('          Creating attention weight info for %s model' % (model_type))
    print('--------------------------------------------------------------------')
    attention = load_weights(sentences, model_type) # [sequence, layers, num heads, sequence]
    
        
    if token_num > len(sentences[sentence_num]) - 1:
        print("word does not exist")
        exit()
        
    print('---------------------------------------')
    print('sentence is = {}'.format(sentences[sentence_num][0]['text']))
    print('---------------------------------------')
    print('word is = {}'.format(sentences[sentence_num][token_num]['token']))
    tok_list = [sentences[sentence_num][idx]['token'] for idx in range(1, len(sentences[sentence_num]))]
    attentions_tok = attention[sentence_num][token_num] # [layers, num_heads, sequence]

    cols = 2

    if by_head:
        shape = attentions_tok.shape
        attention_tok = attention.reshape((shape[1], shape[0], shape[2])) # [num heads, layers, sequence]
        ylabel = 'layers'
        title = 'head'
    else:
        ylabel = 'head'
        title = 'layers'
        
    
    rows = int(MODEL_TYPE_TO_NUM_LAYERS[model_type][1]/cols)
    
    if language == 'hindi':
        hindi_font = FontProperties(fname = 'Nirmala.ttf')
        sns.set(font=hindi_font.get_name())
    fig, axes = plt.subplots( rows,cols, figsize = (20,30))

    axes = axes.flat
    print ("Attention weights for token = {}".format(sentences[sentence_num][token_num]['token']))
    
    for i,att in enumerate(attentions_tok):
    
        #im = axes[i].imshow(att, cmap='gray')
        #if language == 'hindi':
            #sns.heatmap(att,vmin = 0, vmax = 1,ax = axes[i], xticklabels = tok_list, cmap="coolwarm", fontproperties=hindi_font)
        #else:
        sns.heatmap(att,vmin = 0, vmax = 1,ax = axes[i], xticklabels = tok_list, cmap="coolwarm")
        axes[i].set_title(f'{title} - {i} ' )
        axes[i].set_ylabel(ylabel)    

    if language == 'hindi':        
        plt.suptitle(plttitle, fontproperties=hindi_font)
    else:
        plt.suptitle(plttitle)
    #plt.show() 
    plt.savefig(outfile)
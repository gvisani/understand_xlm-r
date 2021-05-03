import numpy as np
from tqdm import tqdm
import pickle
import torch
from transformers import XLMRobertaTokenizer, XLMRobertaModel
#test

def test_pretrained():
    tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-large")
    model = XLMRobertaModel.from_pretrained("xlm-roberta-large")

    input_ids = torch.tensor([tokenizer("Hello world ")['input_ids']])

    print(model(input_ids, output_hidden_states=True, output_attentions=True))

def collect_dictionary(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        for ll, line in enumerate(f):
            if ll == 0:
                dictionary = {}
                languages = line.strip().split()
                for language in languages:
                    dictionary[language] = []
            else:
                words = line.strip().split()
                for lan_idx, language in enumerate(languages):
                    dictionary[language].append(words[lan_idx])
    return dictionary

def get_pretokenized_text(sentence):
    pretokenized_text = []
    for i in range(1, len(sentence)):
        pretokenized_text.append(sentence[i]['token'])
    return pretokenized_text

def collapse_weights(attn_weights, tok_map):
    collapsed_weights= np.zeros((attn_weights.shape[0], len(tok_map)))
    for word_i in range(len(tok_map)):
        toks = tok_map[word_i]
        start_cnt = int(np.sum([len(l) for l in tok_map[:word_i]]))
        word_weight = 0.0
        for cnt in range(len(toks)):
            word_weight += attn_weights[:,start_cnt+cnt]
        collapsed_weights[:,word_i] = word_weight/len(toks)
        
    return collapsed_weights

if __name__ == '__main__':
    dictionary = collect_dictionary('english_hindi_dictionary.txt')

    # The tokenizer adds an extra embedding both at the start and at the end of a sentence: the bos and eos embeddings.
    # It looks like XLM-Roberta does not use language embeddings??? It just knows which language it is?
    tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")
    model = XLMRobertaModel.from_pretrained("xlm-roberta-base")

    embeddings = {}

    # try:
    for language in dictionary:
        pre_tok_sen = dictionary[language]
        input_ids = torch.tensor([tokenizer(pre_tok_sen, is_split_into_words=True)['input_ids']])
        tok_map = [tokenizer.encode(x, add_special_tokens=False) for x in pre_tok_sen]
        outputs = model(input_ids, output_hidden_states=True, output_attentions=True)

        embeddings[language] = []

        for word_i, word in enumerate(pre_tok_sen):
            embeddings[language].append({'token': word})

            toks = tok_map[word_i]
            start_cnt = int(np.sum([len(l) for l in tok_map[:word_i]]))+1
            #print("at word {}".format(word_i))
            for emb_i in range(len(outputs.hidden_states)):
                embeddings[language][word_i]['Layer %d' % (emb_i)] = 0.0
                for cnt in range(len(toks)):
                    embeddings[language][word_i]['Layer %d' % (emb_i)] += outputs.hidden_states[emb_i][0, start_cnt+cnt, :].detach().numpy()
                embeddings[language][word_i]['Layer %d' % (emb_i)] /= len(toks)
            #print("Attn for word {}".format(word_i))
            for emb_i in range(len(outputs.attentions)):
                attn_weights = 0.0
                for cnt in range(len(toks)):
                    attn_weights += outputs.attentions[emb_i][0, :, start_cnt+cnt, :].detach().numpy()
                attn_weights /= len(toks)
                attn_weights = collapse_weights(attn_weights, tok_map)
                embeddings[language][word_i]['Attention Layer %d' % (emb_i)] = attn_weights
                    
    # except:
    #     print(language)
    #     print(len(sentence))
    #     print(len(input_ids))
    #     print(input_ids)
    #     print(outputs.hidden_states[emb_i].shape)

    with open('english_hindi_dictionary_with_embeddings.pkl', 'wb') as f:
        pickle.dump(embeddings, f)

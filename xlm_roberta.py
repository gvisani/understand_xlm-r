import numpy as np
from tqdm import tqdm
import pickle
import torch
from transformers import XLMRobertaTokenizer, XLMRobertaModel

def test_pretrained():
    tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-large")
    model = XLMRobertaModel.from_pretrained("xlm-roberta-large")

    input_ids = torch.tensor([tokenizer("Hello world ")['input_ids']])

    print(model(input_ids, output_hidden_states=True, output_attentions=True))


# Each sentence is a list of dictionaries. send_id and text at at index zero. Each token is indexed by the index in the original document.
def collect_PUD(language, shorthand):
    sentences = []
    with open('../ud-treebanks-v2.7/UD_%s-PUD/%s_pud-ud-test.conllu' % (language, shorthand), 'r', encoding='utf-8') as f:
        for ll, line in enumerate(f):
            line = line.strip()
            line_items = line.split()

            if len(line_items) == 0: # separation between sentences
                sentences.append(sentence)
            elif line_items[1] == 'newdoc' or line_items[1] == 'Checktree:' or line_items[1] == 'text_en': # just skip these lines whenever they occur
                pass
            elif line_items[1] == 'sent_id':
                sentence = [{'sent_id': line_items[3]}]
            elif line_items[1] == 'text':
                sentence[0]['text'] = line[9:]
            else: # token case
                try:
                    if '.' in line_items[0]: # ignore extras
                        continue
                    token = {}
                    token['token'] = line_items[1]
                    token['token_processed'] = line_items[2]
                    token['UPOS'] = line_items[3]
                    token['XPOS'] = line_items[4]
                    token['features'] = line_items[5]
                    token['relation'] = line_items[6:]
                    sentence.append(token)
                except:
                    print('Error processing line %d.' % (ll))
                    print('Line reads: %s.' % (line))

    return sentences

def get_pretokenized_text(sentence):
    pretokenized_text = []
    for i in range(1, len(sentence)):
        pretokenized_text.append(sentence[i]['token'])
    return pretokenized_text

if __name__ == '__main__':
    sentences_english = collect_PUD('English', 'en')
    # sentences_italian = collect_PUD('Italian', 'it')
    # sentences_hindi = collect_PUD('Hindi', 'hi')

    # The tokenizer adds an extra embedding both at the start and at the end of a sentence: the bos and eos embeddings.
    # It looks like XLM-Roberta does not use language embeddings??? It just knows which language it is?
    tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")
    model = XLMRobertaModel.from_pretrained("xlm-roberta-base")

    try:
        for sentence in tqdm(sentences_english):
            input_ids = torch.tensor([tokenizer(get_pretokenized_text(sentence), is_split_into_words=True)['input_ids']])
            outputs = model(input_ids, output_hidden_states=True, output_attentions=False)

            for word_i in range(1, len(sentence)):
                for emb_i in range(len(outputs.hidden_states)):
                    sentence[word_i]['Layer %d' % (emb_i)] = outputs.hidden_states[emb_i][0, word_i, :].detach().numpy()
    except:
        print(sentence[0]['text'])
        print(len(sentence))
        print(len(input_ids))
        print(input_ids)
        print(outputs.hidden_states[emb_i].shape)

    with open('sentences_english_with_base_embeddings.pkl', 'wb') as f:
        pickle.dump(sentences_english, f)

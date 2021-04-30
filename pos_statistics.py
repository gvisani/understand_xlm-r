import json
import pickle
from tqdm import tqdm

if __name__ == '__main__':
    
    for language in ['hindi', 'icelandic']:
        print('Processing %s ...' % language)

        with open('../sentences_%s_with_base_embeddings.pkl' % language, 'rb') as f:
            sentences = pickle.load(f)

        upos_counts = {}
        xpos_counts = {}
        for sentence in tqdm(sentences):
            for word_i in range(1, len(sentence)):
                word = sentence[word_i]
                upos = word['UPOS']
                xpos = word['XPOS']
                
                if upos not in upos_counts:
                    upos_counts[upos] = 0
                upos_counts[upos] += 1

                if xpos not in xpos_counts:
                    xpos_counts[xpos] = 0
                xpos_counts[xpos] += 1

        with open('POS_results/%s_UPOS_counts.pkl' % language, 'wb') as f:
            pickle.dump(upos_counts, f)

        with open('POS_results/%s_XPOS_counts.pkl' % language, 'wb') as f:
            pickle.dump(xpos_counts, f)
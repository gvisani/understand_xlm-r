import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from umap import UMAP
import pickle




if __name__ == '__main__':

    with open('english_hindi_dictionary_with_embeddings.pkl', 'rb') as f:
        dictionary = pickle.load(f)

    layer = 'Layer 12'

    word_names = []
    word_embeddings = []
    word_languages = []
    for language in dictionary:
        for word_info in dictionary[language]:
            if word_info['token'] not in word_names:
                word_names.append(word_info['token'])
                word_embeddings.append(word_info[layer])
                word_languages.append(language)

    word_embeddings = np.vstack(word_embeddings)
    
    reducer = TSNE(n_components=2, perplexity=5)
    # reducer = UMAP(n_components=2)

    projections = reducer.fit_transform(word_embeddings)

    def color_fn(language):
        if language == 'english':
            return 'blue'
        elif language == 'italian':
            return 'red'
        elif language == 'hindi':
            return 'green'
        else:
            return 'black'

    colors = list(map(color_fn, word_languages))
    plt.scatter(projections[:, 0], projections[:, 1], c=colors)
    for xy_i, xy in enumerate(projections):
        plt.annotate(word_names[xy_i], xy)
    plt.show()


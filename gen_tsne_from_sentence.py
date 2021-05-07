import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from umap import UMAP
import pickle
from matplotlib.colors import hsv_to_rgb
from cycler import cycler
import scipy
import seaborn as sns
from scipy.spatial.distance import euclidean
import matplotlib as mpl
from matplotlib.font_manager import FontProperties
import argparse

mpl.rcParams['font.sans-serif'] = ['Source Han Sans TW',
                               'DejaVu Serif',
                               'Arial Unicode MS',  # fc-list :lang=hi family
                               'Nirmala UI'
                               ]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--languages', default='english,hindi', type=str)
    parser.add_argument('--index_pairs', default='16-21,19-19,26-31,34-35,6-7,8-17,25-30,21-24')
    args = parser.parse_args()

    # english-italian: '16-20,19-23,26-39,34-31,6-9,8-10,25-40,21-24'
    # english-icelandic: '16-12,26-22,34-28,6-5,8-7,25-21,21-15'

    languages = args.languages.split(',')

    sentences = {}
    for language in languages:
        with open('../sentences_%s_with_base_embeddings.pkl' % (language), 'rb') as f:
            sentences[language] = pickle.load(f)

    indices = {}
    for language in languages:
        indices[language] = []
    index_pairs = args.index_pairs.split(',')
    for pair in index_pairs:
        first, second = list(map(int, pair.split('-')))
        indices[languages[0]].append(first)
        indices[languages[1]].append(second)


    colors = [hsv_to_rgb([(i * 0.618033988749895) % 1.0, 1, 1]) for i in range(100)]
    plt.rc('axes', prop_cycle=(cycler('color', colors)))

    hindi_font = FontProperties(fname = 'Nirmala.ttf')
    sns.set(font=hindi_font.get_name())
    sns.set_style("whitegrid")
    sns.set_context("notebook", font_scale=1.25)

    fig, ax_grid = plt.subplots(nrows=2, ncols=6, figsize=(30, 10))
    ax_grid = ax_grid.flatten()

    for layer_i in range(12):
        layer = 'Layer %d' % (layer_i + 1)

        word_names = []
        word_embeddings = []
        word_languages = []
        for language in languages:
            sentence = sentences[language][0] # we use only the first sentence in our experiments
            for index in indices[language]:
                word_info = sentence[index]
                word_names.append(word_info['token'])
                word_embeddings.append(word_info[layer])
                word_languages.append(language)

        word_embeddings = np.vstack(word_embeddings)

        # compute distances between embeddings
        for i in range(word_embeddings.shape[0] // 2):
            anchor_word = word_embeddings[i]
            translated_word = word_embeddings[i + word_embeddings.shape[0]//2]
            translation_distance = euclidean(anchor_word, translated_word)
            other_distances = []
            for j in range(word_embeddings.shape[0] // 2):
                if j != i: # exclude translated word
                    other_distances.append(euclidean(anchor_word, word_embeddings[j + word_embeddings.shape[0]//2]))
            other_avg_distance = np.mean(other_distances)
            print('%s:\t%.3f\t%.3f' % (word_names[i], translation_distance, other_avg_distance))

        pca = PCA()
        reducer = TSNE(n_components=2, perplexity=5)
        # reducer = UMAP(n_components=2, densmap=True)

        projections = reducer.fit_transform(pca.fit_transform(word_embeddings))

        ax = ax_grid[layer_i]
        ax.set_title(layer)
        for i in range(projections.shape[0] // 2):
            ax.scatter([projections[i][0], projections[i + projections.shape[0]//2][0]], [projections[i][1], projections[i + projections.shape[0]//2][1]])
        for xy_i, xy in enumerate(projections):
            ax.annotate(word_names[xy_i], xy, fontproperties=hindi_font)
    
    plt.savefig('tsne_plots/%s_%s_sentence.png' % (languages[0], languages[1]), bbox_inches='tight', pad_inches=0)


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
    parser.add_argument('--languages', default='english_hindi', type=str)
    args = parser.parse_args()

    with open('%s_dictionary_with_embeddings.pkl' % (args.languages), 'rb') as f:
        dictionary = pickle.load(f)

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
        for language in dictionary:
            for word_info in dictionary[language]:
                if word_info['token'] not in word_names:
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
    
    plt.savefig('tsne_plots/%s_words_together.png' % (args.languages), bbox_inches='tight', pad_inches=0)


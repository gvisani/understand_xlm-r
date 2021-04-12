
import pickle
import numpy as np
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import f1_score, accuracy_score
import matplotlib.pyplot as plt
import argparse

import torch
from torch import nn

MODEL_TYPE_TO_NUM_LAYERS = {'base': 13}

def create_dataset(sentences, model_type='base'):
    X_by_layer = {}
    for l in range(MODEL_TYPE_TO_NUM_LAYERS[model_type]):
        layer_key = 'Layer %d' % (l)
        if layer_key not in X_by_layer:
            X_by_layer[layer_key] = []

        for sentence in sentences:
            X_sentence = []
            for token_i in range(1, len(sentence)):
                X_sentence.append(sentence[token_i][layer_key])

            X_by_layer[layer_key].append(np.vstack(X_sentence))


    y_upos_all = []
    y_xpos_all = []
    y_upos = []
    y_xpos = []
    for sentence in sentences:
        y_upos_sentence = []
        y_xpos_sentence = []
        for token_i in range(1, len(sentence)):
            # UPOS are universal POS tags, whereas XPOS are language-specific
            y_upos_all.append(sentence[token_i]['UPOS'])
            y_xpos_all.append(sentence[token_i]['XPOS'])
            y_upos_sentence.append(sentence[token_i]['UPOS'])
            y_xpos_sentence.append(sentence[token_i]['XPOS'])

        y_upos.append(y_upos_sentence)
        y_xpos.append(y_xpos_sentence)

    one_hot_encoder_upos = OneHotEncoder(sparse = False)
    one_hot_encoder_upos = one_hot_encoder_upos.fit(np.array(y_upos_all).reshape(-1, 1))
    for i, y_upos_sentence in enumerate(y_upos):
        y_upos[i] = one_hot_encoder_upos.transform(np.array(y_upos_sentence).reshape(-1, 1))

    one_hot_encoder_xpos = OneHotEncoder(sparse = False)
    one_hot_encoder_xpos = one_hot_encoder_xpos.fit(np.array(y_xpos_all).reshape(-1, 1))
    for i, y_xpos_sentence in enumerate(y_xpos):
        y_xpos[i] = one_hot_encoder_xpos.transform(np.array(y_xpos_sentence).reshape(-1, 1))
    
    return X_by_layer, y_upos, y_xpos

class dataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.len = X.shape[0]
    def __getitem__(self, index):
        return self.X[index], self.y[index]
    def __len__(self):
        return self.len


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # if batch % 100 == 0:
        #     loss, current = loss.item(), batch * len(X)
        #     print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    test_loss, correct = 0, 0
    model.eval()
    indices_preds = []
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            indices_preds.append(pred.argmax(1).numpy())

    test_loss /= size
    correct /= size

    # print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    return correct, np.hstack(indices_preds)

def one_hot_to_indices(y_NT):
    y_N = []
    for y in y_NT:
        y_N.append(np.asarray(y == 1.0).nonzero()[0][0])
    return np.array(y_N)

def indices_to_one_hot(y_N):
    y_NT = []
    for y in y_N:
        arr = np.zeros(np.max(y_N), dytpe=float)
        arr[y] = 1.0
        y_NT.append(n.copy(arr))
    return np.vstack(y_NT)

if __name__ == '__main__':
    np.random.seed(10) # Seed random number generator to keep data splits fixed across datasets. DO NOT ever change this line.

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', default='base', type=str)
    parser.add_argument('--language', default='english', type=str)
    parser.add_argument('--tag', default='UPOS', type=str)
    args = parser.parse_args()

    model_type = args.model_type
    language = args.language
    tag = args.tag

    print('-----------------------------------------------------')
    print('       Loading pickled sentences for %s model' % model_type)
    print('-----------------------------------------------------')
    with open('../sentences_%s_with_%s_embeddings.pkl' % (language, model_type), 'rb') as f:
        sentences = pickle.load(f)

    print('-----------------------------------------------------')
    print('          Creating dataset for %s model' % (model_type))
    print('-----------------------------------------------------')
    X_by_layer, y_upos, y_xpos = create_dataset(sentences, model_type)

    print('There are %d sentences, %d tokens, %d unique UPOS tags, and %d unique XPOS tags.'
                        % (len(X_by_layer['Layer 0']), np.sum([x.shape[0] for x in X_by_layer['Layer 0']]), y_upos[0].shape[1], y_xpos[0].shape[1]))


    print('The embedding dimensions across layers are as follows:')
    for layer_key in X_by_layer:
        print('%s: %d' % (layer_key, X_by_layer[layer_key][0].shape[1]))


    ## Train classifiers!
    ## S = number of sentences
    ## N = number of training data points (tokens)
    ## M = number of test data points (tokens)
    ## F = number of features
    ## T = number of unique tags
    
    # create indices fo shuffling data
    indices = np.arange(len(X_by_layer['Layer 0']), dtype=np.int32)
    np.random.shuffle(indices)

    accuracy_scores = []
    f1_micro_scores = []
    f1_macro_scores = []
    f1_weighted_scores = []
    for layer_key in X_by_layer:
        print('Using %s ...' % layer_key)
        X_S = np.array(X_by_layer[layer_key], dtype=object)[indices]
        if tag == 'UPOS':
            y_S = np.array(y_upos, dtype=object)[indices]
        elif tag == 'XPOS':
            y_S = np.array(y_xpos, dtype=object)[indices]
        else:
            print('Incorrect tag specified, exiting.')
            exit(1)
        
        y_test_total = []
        y_pred_total = []
        for ff in tqdm(range(10)): # 10-fold CV as recommended by the PUD folks
            X_train_NF = np.vstack(np.append(X_S[: ff*100], X_S[(ff+1)*100 :]))
            X_test_MF = np.vstack(X_S[ff*100 : (ff+1)*100])
            y_train_NT = np.vstack(np.append(y_S[: ff*100], y_S[(ff+1)*100 :]))
            y_test_MT = np.vstack(y_S[ff*100 : (ff+1)*100])

            # for compatibility with CrossEntropyLoss
            y_train_N = one_hot_to_indices(y_train_NT)
            y_test_M = one_hot_to_indices(y_test_MT)

            model = nn.Sequential(
                            nn.Linear(X_train_NF.shape[1], y_train_NT.shape[1])
                            )
            learning_rate = 1e-3
            batch_size = 64
            epochs = 10
            loss_fn = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

            train_dataset = dataset(X_train_NF, y_train_N)
            test_dataset = dataset(X_test_MF, y_test_M)

            train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size)
            test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size)

            for t in range(epochs):
                # print(f"Epoch {t+1}\n-------------------------------")
                train_loop(train_dataloader, model, loss_fn, optimizer)
            
            _, y_pred_M = test_loop(test_dataloader, model, loss_fn)

            # model = LogisticRegression(C=1.0, multi_class='multinomial', max_iter=100)
            # model.fit(X_train_NF, y_train_N)
            # y_pred_M = model.predict(X_test_MF)

            y_test_total.append(y_test_M)
            y_pred_total.append(y_pred_M)
        
        y_test_total = np.hstack(y_test_total)
        y_pred_total = np.hstack(y_pred_total)

        accuracy = accuracy_score(y_test_total, y_pred_total)
        f1_micro = f1_score(y_test_total, y_pred_total, average='micro', zero_division=0)
        f1_macro = f1_score(y_test_total, y_pred_total, average='macro', zero_division=0)
        f1_weighted = f1_score(y_test_total, y_pred_total, average='weighted', zero_division=0)

        accuracy_scores.append(accuracy)
        f1_micro_scores.append(f1_micro)
        f1_macro_scores.append(f1_macro)
        f1_weighted_scores.append(f1_weighted)

    with open('POS_results/%s_%s_%s_model_accuracy_scores_by_layer.pkl' % (language, tag, model_type), 'wb') as f:
        pickle.dump(accuracy_scores, f)

    with open('POS_results/%s_%s_%s_model_f1_micro_scores_by_layer.pkl' % (language, tag, model_type), 'wb') as f:
        pickle.dump(f1_micro_scores, f)

    with open('POS_results/%s_%s_%s_model_f1_macro_scores_by_layer.pkl' % (language, tag, model_type), 'wb') as f:
        pickle.dump(f1_macro_scores, f)

    with open('POS_results/%s_%s_%s_model_f1_weighted_scores_by_layer.pkl' % (language, tag, model_type), 'wb') as f:
        pickle.dump(f1_weighted_scores, f)

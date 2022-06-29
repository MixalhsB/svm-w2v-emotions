from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold
from sklearn.svm import SVC
from os import path
import pandas as pd
import numpy as np


def load_emolex():
    lexicon = {}
    counts = {}
    with open('data/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt') as f:
        for line in f:
            word, emotion, present = line.rstrip('\n').split('\t')
            if emotion in ('positive', 'negative'):
                continue
            if word not in lexicon:
                lexicon[word] = {emotion: bool(int(present))}
            else:
                lexicon[word][emotion] = bool(int(present))
            if emotion not in counts:
                counts[emotion] = {bool(int(present)): 1}
            elif bool(int(present)) not in counts[emotion]:
                counts[emotion][bool(int(present))] = 1
            else:
                counts[emotion][bool(int(present))] += 1
    print('\nCOUNTS:')
    for emotion in counts:
        print('\nEmotion "' + emotion + '"')
        print('\tTrue:\t' + str(counts[emotion][True]))
        print('\tFalse:\t' + str(counts[emotion][False]))
    print()
    return (lexicon, counts)


def load_enwiki_words(filename='data/enwiki_20180420_300d.txt'):
    enwiki_words = {}
    with open(filename, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            word, vector_string = line.split(' ', 1)
            enwiki_words[word] = i
    return enwiki_words


def match_emolex_enwiki(lexicon, enwiki_words):
    matched_enwiki_words = {w: enwiki_words[w] for w in lexicon if w in enwiki_words}
    return matched_enwiki_words


def get_matched_enwiki_vectors(matched_enwiki_words, filename='data/enwiki_20180420_300d.txt'):
    matched_enwiki_vectors = {}
    with open(filename, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            word, vector_string = line.split(' ', 1)
            try:
                assert matched_enwiki_words[word] == i
            except (KeyError, AssertionError):
                continue
            matched_enwiki_vectors[word] = vector_string
    return matched_enwiki_vectors


def export_vectors(vectors, out_filename):
    with open(out_filename, 'w', encoding='utf-8') as f:
        for word in vectors:
            f.write(word + ' ' + vectors[word])
    return None


def properly_load_vector_data(filename):
    vector_data_dict = {}
    with open(filename) as f:
        for line in f:
            items = line.rstrip().split()
            row = items[0]
            vec = np.array([float(i) for i in items[1:]])
            vector_data_dict[row] = vec
    return pd.DataFrame(vector_data_dict).T


def add_emolex_columns(data, lexicon, counts):
    for emotion in counts:
        labels = []
        for word in data.index:
            labels.append(lexicon[word][emotion])
        data[emotion] = labels
    return None


def calculate_dummy_baseline(counts):
    print('\nBASELINE:')
    for emotion in counts:
        print('\nEmotion "' + emotion + '"')
        assert counts[emotion][True] < counts[emotion][False]
        precision_False = counts[emotion][False] / (counts[emotion][True] + counts[emotion][False])
        f1_micro_True = 0
        f1_micro_False = 2 * (precision_False * 1) / (precision_False + 1)
        f1_macro = (f1_micro_True + f1_micro_False) / 2
        print('\tMacro-F1:', format(f1_macro, '.3f'))
    print()
    return None
        

def run_SVM_classification(data, counts):
    print('\nSVM:')
    for emotion in counts:
        print('\nEmotion "' + emotion + '"')
        X, y = data.loc[:, 0:299], np.ravel(data[[emotion]])
        model = SVC(gamma='scale', kernel='rbf', class_weight='balanced')
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=1)
        scores = cross_val_score(model, X, y, scoring='f1_macro', cv=cv, n_jobs=-1)
        print('\tMacro-F1:', [format(f1_macro, '.3f') for f1_macro in scores])
        print('\tAvg. Macro-F1:', format(np.mean(scores), '.3f'))
    print()
    return None


def main():
    lexicon, counts = load_emolex()
    if not path.exists('data/matched_enwiki_vectors.txt'):
        enwiki_words = load_enwiki_words()
        matched_enwiki_words = match_emolex_enwiki(lexicon, enwiki_words)
        matched_enwiki_vectors = get_matched_enwiki_vectors(matched_enwiki_words)
        export_vectors(matched_enwiki_vectors, 'data/matched_enwiki_vectors.txt')
    data = properly_load_vector_data('data/matched_enwiki_vectors.txt')
    add_emolex_columns(data, lexicon, counts)
    calculate_dummy_baseline(counts)
    run_SVM_classification(data, counts)
    return data


if __name__ == '__main__':
    data = main()

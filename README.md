# SVM on Word2Vec embeddings for imbalanced binary classification of emotions in words

## Data

In order to run the analysis, please get the files
- _NRC-Emotion-Lexcion-Wordlevel-v0.92.txt_ from http://saifmohammad.com/WebPages/lexicons.html
- _enwiki_20180420_300d.txt_ from https://wikipedia2vec.github.io/wikipedia2vec/pretrained/
and put them into the _./data_ directory.

Note: It is possible to run the script without _enwiki_20180420_300d.txt_, provided that _matched_enwiki_vectors.txt_ is already present in the _./data_ path.

## Requirements

Make sure to install the necessary dependencies:

```
% python3 -m pip install -r requirements.txt  
```

## Run the script

Simply execute this command:

```
% python3 run.py
```

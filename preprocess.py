import torch
from torch.utils.data import TensorDataset
from collections import Counter


def load_text():
    with open("reviews.txt", "r") as f:
        data =f.read()
    return data.split()
    

def get_vocab(text, min_occurences):
    vocab = Counter()
    for word in text:
        vocab[word] += 1

    vocab_top = Counter({k: c for k, c in vocab.items() if c >= min_occurences})
    vocab_tuples = vocab_top.most_common(len(vocab_top))

    word_to_id = Counter({word: i+1 for i,(word, c) in enumerate(vocab_tuples)})
    id_to_word = ["_"] + [word for word, index in word_to_id.items()]

    return word_to_id, id_to_word


def get_tensor_dataset(list_of_ids, n):
    features = []
    labels = []
    for i in range(n, len(list_of_ids)):
        labels.append(list_of_ids[i])
        features.append(list_of_ids[i-n:i])

    return TensorDataset(torch.tensor(features), torch.tensor(labels))
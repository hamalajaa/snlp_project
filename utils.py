import numpy as np
import os
import torch

from data_helper import SentenceMapper
from itertools import chain

data_file = "testdata.txt"
ngram_size = 6


def process_lines(lines):
    """
    Preprocesses lines of text by removing too short and too long sentences.
    """
    l = []
    for line in lines:
        if 3 < len(line.split()) < 50:
            l.append(line)
    return l


def read_file(filename):
    """
    Reads a file and preprocesses it by removing too short and too long sentences.
    """
    with open(filename, 'r', encoding="utf-8") as file:
        lines = list(map(lambda line: line.strip(), file.readlines()))
    return process_lines(lines)


class ReadLines(torch.utils.data.Dataset):
    def __init__(self, filename):
        self.lines = read_file(filename)

    def __getitem__(self, idx):
        return self.lines[idx]

    def __len__(self):
        return len(self.lines)


def create_unique_words(lines):
    """
    Input:
        lines                   Lines from corpus, one sentence per line
    Output:
        unique_words            all unique words in alphabetic order
        nof_unique_words        length of unique words
        max_sentence_length     length of the longest sentence
    """
    np_lines = np.array(lines)
    split = np.char.split(np_lines).tolist()

    all_words = list(chain.from_iterable(split))

    unique_words = np.unique(all_words).tolist()

    """
    Add the stop icon to unique words.
    This will allow our RNN to learn stopping of sentences.
    In addition, it will allow us to have a fixed size
    for our sentence tensors.
    """
    unique_words.append('</s>')
    nof_unique_words = len(unique_words)

    # Calculate the maximum sentence length used in fixing tensor size
    max_sentence_length = np.max([len(sentence.split()) for sentence in np_lines])

    return unique_words, nof_unique_words, max_sentence_length


def build_index(unique_words):
    """
    Builds mapping from word to index and from index to word.
    Mapping can be used to map model outputs to words, as well as,
    to map input words to numerical format.
    """
    word_to_idx = {}
    idx_to_word = {}
    for i, word in enumerate(unique_words):
        word_to_idx[word] = i
        idx_to_word[i] = word

    return word_to_idx, idx_to_word


def inputs_and_targets_from_sequences(tensor):
    """
    An input tensor of shape [d x n] is expected where:
          d = number of sentences
          n = length of the longest sentence
    """
    inputs = tensor[:, :-1]
    targets = tensor[:, 1:]

    return inputs, targets


"""
File saving and loading utility functions:
"""

def createPath(filePath):
    dir = os.path.dirname(filePath)
    # create directory if it does not exist
    if not os.path.exists(dir):
        os.makedirs(dir)


def perplexity_save_path(data_file_size, lstm_h_dim, embedding_dim):
    path = "./results/" + str(data_file_size / 1000) + "k_" + str(lstm_h_dim) + "_" + str(
        embedding_dim) + "/perplexity.csv"
    createPath(path)

    return path


def perplexity_test_save_path(data_file_size, lstm_h_dim, embedding_dim):
    path = "./results/" + str(data_file_size / 1000) + "k_" + str(lstm_h_dim) + "_" + str(
        embedding_dim) + "/perplexity.data"
    createPath(path)

    return path


def model_save_path(data_file_size, lstm_h_dim, embedding_dim):
    path = "./results/" + str(data_file_size / 1000) + "k_" + str(lstm_h_dim) + "_" + str(embedding_dim) + "/model.pth"
    createPath(path)

    return path


def vocab_info_save_path(data_file_size, lstm_h_dim, embedding_dim):
    path = "./results/" + str(data_file_size / 1000) + "k_" + str(lstm_h_dim) + "_" + str(embedding_dim) + "/vocab.json"
    createPath(path)

    return path


def embedding_model_save_path(data_file_size, lstm_h_dim, embedding_dim):
    path = "./results/" + str(data_file_size / 1000) + "k_" + str(lstm_h_dim) + "_" + str(
        embedding_dim) + "/embedding.bin"
    createPath(path)

    return path
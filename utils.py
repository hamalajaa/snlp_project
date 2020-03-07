import numpy as np
from data_helper import SentenceMapper
data_file = "testdata.txt"
ngram_size = 6


def read_file(filename):
    with open(filename, 'r') as file:
        lines = list(map(lambda line: line.strip(), file.readlines()))
    return lines


def create_unique_words(lines):
    """
    Input:
        lines                   Lines from corpus, one sentence per line
    Output:
        unique_words            all unique words in alphabetic order
        nof_unique_words        length of unique words
        max_sentence_length     length of the longest sentence
    """
    unique_words = []

    # Collect unique words from the corpus
    for sentence in lines:
        for word in sentence.split():
            if word not in unique_words:
                unique_words.append(word)

        unique_words = sorted(unique_words)

    """
    Add the stop icon to unique words.
    This will allow our RNN to learn stopping of sentences.
    In addition, it will allow us to have a fixed size
    for our sentence tensors.
    """
    unique_words.append('</s>')
    nof_unique_words = len(unique_words)

    # Calculate the maximum sentence length used in fixing tensor size
    np_lines = np.array(lines)
    max_sentence_length = np.max([len(sentence.split()) for sentence in np_lines])

    return unique_words, nof_unique_words, max_sentence_length


def build_index(unique_words):
    word_to_idx = {}
    idx_to_word = {}
    for i, word in enumerate(unique_words):
        word_to_idx[word] = i
        idx_to_word[i] = word

    return word_to_idx, idx_to_word


def inputs_and_targets_from_sequences(tensor):
        
        # An input tensor of shape [d x n x v] is expected where:
        #       d = number of sentences
        #       n = length of the longest sentence
        #       v = vocabulary size

        inputs = tensor[:, :-1, :].float()
        targets = tensor[:, 1:, :].float()
            
        return inputs, targets
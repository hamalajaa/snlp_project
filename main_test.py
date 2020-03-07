import numpy as np
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
    d = {}
    for i, word in enumerate(unique_words):
        d[word] = i

    return d

lines = read_file(data_file)
unique_words, nof_unique_words, n = create_unique_words(lines)


print(unique_words, "\n", nof_unique_words, "\n", n)

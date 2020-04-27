import torch
import numpy as np

from torch.utils.data import Dataset

class SentenceMapper:
    """
    Maps a list of sentences to tensor format.
    """

    def __init__(self, sentences, word_to_idx, idx_to_word, N):
        self.N = N
        self.sentences = sentences
        self.word_to_idx = word_to_idx
        self.idx_to_word = idx_to_word
        self.V = len(word_to_idx.keys())
        self.D = len(sentences)

    def map_words_to_indices(self, batch, padding_character="</s>"):
        sentences = batch

        # tensor size parameters
        D = len(sentences)
        N = self.N - 1

        tensor = torch.zeros([D, N], dtype=torch.int64)

        # translates words to their word indices
        for s_idx, sentence in enumerate(sentences):
            for w_idx, word in enumerate(sentence):
                # w_id is the index of the current word in the corpus word_to_idx
                w_id = self.word_to_idx[word]

                # set 1 for
                tensor[s_idx, w_idx] = w_id

        return tensor

    def pad_sentences(self, batch, padding_character="</s>"):
        sentences = batch

        # tensor size parameters
        D = len(sentences)
        N = self.N

        tensor = np.zeros((D, N), dtype=object)

        # pads sentences
        for s_idx, sentence in enumerate(sentences):
            split_sentence = sentence.split()

            # number of words in this sentence
            nof_words = len(split_sentence)
            for w_idx, word in enumerate(split_sentence):
                tensor[s_idx, w_idx] = word

            # If the sentence length is smaller
            # than the maximal sentence length,
            # pad rest of the tensor with padding_character
            if nof_words < N:

                for padding_idx in range(nof_words, N):
                    tensor[s_idx, padding_idx] = padding_character

        return tensor

    def map_sentences_to_padded_embedding(self, batch, embedding, embedding_size):
        sentences = batch

        # tensor size parameters
        D = len(sentences)
        N = self.N - 1

        tensor = torch.zeros([D, N, embedding_size], dtype=torch.float32)

        # translates words to their embedding
        for s_idx, sentence in enumerate(sentences):

            for w_idx, word in enumerate(sentence):
                tensor[s_idx, w_idx, :] = torch.tensor(embedding[word])

        return tensor

from torch.utils.data import Dataset
import torch
class WP2008Dataset(Dataset):
    """
    A custom Dataset for a list of sentences
    in tensor format.
    """
    def __init__(self, tensor):
        self.tensor = tensor
        pass

    def __getitem__(self, index):
        return self.tensor[index]

    def __len__(self):
        return self.tensors.size(0)

class SentenceMapper():
    """
    Maps a list of sentences to tensor format.
    """
    def __init__(self, sentences, index, N):
        self.N = N
        self.sentences = sentences
        self.index = index
        self.V = len(index.keys())
        self.D = len(sentences)

    def map_sentences_to_tensors(self, padding_character="</s>"):
        sentences = self.sentences
        index = self.index

        # tensor size parameters
        D = self.D
        N = self.N
        V = self.V
        
        # Final tensor will be of size DxNxV where:
        #       D = Data size, number of sentences
        #       N = length of the longest sentence
        #       V = vocabulary size
        #
        # Example:
        #       D = 4
        #       N = 2
        #       V = 3
        #       [[[1,0,0],[0,1,0]],
        #        [[0,1,0],[0,1,0]],
        #        [[0,1,0],[0,0,1]],
        #        [[1,0,0],[0,0,1]]]
        
        tensor = torch.zeros([D, N, V], dtype=torch.int32)

        # performs one-hot-encoding to sentence data
        for s_idx, sentence in enumerate(sentences):
            split_sentence = sentence.split()

            # number of words in this sentence
            nof_words = len(split_sentence)
            for w_idx, word in enumerate(split_sentence):

                # w_id is the index of the current word in the corpus index
                w_id = index[word]

                # set 1 for 
                tensor[s_idx, w_idx, w_id] = 1

            # If the sentence length is smaller
            # than the maximal sentence length,
            # pad rest of the tensor with padding_character
            if nof_words < N:

                # index of the stop character in the corpus index
                stop_id = index[padding_character]
                for padding_idx in range(nof_words, N):
                    tensor[s_idx, padding_idx, stop_id] = 1

        print(tensor.size())
        return tensor
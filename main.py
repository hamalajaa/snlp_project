import torch
import torch.nn as nn
import torch.nn.functional as F
import fasttext
import json
import pickle
import numpy as np
import argparse
import pandas as pd
import utils
import time

from utils import collate_fn
from utils import createPath
from utils import perplexity_save_path
from utils import perplexity_test_save_path
from utils import model_save_path
from utils import vocab_info_save_path
from utils import embedding_model_save_path
from data_helper import SentenceMapper
from models import LSTM

data_file_size = 20000
data_file = "testdata_20000.txt"

# Path from which the model is loaded
model_load_path = "./results/16.253k_600_500/model.pth"

# Path from which training vocabulary information is loaded.
# We will use this in the prediction phase.
vocab_info_load_path = "./results/16.253k_600_500/vocab.json"

# Device check, use GPU if possible
cuda = torch.cuda.is_available()
if cuda:
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")


def main(load=False):
    # Init hps
    hps = init_hps()

    criterion = nn.CrossEntropyLoss()

    torch.manual_seed(0)

    # Read file
    if load:
        print("Loading file", data_file, "for testing")
    else:
        print("Using file", data_file, "for training")

    lines = utils.read_file(data_file)

    global data_file_size
    data_file_size = len(lines)

    start = time.time()
    unique_words, vocab_size, n = utils.create_unique_words(lines)

    print("vocab_size", vocab_size)
    print("Constructing unique words took:", (time.time() - start))

    # Construct dataloader
    dataset = utils.ReadLines(data_file)
    
    print("data set length:", len(dataset))

    train_set_len = int(len(dataset) * 0.6)
    test_set_len = int(len(dataset) * 0.2)
    validation_set_len = int(len(dataset) * 0.2)
    while train_set_len + test_set_len + validation_set_len != len(dataset):
        validation_set_len += 1

    train_set, test_set, validation_set = torch.utils.data.random_split(dataset, [train_set_len, test_set_len,
                                                                                  validation_set_len])

    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=hps.batch_size, num_workers=8, shuffle=True, collate_fn=collate_fn)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=hps.batch_size, num_workers=8, shuffle=True, collate_fn=collate_fn)
    validation_loader = torch.utils.data.DataLoader(dataset=validation_set, batch_size=hps.batch_size, num_workers=8, shuffle=True, collate_fn=collate_fn)

    # Init model
    if not load:

        word_to_idx, idx_to_word = utils.build_index(unique_words)
        mapper = SentenceMapper(lines, word_to_idx, idx_to_word)

        vocab_info = {'idx_to_word': idx_to_word, 'word_to_idx': word_to_idx, 'vocab_size': vocab_size}

        with open(vocab_info_save_path(data_file_size, hps.lstm_h_dim, hps.embedding_dim), 'wb') as f:
            pickle.dump(vocab_info, f, protocol=pickle.HIGHEST_PROTOCOL)

        embedding = fasttext.train_unsupervised(data_file, model='cbow', dim=hps.embedding_dim)
        embedding.save_model(embedding_model_save_path(data_file_size, hps.lstm_h_dim, hps.embedding_dim))

        print("Training...")
        model = LSTM(hps, vocab_size)
        train_model(hps, idx_to_word, model, train_loader, validation_loader, mapper, embedding)
    else:

        with open(vocab_info_load_path, 'rb') as f:
            vocab_info = pickle.load(f, encoding='utf-8')

        idx_to_word = vocab_info['idx_to_word']
        word_to_idx = vocab_info['word_to_idx']
        vocab_size = vocab_info['vocab_size']

        mapper = SentenceMapper(lines, word_to_idx, idx_to_word)

        embedding = fasttext.load_model(embedding_model_save_path(data_file_size, hps.lstm_h_dim, hps.embedding_dim))

        print("Loading model...")
        model = LSTM(hps, vocab_size)
        model = nn.DataParallel(model).to(device)

        model.load_state_dict(
            torch.load(model_load_path, map_location=device))
        model.to(device)
        model.eval()

        counter = 0

        perplexities = []

        for _, (data, N) in enumerate(test_loader):
            
            padded_data = mapper.pad_sentences(data, N)

            og_inputs, targets = utils.inputs_and_targets_from_sequences(padded_data)
            inputs = mapper.map_sentences_to_padded_embedding(og_inputs, embedding=embedding,
                                                              embedding_size=hps.embedding_dim, N=N)
            targets = mapper.map_words_to_indices(targets, N=N)

            if cuda:
                inputs = inputs.cuda()
                targets = targets.cuda()

            outputs = model(inputs)

            loss = criterion(outputs.permute(0, 2, 1), targets)

            perplexities.append(np.exp(loss.detach().cpu().numpy()))

            topk = F.softmax(outputs, dim=2)[0, :, :]

            topk = torch.topk(topk, 1, dim=1)[1].squeeze(1)

            # print(topk.shape)

            outputs = F.softmax(outputs, dim=2)[0, :, :].detach().cpu().numpy()

            outs = []
            idxs = np.array(list(range(vocab_size)))

            for i in range(outputs.shape[0]):
                outs.append(np.random.choice(idxs, p=np.array(outputs[i, :])))
            output = torch.tensor(outs)

            input_sequence = og_inputs[0, :]
            predicted_sequence = [idx_to_word[c] for c in topk.detach().cpu().numpy()]
            sampled_sequence = [idx_to_word[c] for c in output.detach().cpu().numpy()]

            print('\nInput sequence')
            print(input_sequence)

            print('\nPredicted sequence:')
            print(predicted_sequence)

            print('\nSampled sequence:')
            print(sampled_sequence)

            prev_word = ""
            for i in range(1, len(predicted_sequence)):
                words = input_sequence[:i]
                predicted_next_word = predicted_sequence[i - 1]
                sampled_next_word = sampled_sequence[i - 1]

                if sampled_next_word == '</s>' and (prev_word == '</s>' or input_sequence[i] == '</s>'):
                    break

                prev_word = sampled_next_word

                print(" ".join(list(words)), "[" + predicted_next_word + "|" + sampled_next_word + "]")

            print("Moving on to next prediction....\n")

        print(perplexities)
        mean_perplexity = np.mean(perplexities)

        print(f'Perplexity: {mean_perplexity}')
        with open(perplexity_test_save_path(data_file_size, hps.lstm_h_dim, hps.embedding_dim), 'a') as f:
            f.write(str(mean_perplexity) + "\n")

    return vocab_size, hps


def train_model(hps, idx_to_word, model, train_loader, validation_loader, mapper, embedding):
    # Hyper-parameters
    num_epochs = hps.n_epochs

    if cuda:
        model = nn.DataParallel(model).cuda()

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()

    if cuda:
        criterion = criterion.cuda()

    print(device, cuda)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0002, betas=(0.5, 0.5))

    # Track loss
    training_loss, validation_loss = [], []

    P = pd.DataFrame(
        {'epoch': list(range(num_epochs)), 'model': model_save_path(data_file_size, hps.lstm_h_dim, hps.embedding_dim),
         'perplexity': [0] * num_epochs}
    )

    for i in range(num_epochs):

        epoch_start = time.time()

        # Track loss
        epoch_training_loss = 0
        epoch_validation_loss = 0

        # Training

        model.train()

        perplexities = []

        for _, (data, N) in enumerate(train_loader):
            padded_data = mapper.pad_sentences(data, N=N)
            inputs, targets = utils.inputs_and_targets_from_sequences(padded_data)
            inputs = mapper.map_sentences_to_padded_embedding(inputs, embedding=embedding,
                                                              embedding_size=hps.embedding_dim, N=N)
            targets = mapper.map_words_to_indices(targets, N=N)

            if cuda:
                inputs = inputs.cuda()
                targets = targets.cuda()

            # Forward pass

            outputs = model(inputs).permute(0, 2, 1)

            # Loss
            loss = criterion(outputs, targets)

            perplexity = torch.exp(loss)
            perplexities.append(perplexity.item())

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update loss
            epoch_training_loss += loss.detach().cpu().numpy()

        # Validation

        P.iloc[i, 2] = sum(perplexities) / len(perplexities)
        # print(P)

        model.eval()

        for _, (data, N) in enumerate(validation_loader):

            padded_data = mapper.pad_sentences(data, N=N)
            inputs, targets = utils.inputs_and_targets_from_sequences(padded_data)
            inputs = mapper.map_sentences_to_padded_embedding(inputs, embedding=embedding,
                                                              embedding_size=hps.embedding_dim, N=N)
            targets = mapper.map_words_to_indices(targets, N=N)

            if cuda:
                inputs = inputs.cuda()
                targets = targets.cuda()

            # Forward pass
            outputs = model(inputs).permute(0, 2, 1)

            # Loss
            loss = criterion(outputs, targets)

            # Update loss
            epoch_validation_loss += loss.detach().cpu().numpy()

        # Save loss for plot
        training_loss.append(epoch_training_loss / len(train_loader))
        validation_loss.append(epoch_validation_loss / len(validation_loader))

        # Print loss every 1 epochs
        if i % 1 == 0:
            print(
                f'Epoch {i} took {time.time() - epoch_start}s, training loss: {training_loss[-1]}, validation loss: {validation_loss[-1]}')

    print("Training finished!\nSaving model information...")
    m_save_path = model_save_path(data_file_size, hps.lstm_h_dim, hps.embedding_dim)
    torch.save(model.state_dict(), m_save_path)
    P.to_csv(perplexity_save_path(data_file_size, hps.lstm_h_dim, hps.embedding_dim), index=False, header=True)
    print(f"Saved model to {m_save_path}")


def init_hps():
    parser = argparse.ArgumentParser()

    parser.add_argument("--lstm_h_dim", type=int, default=600,
                        help="dimension of the hidden layer for lstm")

    parser.add_argument("--embedding_dim", type=int, default=250,
                        help="dimension of the embedding")

    parser.add_argument("--batch_size", type=int, default=64,
                        help="batch size")

    parser.add_argument("--n_epochs", type=int, default=5,
                        help="number of training epochs")

    parser.add_argument('-f', '--file',
                        help='Path for input file. First line should contain number of lines to search in')

    hps = parser.parse_args()

    return hps


if __name__ == "__main__":
    main(load=True)

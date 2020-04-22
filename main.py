import torch
import torch.nn as nn
import torch.nn.functional as F
import fasttext

import argparse

import pandas as pd

import utils
from data_helper import SentenceMapper
from models import LSTM

import time

data_file_size = 1000
train_data_file = "testdata_small.txt"
test_data_file = "testdata_small.txt"
# model_save_path = "model_20k.pth"
# perplexity_save_path = "model_20k.pth"


def perplexity_save_path(data_file_size, lstm_h_dim, embedding_dim):
    return "perplexity_" + str(data_file_size / 1000) + "k_" + str(lstm_h_dim) + "_" + str(embedding_dim) + ".csv"


def model_save_path(data_file_size, lstm_h_dim, embedding_dim):
    return "model_" + str(data_file_size / 1000) + "k_" + str(lstm_h_dim) + "_" + str(embedding_dim) + ".pth"


cuda = torch.cuda.is_available()
if cuda:
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")


def main(load=False):
    # Init hps
    hps = init_hps()

    # Read file
    if load:
        print("Loading file", test_data_file, "for testing")
        lines = utils.read_file(test_data_file)
    else:
        print("Using file", train_data_file, "for training")
        lines = utils.read_file(train_data_file)

    global data_file_size
    data_file_size = len(lines)

    start = time.time()
    unique_words, vocab_size, n = utils.create_unique_words(lines)
    print("vocab_size", vocab_size)
    print("Constructing unique words took:", (time.time() - start))

    word_to_idx, idx_to_word = utils.build_index(unique_words)
    mapper = SentenceMapper(lines, word_to_idx, idx_to_word, n)

    # Construct dataloader
    dataset = utils.ReadLines(train_data_file)
    loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=16, num_workers=8, shuffle=True)

    embedding = fasttext.train_unsupervised(train_data_file, model='cbow', dim=hps.embedding_dim)

    # Init model
    if not load:
        print("Training...")
        model = LSTM(hps, vocab_size)
        train_model(hps, idx_to_word, model, loader, loader, mapper, embedding)
    else:
        print("Loading model...")
        model = LSTM(hps, vocab_size)
        model = nn.DataParallel(model).to(device)
        model.load_state_dict(
            torch.load(model_save_path(data_file_size, hps.lstm_h_dim, hps.embedding_dim), map_location=device))
        model.to(device)
        model.eval()

        for _, data in enumerate(loader):

            padded_data = mapper.pad_sentences(data)

            input_sequences, targets = utils.inputs_and_targets_from_sequences(padded_data)

            inputs = mapper.map_sentences_to_padded_embedding(input_sequences, embedding=embedding,
                                                              embedding_size=hps.embedding_dim)
            inputs = inputs.to(device)
            inputs = inputs[0, :].unsqueeze(0)

            outputs = model(inputs)

            outputs = F.softmax(outputs, dim=2)
            output = torch.topk(outputs, 1, dim=2)[1]

            output = output.squeeze(2).squeeze(0)
            original_input = inputs.squeeze(0)

            input_sequence = input_sequences
            print(input_sequence)

            print('\nPredicted sequence:')
            predicted_sequence = [idx_to_word[c] for c in output.detach().cpu().numpy()]
            print(predicted_sequence)

            prev_word = ""
            for i in range(1, len(predicted_sequence)):
                words = input_sequence[0][:i]
                predicted_next_word = predicted_sequence[i - 1]

                if predicted_next_word == '</s>' and (prev_word == '</s>' or input_sequence[0][i] == '</s>'):
                    break

                prev_word = predicted_next_word

                print(" ".join(list(words)), predicted_next_word)

            break

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

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

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

        for _, data in enumerate(train_loader):

            # ":-D"

            padded_data = mapper.pad_sentences(data)

            inputs, targets = utils.inputs_and_targets_from_sequences(padded_data)

            inputs = mapper.map_sentences_to_padded_embedding(inputs, embedding=embedding,
                                                              embedding_size=hps.embedding_dim)

            targets = mapper.map_words_to_indices(targets)

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
        print(P)

        model.eval()

        for _, data in enumerate(validation_loader):

            # ":-D"

            padded_data = mapper.pad_sentences(data)

            inputs, targets = utils.inputs_and_targets_from_sequences(padded_data)
            inputs = mapper.map_sentences_to_padded_embedding(inputs, embedding=embedding,
                                                              embedding_size=hps.embedding_dim)
            targets = mapper.map_words_to_indices(targets)

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

    torch.save(model.state_dict(), model_save_path(data_file_size, hps.lstm_h_dim, hps.embedding_dim))
    P.to_csv(perplexity_save_path(data_file_size, hps.lstm_h_dim, hps.embedding_dim), index=False, header=True)


def init_hps():
    parser = argparse.ArgumentParser()

    parser.add_argument("--lstm_h_dim", type=int, default=600,
                        help="dimension of the hidden layer for lstm")

    parser.add_argument("--embedding_dim", type=int, default=200,
                        help="dimension of the embedding")

    parser.add_argument("--batch_size", type=int, default=16,
                        help="batch size")

    parser.add_argument("--n_epochs", type=int, default=20,
                        help="number of training epochs")

    parser.add_argument('-f', '--file',
                        help='Path for input file. First line should contain number of lines to search in')

    hps = parser.parse_args()

    return hps


if __name__ == "__main__":
    main(False)

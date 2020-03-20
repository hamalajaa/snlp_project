import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import argparse

import utils
from data_helper import SentenceMapper
from models import LSTM

from timeit import default_timer as timer

data_file = "testdata.txt"
ngram_size = 6

cuda = torch.cuda.is_available()
if cuda:
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

def main():
    # Init hps
    hps = init_hps()

    # Read file
    lines = utils.read_file(data_file)

    start = timer()
    unique_words, vocab_size, n = utils.create_unique_words(lines)

    print("Constructing unique words took:", (timer() - start))

    word_to_idx, idx_to_word = utils.build_index(unique_words)
    mapper = SentenceMapper(lines, word_to_idx, idx_to_word, n)

    # Construct dataloader
    dataset = utils.ReadLines(data_file)
    loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=16, num_workers=8)

    # loader = torch.utils.data.DataLoader(tensor, batch_size=hps.batch_size)

    # Init model
    model = LSTM(hps, vocab_size)

    print("Dummy tests: ")
    train_model(hps, idx_to_word, model, loader, loader, mapper)



def train_model(hps, idx_to_word, model, train_loader, validation_loader, mapper):
    # Hyper-parameters
    num_epochs = hps.n_epochs

    if cuda:
        model = nn.DataParallel(model).cuda()

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()

    if cuda:
        criterion = criterion.cuda()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Track loss
    training_loss, validation_loss = [], []

    for i in range(num_epochs):

        # Track loss
        epoch_training_loss = 0
        epoch_validation_loss = 0

        # Training

        model.train()

        for _, data in enumerate(train_loader):
            train_loop_start = timer()
            
            data_map_start = timer()
            data = mapper.map_sentences_to_indices(data)

            print("Mapping data to tensor took", (timer() - data_map_start))

            inputs, targets = utils.inputs_and_targets_from_sequences(data)
            if cuda:
                inputs = inputs.cuda()
                targets = targets.cuda()

            # Forward pass
            forward_pass_start = timer()
            outputs = model(inputs).permute(0, 2, 1)

            print("Forward pass took", (timer() - forward_pass_start))

            # Loss
            loss_start = timer()
            loss = criterion(outputs, targets)

            print("Loss took", (timer() - loss_start))

            # Backward pass
            backward_pass_start = timer()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print("Backward pass took", (timer() - backward_pass_start))

            # Update loss
            epoch_training_loss += loss.detach().cpu().numpy()

            print("Total", (timer() - train_loop_start))
            print()

        # Validation

        model.eval()

        for _, data in enumerate(validation_loader):

            data = mapper.map_sentences_to_indices(data)

            inputs, targets = utils.inputs_and_targets_from_sequences(data)
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
            print(f'Epoch {i}, training loss: {training_loss[-1]}, validation loss: {validation_loss[-1]}')

        
    print(inputs.max(dim=2)[1][0, :])
    print(targets.max(dim=2)[1][0, :])
    print(outputs.max(dim=2)[1][0, :])

    context = inputs.max(dim=2)[1][0, :]
    target = targets.max(dim=2)[1][0, :]
    output = outputs.max(dim=2)[1][0, :]

    print('\nInput sequence:')
    print([idx_to_word[c] for c in context.detach().cpu().numpy()])

    print('\nTarget sequence:')
    print([idx_to_word[c] for c in target.detach().cpu().numpy()])

    print('\nPredicted sequence:')
    print([idx_to_word[c] for c in output.detach().cpu().numpy()])


def init_hps():
    parser = argparse.ArgumentParser()

    parser.add_argument("--lstm_h_dim", type=int, default=200,
                        help="dimension of the hidden layer for lstm")
    
    parser.add_argument("--embedding_dim", type=int, default=20,
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
    main()

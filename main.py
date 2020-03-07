import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import argparse


import utils
from data_helper import SentenceMapper
from models import LSTM

data_file = "testdata.txt"
ngram_size = 6


def main():
    
    # Init hps
    hps = init_hps()

    # Read file
    lines = utils.read_file(data_file)
    unique_words, vocab_size, n = utils.create_unique_words(lines)

    word_to_idx, idx_to_word = utils.build_index(unique_words)
    mapper = SentenceMapper(lines, word_to_idx, idx_to_word ,n)
    mapper.map_sentences_to_tensors()

    # Sentences in tensor format
    tensor = mapper.map_sentences_to_tensors()

    # Construct dataloader
    loader = torch.utils.data.DataLoader(tensor, batch_size=hps.batch_size)
    
    # Init model
    model = LSTM(hps, vocab_size)

    print("Dummy tests: ")
    train_model(hps, idx_to_word, model, loader, loader)
    


def train_model(hps, idx_to_word, model, train_loader, validation_loader):
    # Hyper-parameters
    num_epochs = hps.n_epochs

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    #optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    # Track loss
    training_loss, validation_loss = [], []

    for i in range(num_epochs):
        
        # Track loss
        epoch_training_loss = 0
        epoch_validation_loss = 0
        
        # Training

        model.train()

        for _, data in enumerate(train_loader):
            
            inputs, targets = utils.inputs_and_targets_from_sequences(data)
            _, target_idx = targets.max(dim=1)

            # Forward pass

            outputs = model(inputs)
            
            # Loss
            loss = criterion(outputs, target_idx)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update loss
            epoch_training_loss += loss.detach().numpy()

        # Validation

        model.eval()

        for _, data in enumerate(validation_loader):
            
            inputs, targets = utils.inputs_and_targets_from_sequences(data)
            _, target_idx = targets.max(dim=1)
            
            # Forward pass
            outputs = model(inputs)
            
            # Loss
            loss = criterion(outputs, target_idx)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update loss
            epoch_validation_loss += loss.detach().numpy()

        # Save loss for plot
        training_loss.append(epoch_training_loss/len(train_loader))
        validation_loss.append(epoch_validation_loss/len(validation_loader))

        # Print loss every 1 epochs
        if i % 1 == 0:
            print(f'Epoch {i}, training loss: {training_loss[-1]}, validation loss: {validation_loss[-1]}')

    print(inputs.max(dim=2)[1][0,:])
    print(targets.max(dim=2)[1][0,:])
    print(outputs.max(dim=2)[1][0,:])

    context = inputs.max(dim=2)[1][0,:]
    target = targets.max(dim=2)[1][0,:]
    output = outputs.max(dim=2)[1][0,:]

    print('\nInput sequence:')
    print([idx_to_word[c] for c in context.detach().numpy()])

    print('\nTarget sequence:')
    print([idx_to_word[c] for c in target.detach().numpy()])

    print('\nPredicted sequence:')
    print([idx_to_word[c] for c in output.detach().numpy()])



def init_hps():

    parser = argparse.ArgumentParser()

    parser.add_argument("--lstm_h_dim",       type=int,   default=200, 
                        help="dimension of the hidden layer for lstm")

    parser.add_argument("--batch_size",       type=int,   default=10, 
                        help="batch size")

    parser.add_argument("--n_epochs",         type=int,   default=50,   
                        help="number of training epochs")

    parser.add_argument('-f', '--file', help='Path for input file. First line should contain number of lines to search in')

    hps = parser.parse_args()

    return hps


if __name__ == "__main__":
    main()







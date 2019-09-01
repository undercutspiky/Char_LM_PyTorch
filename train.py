import os
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils import data

from dataloader import DatasetFactory, PADDING_TOKEN
from model import CharRNN
from util import print_tokens


def save_checkpoint(optimizer, model, epoch, file_path):
    checkpoint_dict = {
        'optimizer': optimizer.state_dict(),
        'model': model.state_dict(),
        'epoch': epoch
    }
    torch.save(checkpoint_dict, file_path)


def load_checkpoint(optimizer, model, file_path):
    if not os.path.exists(file_path):
        return None
    checkpoint_dict = torch.load(file_path)
    epoch = checkpoint_dict['epoch']
    model.load_state_dict(checkpoint_dict['model'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint_dict['optimizer'])
    return epoch


def get_model(dataset, config):
    return CharRNN(vocab_size=dataset.encoder.vocabulary_size,
                   embedding_size=config['model_config']['embedding_size'],
                   hidden_size=config['model_config']['hidden_size'],
                   padding_idx=dataset.encoder.token_to_id(PADDING_TOKEN),
                   n_layers=config['model_config']['n_layers'])


def run_forward_pass_and_get_loss(model, input_x, target_y, lengths):
    input_x = input_x.to(model.device)
    target_y = target_y.to(model.device)
    lengths = lengths.to(model.device)
    predictions = model(input_x, lengths)
    # Mask out padded values
    target_y = target_y.view(-1)  # Flatten out the batch
    mask = (target_y != model.padding_idx)
    target_y *= mask.long()  # Make the target values at padded indices 0
    return model.loss(predictions, target_y, mask)


def validate(dataset, model: CharRNN):
    tmp_hidden = model.hidden
    tmp_loss_func = model.loss_func
    model.reset_intermediate_vars()
    model.loss_func = nn.CrossEntropyLoss(reduction='sum')
    params = {'batch_size': 1,
              'shuffle': False,
              'num_workers': 0 if os.name == 'nt' else 8}
    data_generator = data.DataLoader(dataset, **params)
    cross_entropy = 0
    total_length = 0
    for x_i, y_i, l_i in data_generator:
        total_length += l_i.item()
        cross_entropy += run_forward_pass_and_get_loss(model, x_i, y_i, l_i).item()
        model.detach_intermediate_vars()
    perplexity = np.exp(cross_entropy/total_length)
    bpc = np.log2(perplexity)
    model.hidden = tmp_hidden
    model.loss_func = tmp_loss_func
    return bpc


def run_training(model: CharRNN, dataset, config: dict, validation: bool, valid_dataset):
    optimizer = torch.optim.Adam(model.parameters(), lr=config['initial_lr'])
    epoch = load_checkpoint(optimizer, model, config['filename'])
    if not epoch:
        epoch = 0
    epoch += 1
    params = {'batch_size': config['batch_size'],
              'shuffle': False,
              'num_workers': 0 if os.name == 'nt' else 8}
    data_generator = data.DataLoader(dataset, **params)
    while epoch < config['epochs'] + 1:
        model.reset_intermediate_vars()
        for step, (x_i, y_i, l_i) in enumerate(data_generator):
            loss = run_forward_pass_and_get_loss(model, x_i, y_i, l_i)
            # Gradient descent step
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.5)
            optimizer.step()

            model.detach_intermediate_vars()
            if step % 100 == 0:
                print('Epoch: {} Loss for step {} : {}'.format(epoch, step, round(loss.item(), 4)))
            if step % 1000 == 1:
                gen_sample = model.generate_text(dataset.encoder, 't', 200)
                print_tokens(dataset.encoder.map_ids_to_tokens(gen_sample), config['is_bytes'])
        save_checkpoint(optimizer, model, epoch, config['filename'])
        if validation and epoch % 2:
            bpc = validate(valid_dataset, model)
            print('BPC on validation set: ', bpc)
        if epoch in config['lr_schedule']:
            optimizer = torch.optim.Adam(model.parameters(), lr=config['lr_schedule'][epoch])
        epoch += 1


def main(dataset_name: str):
    print('Preparing training data')
    train_ds = DatasetFactory.get_dataset(dataset=dataset_name, mode='train')
    ds_config = DatasetFactory.get_config(dataset_name)
    print('Training data prepared')
    model = get_model(train_ds, ds_config)
    model.to(model.device)
    valid_ds = DatasetFactory.get_dataset(dataset=dataset_name, mode='valid')
    run_training(model, train_ds, ds_config, True, valid_ds)


def test_model(dataset_name: str):
    test_ds = DatasetFactory.get_dataset(dataset=dataset_name, mode='test')
    ds_config = DatasetFactory.get_config(dataset_name)
    model = get_model(test_ds, ds_config)
    load_checkpoint(optimizer=None, model=model, file_path=ds_config['filename'])
    model.to(model.device)
    bpc = validate(test_ds, model)
    print('BPC on test set: ', bpc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a LSTM network defined in model.py.')
    parser.add_argument('-d', '--dataset',
                        type=str,
                        help='Name of the dataset',
                        default='text8',
                        choices=['text8', 'ptb', 'hutter'])
    args = parser.parse_args()
    main(args.dataset)
    test_model(args.dataset)

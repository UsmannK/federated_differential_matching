import copy
from pathlib import Path
import logging

import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np


class MatchOneLayer(nn.Module):
    def __init__(self, input_size, output_size, num_models):
        super(MatchOneLayer, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.num_models = num_models

        self.pi_li = nn.Parameter(torch.rand(num_models, output_size, output_size))
        self.log_softmax = nn.LogSoftmax(dim=1)
        

    def forward(self, layer_weights):
        # Cost: euclidean distance
        pi_li_2 = self.pi_li ** 2
        global_layer_weights = (torch.sum(torch.matmul(layer_weights, pi_li_2),axis=0) / self.num_models).detach()
        c = torch.cdist(global_layer_weights.T, layer_weights.permute(0,2,1)) #torch.cdist(global_layer_weights, layer_weights) # Get the euclidean distance from theta to local model weights
        # -log of cost
        # log_p = self.log_softmax(c)
        # neg_log_p = -1 * log_p
        # Sum across neurons and models
        loss = torch.sum(pi_li_2/self.output_size * c) / self.num_models # Multiply our "permutation" matrix by -log(c) and square the sum, we want to minimize this

        # Col sums
        col_sums = torch.sum(pi_li_2, dim=1)
        col_diff_to_one = col_sums - torch.ones_like(col_sums)
        squared_col_diff = torch.sum(col_diff_to_one ** 2)

        # Row sums
        row_sums = torch.sum(pi_li_2, dim=2)
        row_diff_to_one = row_sums - torch.ones_like(row_sums)
        squared_row_diff = torch.sum(row_diff_to_one ** 2)

        post_loss = loss + squared_row_diff + squared_col_diff

        return post_loss, [loss, squared_row_diff, squared_col_diff], global_layer_weights.detach()

def prepare_weights(models):
    weights = []
    for model in models:
        cur_weights = []
        for weight_key, weight in model.state_dict().items():
            if len(weight.shape) == 1:
                cur_weights.append(weight.unsqueeze(0).detach())
            elif 'conv' in weight_key:
                weight = weight.reshape(-1, weight.shape[0])
                cur_weights.append(weight.detach())
            else:
                cur_weights.append(weight.T.detach())
        weights.append(cur_weights)
    return weights

def compute_diff_matching(weights, layer_idx, args):
    cur_weights = [w[layer_idx] for w in weights]
    matched_weights, pi_li = get_matched_weights(cur_weights, layer_idx, args)
    return matched_weights, pi_li

def get_matched_weights(cur_weights, layer_idx, args):
    device = torch.device(0) if torch.cuda.is_available() else torch.device("cpu")
    layer_weights = torch.stack(cur_weights)
    best_loss = None
    num_model = layer_weights.shape[0]
    output_size = layer_weights.shape[2]
    input_size = layer_weights.shape[1]

    match = MatchOneLayer(input_size, output_size, num_model)
    match = match.to(device)
    layer_weights = layer_weights.to(device)
    optimizer = optim.Adam(match.parameters(), lr=args.lr)
    for epoch in range(args.diff_match_epochs):
        optimizer.zero_grad()
        loss, losses_arr, global_layer_weights = match(layer_weights)
        
        if best_loss is None or loss < best_loss:
            best_dict = copy.deepcopy(match.state_dict())
            best_loss = loss

        loss.backward()
        optimizer.step()

        if (epoch+1) % 1000 == 0 or epoch == 0:
            logging.debug(f'Epoch {epoch+1:5}')
            logging.debug(f'Loss: {loss}')
            logging.debug(f'Matching Loss: {losses_arr[0]}')
            logging.debug(f'Row Loss: {losses_arr[1]}')
            logging.debug(f'Col Loss: {losses_arr[2]}')
            logging.debug(f'')

    match.load_state_dict(best_dict)
    if args.dump_intermediate_models:
        torch.save(match, Path(args.logdir)/f'match_{layer_idx}.pth')
    loss, losses_arr, global_layer_weights = match(layer_weights)
    pi_li = match.pi_li.detach().cpu() ** 2
    logging.debug(f'Best Loss (should be {best_loss})\nLoss: {loss}\nMatching Loss: {losses_arr[0]}\nRow Loss: {losses_arr[1]}\nCol Loss: {losses_arr[2]}\n')

    return global_layer_weights, pi_li
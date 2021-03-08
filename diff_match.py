import copy
from pathlib import Path
import logging

import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import scipy

from geomloss import SamplesLoss
from layers import SinkhornDistance
import ot


class MatchOneLayer(nn.Module):
    def __init__(self, input_size, output_size, num_models):
        super(MatchOneLayer, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.num_models = num_models

        self.pi_li = nn.Parameter(torch.rand(num_models, output_size, output_size))
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.pdist = nn.PairwiseDistance(p=2)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, layer_weights):
        # Cost: euclidean distance
        pi_li_2 = self.softmax(self.pi_li) ** 2
        pi_li_2 = self.pi_li
        for _ in range(100):
            pi_li_2 = pi_li_2 - torch.logsumexp(pi_li_2, 2).reshape(-1, self.output_size, 1)
            pi_li_2 = pi_li_2 - torch.logsumexp(pi_li_2, 1).reshape(-1, 1, self.output_size)
        pi_li_2 = torch.exp(pi_li_2)
        transported_layer_weights = torch.matmul(layer_weights, pi_li_2)
        global_layer_weights = (torch.sum(transported_layer_weights,axis=0) / self.num_models).detach()
        c = torch.cdist(global_layer_weights.T, layer_weights.permute(0,2,1)) #torch.cdist(global_layer_weights, layer_weights) # Get the euclidean distance from theta to local model weights
        # -log of cost
        log_p = self.log_softmax(c)
        neg_log_p = -1 * log_p
        # Sum across neurons and models
        loss = torch.sum(pi_li_2/self.output_size * c) / self.num_models # Multiply our "permutation" matrix by -log(c) and square the sum, we want to minimize this
        loss = self.pdist(transported_layer_weights[0].T, transported_layer_weights[1].T).sum()
        
        # Col sums
        col_sums = torch.sum(pi_li_2, dim=1)
        col_diff_to_one = col_sums - torch.ones_like(col_sums)
        squared_col_diff = torch.sum(col_diff_to_one ** 2)

        # Row sums
        row_sums = torch.sum(pi_li_2, dim=2)
        row_diff_to_one = row_sums - torch.ones_like(row_sums)
        squared_row_diff = torch.sum(row_diff_to_one ** 2)

        post_loss = loss.sum() + squared_col_diff + squared_row_diff

        return post_loss, [loss, squared_col_diff, squared_row_diff]

def prepare_weights(models):
    weights = []
    for model in models:
        cur_weights = []
        for weight_key, weight in model.state_dict().items():
            if len(weight.shape) == 1:
                cur_weights.append(weight.unsqueeze(0).detach())
            elif len(weight.shape) > 2:
                weight = weight.reshape(weight.shape[0], -1).T
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
    layer_weights = layer_weights.permute(0,2,1).cpu().numpy()
    layer_weights = np.float64(layer_weights)
    weight_weights = np.ones(layer_weights.shape[:-1],dtype=np.double)/layer_weights.shape[1]
    optimizer = optim.Adam(match.parameters(), lr=args.lr)
    for epoch in range(args.diff_match_epochs):
        optimizer.zero_grad()
        loss, losses_arr, global_layer_weights,pi_li = match(layer_weights)
        
        if best_loss is None or loss < best_loss:
            best_dict = copy.deepcopy(match.state_dict())
            best_loss = loss

        loss.backward()
        optimizer.step()
        
        pi_li_ones = torch.zeros_like(pi_li)
        pi_li_ones[[0,1], torch.stack([torch.arange(0,output_size), torch.arange(0,output_size)]).T, pi_li.argmax(2).T] = 1
        transported_ones = torch.matmul(layer_weights, pi_li_ones)

        if (epoch+1) % 10 == 0 or epoch == 0:
            logging.debug(f'Epoch {epoch+1:5}')
            logging.debug(f'Loss: {loss}')
            logging.debug(f'pdist: {torch.nn.PairwiseDistance(p=2)(transported_ones[0].T, transported_ones[1].T).sum()}')
            logging.debug(f'Matching Loss: {losses_arr[0]}')
            logging.debug(f'Row Loss: {losses_arr[1]}')
            logging.debug(f'Col Loss: {losses_arr[2]}')
            logging.debug(f'')

    match.load_state_dict(best_dict)
    if args.dump_intermediate_models:
        torch.save(match, Path(args.logdir)/f'match_{layer_idx}.pth')
    loss, losses_arr, global_layer_weights, pi_li = match(layer_weights)
    pi_li = pi_li*100

    # mu_cardinality, nu_cardinality = output_size,output_size
    # mu = torch.ones(mu_cardinality)/mu_cardinality
    # nu = torch.ones(nu_cardinality)/nu_cardinality

    # X_init =  np.random.normal(0., 1., (output_size, input_size))
    # b = np.ones(output_size)/output_size
    # global_layer_weights = ot.lp.free_support_barycenter(layer_weights, weight_weights, layer_weights[1])
    # X_init_global_weights = ot.lp.free_support_barycenter(layer_weights, weight_weights, X_init)
    # lw0_global_weights = ot.lp.free_support_barycenter(layer_weights, weight_weights, layer_weights[0])

    # M = scipy.spatial.distance.cdist(global_layer_weights,layer_weights[1])
    # barycenter_transport_map = ot.emd(mu,nu,M)*output_size
    # global_layer_weights = (global_layer_weights.T @ barycenter_transport_map).T

    # pi_li = torch.stack([torch.Tensor(ot.emd(mu,nu,scipy.spatial.distance.cdist(layer_weights[x],global_layer_weights))*output_size) for x in range(2)])

    # layer_weights = torch.Tensor(layer_weights).permute(0,2,1)
    # T = ot.emd(mu,nu,torch.cdist(layer_weights[0].T, layer_weights[1].T).detach().cpu()) * output_size
    # T = torch.Tensor(T)
    # pi_li = torch.stack([T, torch.eye(output_size)])
    layer_weights = torch.Tensor(layer_weights).permute(0,2,1)
    transported_layer_weights = torch.matmul(layer_weights.cpu(), pi_li)
    global_layer_weights = (torch.sum(transported_layer_weights,axis=0) / len(transported_layer_weights))

    # Get the exact T_matrices that would transport the originals to the global model
    # T_matrices = []
    # for layer_weight in layer_weights:
    #     C = torch.cdist(layer_weight.T, match.global_model.T)
    #     t_matrix = ot.emd(mu,nu,C.detach().cpu()) * output_size
    #     T_matrices.append(torch.Tensor(t_matrix).to(layer_weights.device))
    # T_matrices = torch.stack(T_matrices)
    # exact_transported_layer_weights = torch.matmul(layer_weights, T_matrices)

    # _,pi_li ,_ = SinkhornDistance(eps=0.001, max_iter=1000, reduction=None)(layer_weights[0].T, layer_weights[1].T)
    # import ot
    # mu = torch.ones(output_size)/output_size
    # nu = torch.ones(output_size)/output_size
    # pi_li = torch.Tensor(ot.emd(mu,nu,torch.cdist(layer_weights[0].T, layer_weights[1].T)))

    # C = torch.cdist(*layer_weights.permute(0,2,1)).cpu()
    # ot_matrix = torch.Tensor(ot.emd(mu,nu,C)) * output_size
    # pi_li_ot = torch.stack([ot_matrix , torch.eye(output_size)])
    # transported_layer_weights = torch.matmul(layer_weights, pi_li)#layer_weights.permute(0,2,1)).permute(0,2,1)

    # transported_layer_weights = torch.matmul(layer_weights, pi_li)
    # global_layer_weights = (torch.sum(transported_layer_weights,axis=0) / len(transported_layer_weights))
    # del layer_weights
    # transport_matrix[pi_li]
    # pi_li = match.pi_li.detach().cpu()# ** 2
    # for _ in range(100):
    #     pi_li = pi_li - torch.logsumexp(pi_li, 2).reshape(-1, match.output_size, 1)
    #     pi_li = pi_li - torch.logsumexp(pi_li, 1).reshape(-1, 1, match.output_size)
    # pi_li = torch.exp(pi_li)
    # logging.debug(f'Best Loss (should be {best_loss})\nLoss: {loss}\nMatching Loss: {losses_arr[0]}\nRow Loss: {losses_arr[1]}\nCol Loss: {losses_arr[2]}\n')



    return torch.Tensor(global_layer_weights), pi_li
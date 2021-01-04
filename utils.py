import logging
import datetime
import argparse
import json
from functools import reduce
from pathlib import Path

import torch

def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--logdir', type=str, required=True, help='Log directory path')
    parser.add_argument('--dataset', type=str, required=True, help="Dataset [mnist/cifar10]")
    parser.add_argument('--datadir', type=str, required=False, default="./data/mnist", help="Data directory")
    parser.add_argument('--init_seed', type=int, required=False, default=0, help="Random seed")

    parser.add_argument('--net_config', type=lambda x: list(map(int, x.split(', '))))
    parser.add_argument('--model_type', type=str, default='mlp', help="Net type [mlp/lenet/vgg9/vgg11]")
    parser.add_argument('--n_nets', type=int , required=True, help="Number of nets to initialize")

    parser.add_argument('--lr', type=float, required=True, help="Learning rate")
    parser.add_argument('--epochs', type=int, required=True, help="Epochs")
    parser.add_argument('--batch_size', type=int, required=False, default=64, help="Batch Size")
    parser.add_argument('--num_train_workers', type=int, required=False, default=0, help="Number of training workers")

    parser.add_argument('--partition', type=str, required=True, help="Partition = homo/hetero/hetero-dir")
    parser.add_argument('--alpha', type=float, required=False, default=0.5, help="Dirichlet distribution constant used for data partitioning")
    parser.add_argument('--diff_match_epochs', type=int, required=False, default=20_000, help="Epochs for differentiable method")
    parser.add_argument('--match_all_layers', action="store_true", help="Match final layers instead of weighted avg")
    parser.add_argument('--skip_bias_match', action="store_true", help="Use weight matrices for bias transformation")

    parser.add_argument('--load_pretrained_models', action="store_true", help="Load saved initial models")

    # Debugging settings
    parser.add_argument('--debug', action="store_true", help="Debug mode, skips some steps")
    parser.add_argument('--speed_run', action="store_true", help="Super debug mode, skip all learning steps")
    parser.add_argument('--skip_eval', action="store_true", help="Skip occasional model evals")
    parser.add_argument('--skip_training', action="store_true", help="Skip initial model training")
    parser.add_argument('--skip_retraining', action="store_true", help="Skip model retraining")
    parser.add_argument('--log_stderr', action="store_true", help="Print logs to stderr as well as logfile")
    parser.add_argument('--dump_intermediate_models', action="store_true", help="Dump intermediate states for debug")

    args = parser.parse_args()

    # Meddle with arguments
    cur_time = str(datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S'))
    log_path = Path(args.logdir) / cur_time
    log_path.mkdir(parents=True, exist_ok=True)
    args.logdir = str(log_path)
    if args.speed_run:
        args.skip_eval = True
        args.skip_training = True
        args.diff_match_epochs = 1
        args.epochs = 1
    dump_path = Path(args.logdir) / 'arguments.json'
    with dump_path.open(mode='w') as f:
        f.write(json.dumps(vars(args)))
    return args

def dump(args, data, name):
    if not args.dump_intermediate_models:
        return
    dump_path = Path(args.logdir) / name
    with dump_path.open(mode='w') as f:
        f.write(data)

def setup_logging(args):
    logging.basicConfig(
        filename = Path(args.logdir) / 'experiment_log.log',
        format = '%(asctime)s %(levelname)-8s %(message)s',
        datefmt = '%m-%d %H:%M:%S', level=logging.DEBUG, filemode='w')
    if args.log_stderr:
        logging.getLogger().addHandler(logging.StreamHandler())

def load_models_from_disk(args):
    models = []
    for n in range(args.n_nets):
        model_path = Path(args.logdir).parent / f'local_model_{args.model_type}_{n}_0.pth'
        models.append(torch.load(model_path))
    return models

def set_params(models, new_weights, layer_idx):
    try:
        len(models)
    except TypeError:
        models = [models]
        
    new_weights = new_weights.T
    for model in models:
        statedict = model.state_dict()
        weight_key = list(statedict.keys())[layer_idx]
        if statedict[weight_key].shape != new_weights.shape:
            new_weights = new_weights.reshape(statedict[weight_key].shape)
        statedict[weight_key] = new_weights.squeeze()
        model.load_state_dict(statedict)

def permute_params(models, pi_li, layer_idx, args):
    statedict = models[0].state_dict()
    if args.skip_bias_match:
        bias_key = list(statedict.keys())[layer_idx+1]
        if 'bias' in bias_key:
            logging.debug(f'permuting [{bias_key}] with last matrix')
            cur_biases = [model.state_dict()[bias_key].detach() for model in models]
            cur_biases = torch.stack([bias.unsqueeze(0) for bias in cur_biases])
            new_biases = torch.sum(torch.matmul(cur_biases, pi_li),axis=0) / len(models)
            set_params(models, new_biases, layer_idx+1)
            layer_idx += 1
    # If there's a next weight to permute then do so
    if len(statedict) > layer_idx+1:
        for idx, model in enumerate(models):
            statedict = model.state_dict()
            weight_key = list(statedict.keys())[layer_idx+1]
            cur_weight = statedict[weight_key].detach()
            if len(cur_weight.shape) > 2:
                original_shape = cur_weight.shape
                cur_weight = cur_weight.data.view(cur_weight.shape[0], cur_weight.shape[1], -1)
                permutation_matrix = pi_li[idx].T.unsqueeze(0).repeat(cur_weight.shape[2], 1, 1).cpu()
                statedict[weight_key] = torch.bmm(cur_weight.permute(2, 0, 1), permutation_matrix).permute(1, 2, 0).view(original_shape)
            else:
                if pi_li[idx].T.shape[1] != cur_weight.T.shape[0]:
                    reshaped_cur_weight = cur_weight.T.reshape(pi_li[idx].T.shape[1], -1, cur_weight.T.shape[-1]).permute(1,0,2)
                    pi_li_augmented = pi_li[idx].T.unsqueeze(0).repeat(reshaped_cur_weight.shape[0],1,1).cpu()
                    permuted_weight = torch.bmm(pi_li_augmented, reshaped_cur_weight)
                    permuted_weight = permuted_weight.permute(1,0,2).reshape(256*16,512).T
                else:
                    permuted_weight = (pi_li[idx].T.cpu() @ cur_weight.T).T
                statedict[weight_key] = permuted_weight
            model.load_state_dict(statedict)

def equalize_models(models):
    sd = models[0].state_dict()
    for model in models:
        model.load_state_dict(sd)
    return models

def compute_weighted_avg_of_weights(batch_weights, traindata_cls_counts):
    n_models = len(traindata_cls_counts)
    class_list = set()
    for class_count in traindata_cls_counts.values():
        class_list.update(list(class_count.keys()))
    n_classes = len(class_list)
    counts = torch.zeros((n_models, n_classes))
    for model_idx, model_cls_counts in traindata_cls_counts.items():
        for cls_idx, cls_count in model_cls_counts.items():
            counts[model_idx][cls_idx] = cls_count
    weights = counts / counts.sum(axis=0)
    new_weights = torch.zeros_like(batch_weights[0][-1])
    for model_idx, weightings in enumerate(weights):
        for output_dim in range(len(weightings)):
            new_weights[:,output_dim] += weightings[output_dim] * batch_weights[model_idx][-1][:,output_dim]
    return new_weights
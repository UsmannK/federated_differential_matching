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
    parser.add_argument('--model_type', type=str, default='mlp', help="Net type [mlp/lenet/vgg9]")
    parser.add_argument('--n_nets', type=int , required=True, help="Number of nets to initialize")

    parser.add_argument('--lr', type=float, required=True, help="Learning rate")
    parser.add_argument('--epochs', type=int, required=True, help="Epochs")
    parser.add_argument('--batch_size', type=int, required=False, default=64, help="Batch Size")

    parser.add_argument('--partition', type=str, required=True, help="Partition = homo/hetero/hetero-dir")
    parser.add_argument('--alpha', type=float, required=False, default=0.5, help="Dirichlet distribution constant used for data partitioning")
    parser.add_argument('--diff_match_epochs', type=int, required=False, default=20_000, help="Epochs for differentiable method")

    parser.add_argument('--load_pretrained_models', action="store_true", help="Load saved initial models")

    # Debugging settings
    parser.add_argument('--debug', action="store_true", help="Debug mode, skips some steps")
    parser.add_argument('--speed_run', action="store_true", help="Super debug mode, skip all learning steps")
    parser.add_argument('--skip_training', action="store_true", help="Skip model retraining")
    parser.add_argument('--log_stderr', action="store_true", help="Print logs to stderr as well as logfile")
    parser.add_argument('--dump_intermediate_models', action="store_true", help="Dump intermediate states for debug")

    args = parser.parse_args()

    # Meddle with arguments
    cur_time = str(datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S'))
    log_path = Path(args.logdir) / cur_time
    log_path.mkdir(parents=True, exist_ok=True)
    args.logdir = str(log_path)
    if args.debug:
        args.diff_match_epochs = 2000
    if args.speed_run:
        args.debug = True
        args.skip_training = True
        args.diff_match_epochs = 1
        args.epochs = 1
    dump(args, json.dumps(vars(args)), 'arguments.json')
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
        model_path = Path(args.logdir).parent / f'local_model_{n}_0.pth'
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
        statedict[weight_key] = new_weights.squeeze()
        model.load_state_dict(statedict)

def permute_params(models, pi_li, layer_idx):
    for idx, model in enumerate(models):
        statedict = model.state_dict()
        weight_key = list(statedict.keys())[layer_idx+1]
        cur_weight = statedict[weight_key].detach()
        if 'conv' in weight_key:
            #TODO: debug over this
            cur_weight = cur_weight.permute(1,2,3,0)
            original_shape = cur_weight.shape
            cur_weight = cur_weight.reshape(cur_weight.shape[0],-1)
            statedict[weight_key] = (pi_li[idx].T @ cur_weight).reshape(original_shape).permute(3,0,1,2)
        else:
            statedict[weight_key] = (pi_li[idx].T @ cur_weight.T).T
        model.load_state_dict(statedict)

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
    new_weights = torch.zeros_like(batch_weights[0][-2])
    new_biases = torch.zeros_like(batch_weights[0][-1][0])
    for model_idx, weightings in enumerate(weights):
        for output_dim in range(len(weightings)):
            # weight biases
            new_biases[output_dim] += weightings[output_dim] * batch_weights[model_idx][-1][0][output_dim]
            new_weights[:,output_dim] += weightings[output_dim] * batch_weights[model_idx][-2][:,output_dim]
    return new_weights, new_biases
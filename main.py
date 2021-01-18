# system imports
import sys
import argparse
import logging
import json
from pathlib import Path
import copy

# library imports
import torch
import torch.multiprocessing
import torch.optim as optim
import torch.nn as nn
import numpy as np
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

# our imports
import utils
import model_zoo
import datasets
import diff_match
import external_models

def eval_model(models):
    train_dl, test_dl = datasets.get_dataloader(args.dataset, args.datadir, args.batch_size, args.batch_size)
    for idx, model in enumerate(models):
        train_acc = utils.compute_accuracy(model, train_dl)
        test_acc = utils.compute_accuracy(model, test_dl)
        logging.debug('')
        logging.debug(f'Model {idx}')
        logging.debug(f'Global Training accuracy: {train_acc}')
        logging.debug(f'Global Test accuracy: {test_acc}')

def train(models, args, net_dataidx_map):
    # Not that interesting
    # parallel model training according to settings in args
    GPU, CPU = 'GPU', 'CPU'
    num_gpus = torch.cuda.device_count()
    mode = GPU if (num_gpus > 0 and torch.cuda.is_available) else CPU
    logging.debug(f'Training on {mode}')
    utils.conditional_log(mode == GPU, f'{num_gpus} GPUs available')

    params = {}

    for model_id, model in enumerate(models):
        dataidxs = net_dataidx_map[model_id]
        train_dl, test_dl = datasets.get_dataloader(args.dataset, args.datadir,
            args.batch_size, args.batch_size, dataidxs, args.num_train_workers)
        logging.debug('')
        logging.debug(f'Network {model_id}')
        logging.debug(f'n_training: {len(train_dl)}')
        logging.debug(f'n_test: {len(test_dl)}')

        if mode == GPU:
            gpu_id = model_id % num_gpus
            device = torch.device(gpu_id)
            logging.debug(f'GPU {gpu_id}')
        else:
            device = torch.device('cpu')

        params[model_id] = {
            'model': model, 
            'train_dl': train_dl,
            'test_dl': test_dl,
            'device': device,
            'optimizer': optim.SGD(model.parameters(), lr=args.lr, momentum=0.9),
            'epoch': 1,
            'epoch_losses': [],
            'criterion': nn.CrossEntropyLoss().to(device),
            'training': True,
            'best_dict': None,
            'best_test_acc': -1
        }
        model.to(device)
        model.train()

    all_epoch_losses = {model_id:[] for model_id in params}
    progress_bars = {model_id: tqdm() for model_id in params}
    train_iterators = {model_id: iter(cur_params['train_dl']) for model_id,cur_params in params.items()}

    for model_id, pbar in progress_bars.items():
        pbar.total = len(params[model_id]['train_dl'])

    keep_training = True
    while keep_training:
        keep_training = False
        for model_id, cur_params in params.items():
            try:
                iterator = train_iterators[model_id]
                x, target = next(iterator)
            except StopIteration:
                if cur_params['training']:
                    all_epoch_losses[model_id].append(sum(cur_params['epoch_losses']) / len(cur_params['epoch_losses']))
                    test_acc = compute_accuracy(cur_params['model'], test_dl, device=cur_params['device'], show_conf_matrix=False)
                    if test_acc > cur_params['best_test_acc']:
                        cur_params['best_test_acc'] = test_acc
                        cur_params['best_dict'] = copy.deepcopy(cur_params['model'].state_dict())
                    cur_params['model'].train()
                if cur_params['epoch'] < args.epochs:
                    cur_params['epoch'] = cur_params['epoch'] + 1
                    progress_bars[model_id].reset()
                    progress_bars[model_id].total = len(cur_params['train_dl'])
                    train_iterators[model_id] = iter(cur_params['train_dl'])
                    cur_params['epoch_losses'] = []
                else:
                    cur_params['training'] = False
            else:
                x, target = x.to(cur_params['device']), target.to(cur_params['device'])
                optimizer = cur_params['optimizer']
                optimizer.zero_grad()
                out = cur_params['model'](x)
                loss = cur_params['criterion'](out, target)
                loss.backward()
                optimizer.step()

                cur_params['epoch_losses'].append(loss)
                avg_loss = sum(cur_params['epoch_losses']) / len(cur_params['epoch_losses'])

                pbar = progress_bars[model_id]
                pbar.update(1)
                pbar.set_description(f'[Epoch {cur_params["epoch"]}/{args.epochs}]  Model {model_id} Loss: {avg_loss:.5f}')
            finally:
                keep_training = keep_training or cur_params['training']
    for model_id in params:
        progress_bars[model_id].close()
    sys.stderr.flush()
    
    train_accs, test_accs = [], []
    for model_id in params:
        cur_params = params[model_id]
        cur_params['model'].load_state_dict(cur_params['best_dict'])
        losses = all_epoch_losses[model_id]
        loss_strings = [f'{loss:.5f}' for loss in losses]

        model, train_dl, test_dl = cur_params['model'], cur_params['train_dl'], cur_params['test_dl']
        train_acc = utils.compute_accuracy(model, train_dl, device=cur_params['device'], show_conf_matrix=False)
        test_acc = utils.compute_accuracy(model, test_dl, device=cur_params['device'], show_conf_matrix=False)
        train_accs.append(train_acc)
        test_accs.append(test_acc)

        logging.debug('')
        logging.debug(f'Model {model_id}')
        logging.debug(f'Last 10 retraining epoch losses: {loss_strings[-10:]}')
        logging.debug(f'Local Training accuracy: {train_acc}')
        logging.debug(f'Local Test accuracy: {test_acc}')
        logging.debug(f"Best Test accuracy: {cur_params['best_test_acc']}")
    return train_accs, test_accs

def run_diff_match(args, models, net_dataidx_map, traindata_cls_counts, model_dump_path):
    train_accs = [[] for _ in range(args.n_nets)]
    test_accs = [[] for _ in range(args.n_nets)]
    # Diff Matching
    global_weights = {}
    batch_weights = diff_match.prepare_weights(models)
    n_layers = len(batch_weights[0])
    weight_names = list(models[0].state_dict().keys())
    # Loop over model layers
    max_matching_layer = n_layers if args.match_all_layers else n_layers-1
    for layer_idx in range(max_matching_layer):
        logging.debug('*'*50)
        logging.debug(f'>> Layer {layer_idx+1} / {n_layers} <<')
        if args.skip_bias_match and 'bias' in weight_names[layer_idx]:
                logging.debug(f'Skipping bias layer: {weight_names[layer_idx]}')
                logging.debug('')
                continue
        # Matching algo
        new_weights, pi_li = diff_match.compute_diff_matching(batch_weights, layer_idx, args)
        global_weights[layer_idx] = new_weights
        utils.set_params(models, new_weights, layer_idx)
        # Permute next layer
        if layer_idx < n_layers-1:
            utils.permute_params(models, pi_li, layer_idx, args)
        if not args.skip_retraining:
            # Freeze layers
            freeze_idx = layer_idx+1 if (args.skip_bias_match and 'bias' in weight_names[layer_idx+1]) else layer_idx
            for model in models:
                for param_idx, param in enumerate(model.parameters()):
                    if param_idx <= freeze_idx:
                        param.requires_grad=False
            # Retrain local models
            cur_train_accs, cur_test_accs = train(models, args, net_dataidx_map)
            for model_idx, model in enumerate(models):
                train_accs[model_idx].append(cur_train_accs[model_idx])
                test_accs[model_idx].append(cur_test_accs[model_idx])
        for model_idx, model in enumerate(models):
            if args.dump_intermediate_models:
                torch.save(model, model_dump_path/f'local_model_{args.model_type}_{model_idx}_{layer_idx+1}.pth')
        if not args.skip_eval:
            # Eval model perf
            eval_model(models)
        # Get newly trained weights
        batch_weights = diff_match.prepare_weights(models)
    logging.debug('Done matching')
        
    if not args.match_all_layers:
        # For final layer+bias, take weighted average of local models
        new_weights = utils.compute_weighted_avg_of_weights(batch_weights, traindata_cls_counts)
        global_weights[n_layers-1] = new_weights

    # Use model_0 as a skeleton for the global model
    global_model = models[0]
    for layer_idx in global_weights:
        utils.set_params(global_model, global_weights[layer_idx], layer_idx)
    if args.dump_intermediate_models:
        torch.save(global_model, model_dump_path/f'global_model_{args.model_type}.pth')
    
    train_dl, test_dl = datasets.get_dataloader(args.dataset, args.datadir, args.batch_size, args.batch_size)
    diff_train_acc = utils.compute_accuracy(global_model, train_dl)
    diff_test_acc = utils.compute_accuracy(global_model, test_dl)
    logging.debug('****** Diff matching ******** ')
    logging.debug(f'Diff matching (Train acc): {diff_train_acc}')
    logging.debug(f'Diff matching (Test acc): {diff_test_acc}')
    return train_accs, test_accs, diff_train_acc, diff_test_acc

def main(args):
    # Set up data loaders
    logging.debug('Partitioning data')
    model_dump_path = Path(args.logdir).parent
    net_dataidx_map, traindata_cls_counts = datasets.partition_data(
                    args.dataset, args.datadir, args.logdir, args.partition, args.n_nets, args.alpha)
    utils.dump(args, json.dumps(net_dataidx_map), 'net_dataidx_map')
    train_accs = [[] for _ in range(args.n_nets)]
    test_accs = [[] for _ in range(args.n_nets)]

    if args.load_pretrained_models:
        logging.debug('Loading pretrained models')
        models = utils.load_models_from_disk(args)
        logging.debug('Evaluating pretrained models')
        if not args.skip_eval:
            eval_model(models)
    else:
        logging.debug('Training new models')
        models = [model_zoo.get_model(args) for _ in range(args.n_nets)]
        models = utils.equalize_models(models)
        if not args.skip_training:
            cur_train_accs, cur_test_accs = train(models, args, net_dataidx_map)
            for idx, model in enumerate(models):
                train_accs[idx].append(cur_train_accs[idx])
                test_accs[idx].append(cur_test_accs[idx])
                if args.dump_intermediate_models:
                    model_dump_path = Path(args.logdir).parent
                    torch.save(model, model_dump_path / f'local_model_{args.model_type}_{idx}_0.pth')
    
    diff_match_train_accs, diff_match_test_accs, diff_match_global_train_acc, diff_match_global_test_acc = run_diff_match(args,
                                                                                                            models, net_dataidx_map, 
                                                                                                            traindata_cls_counts, model_dump_path)

if __name__ == '__main__':
    args = utils.get_parser()
    seed = args.init_seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.multiprocessing.set_sharing_strategy('file_system')

    utils.setup_logging(args)
    main(args)
# system imports
import sys
import argparse
import logging
import json
from pathlib import Path

# library imports
import torch
import torch.multiprocessing
import torch.optim as optim
import torch.nn as nn
import numpy as np
from tqdm import tqdm

# our imports
import utils
import model_zoo
import datasets
import diff_match

def conditional_log(condition, message):
    if condition:
        logging.debug(message)

def compute_accuracy(model, dataloader):
    model.cpu()
    model.eval()
    true_labels_list, pred_labels_list = np.array([]), np.array([])
    correct, total = 0, 0
    for x, target in dataloader:
        out = model(x)
        _, pred_label = torch.max(out.data, 1)

        total += x.data.size()[0]
        correct += (pred_label == target.data).sum().item()

        pred_labels_list = np.append(pred_labels_list, pred_label.cpu().numpy())
        true_labels_list = np.append(true_labels_list, target.data.cpu().numpy())
    return correct/float(total)

def eval_model(models):
    train_dl, test_dl = datasets.get_dataloader(args.dataset, args.datadir, args.batch_size, args.batch_size)
    for idx, model in enumerate(models):
        train_acc = compute_accuracy(model, train_dl)
        test_acc = compute_accuracy(model, test_dl)
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
    conditional_log(mode == GPU, f'{num_gpus} GPUs available')

    params = {}

    for model_id, model in enumerate(models):
        dataidxs = net_dataidx_map[model_id]
        train_dl, test_dl = datasets.get_dataloader(args.dataset, args.datadir,
            args.batch_size, args.batch_size, dataidxs)
        train_acc = compute_accuracy(model, train_dl)
        test_acc = compute_accuracy(model, test_dl)
        logging.debug('')
        logging.debug(f'Network {model_id}')
        logging.debug(f'n_training: {len(train_dl)}')
        logging.debug(f'n_test: {len(test_dl)}')
        logging.debug(f'Pre-Training Training accuracy: {train_acc}')
        logging.debug(f'Pre-Training Test accuracy: {test_acc}')

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
            'training': True
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
        losses = all_epoch_losses[model_id]
        loss_strings = [f'{loss:.5f}' for loss in losses]

        model, train_dl, test_dl = cur_params['model'], cur_params['train_dl'], cur_params['test_dl']
        train_acc = compute_accuracy(model, train_dl)
        test_acc = compute_accuracy(model, test_dl)
        train_accs.append(train_acc)
        test_accs.append(test_acc)

        logging.debug('')
        logging.debug(f'Model {model_id}')
        logging.debug(f'Last 10 retraining epoch losses: {loss_strings[-10:]}')
        logging.debug(f'Local Training accuracy: {train_acc}')
        logging.debug(f'Local Test accuracy: {test_acc}')
    return train_accs, test_accs

def main(args):
    # Set up data loaders
    logging.debug('Partitioning data')
    model_dump_path = Path(args.logdir).parent
    net_dataidx_map, traindata_cls_counts = datasets.partition_data(
                    args.dataset, args.datadir, args.logdir, args.partition, args.n_nets, args.alpha)
    utils.dump(args, json.dumps(net_dataidx_map), 'net_dataidx_map')
    train_accs = [[] for _ in range(args.n_nets)]
    test_accs = [[] for _ in range(args.n_nets)]
    global_weights = {}

    if args.load_pretrained_models:
        logging.debug('Loading pretrained models')
        models = utils.load_models_from_disk(args)
        logging.debug('Evaluating pretrained models')
        if not args.debug:
            eval_model(models)
    else:
        logging.debug('Training new models')
        models = [model_zoo.get_model(args) for _ in range(args.n_nets)]
        if not args.skip_training:
            cur_train_accs, cur_test_accs = train(models, args, net_dataidx_map)
            for idx, model in enumerate(models):
                train_accs[idx].append(cur_train_accs[idx])
                test_accs[idx].append(cur_test_accs[idx])
                if args.dump_intermediate_models:
                    model_dump_path = Path(args.logdir).parent
                    torch.save(model, model_dump_path / f'local_model_{idx}_0.pth')

    # Diff Matching
    batch_weights = diff_match.prepare_weights(models)
    n_layers = len(batch_weights[0])
    # Loop over model layers
    max_matching_layer = n_layers if args.match_all_layers else n_layers-2
    for layer_idx in range(max_matching_layer):
        logging.debug('*'*50)
        logging.debug(f'>> Layer {layer_idx+1} / {n_layers} <<')
        if args.skip_bias_match:
            if layer_idx % 2 == 1:
                logging.debug('Skipping bias layer')
                logging.debug('')
                continue
        # Matching algo
        new_weights, pi_li = diff_match.compute_diff_matching(batch_weights, layer_idx, args)
        global_weights[layer_idx] = new_weights
        utils.set_params(models, new_weights, layer_idx)
        # Permute next layer
        if layer_idx < n_layers-1:
            utils.permute_params(models, pi_li, layer_idx, args)
        if not args.skip_training:
            # Freeze layers
            freeze_idx = layer_idx+1 if args.skip_bias_match else layer_idx
            for model in models:
                for param_idx, param in enumerate(model.parameters()):
                    if param_idx <= freeze_idx:
                        param.requires_grad=False
            # Retrain local models
            cur_train_accs, cur_test_accs = train(models, args, net_dataidx_map)
            for model_idx, model in enumerate(models):
                train_accs[model_idx].append(cur_train_accs[model_idx])
                test_accs[model_idx].append(cur_test_accs[model_idx])
                if args.dump_intermediate_models:
                    torch.save(model, model_dump_path/f'local_model_{model_idx}_{layer_idx+1}.pth')
        if not args.debug:
            # Eval model perf
            eval_model(models)
        # Get newly trained weights
        batch_weights = diff_match.prepare_weights(models)
    logging.debug('Done matching')
        
    if not args.match_all_layers:
        # For final layer+bias, take weighted average of local models
        new_weights, new_biases = utils.compute_weighted_avg_of_weights(batch_weights, traindata_cls_counts)
        global_weights[n_layers-2] = new_weights
        global_weights[n_layers-1] = new_biases

    global_model = model_zoo.get_model(args)
    for layer_idx in global_weights:
        utils.set_params(global_model, global_weights[layer_idx], layer_idx)
    if args.dump_intermediate_models:
        torch.save(global_model, model_dump_path/'global_model.pth')
    
    train_dl, test_dl = datasets.get_dataloader(args.dataset, args.datadir, args.batch_size, args.batch_size)
    diff_train_acc = compute_accuracy(global_model, train_dl)
    diff_test_acc = compute_accuracy(global_model, test_dl)
    logging.debug('****** Diff matching ******** ')
    logging.debug(f'Diff matching (Train acc): {diff_train_acc}')
    logging.debug(f'Diff matching (Test acc): {diff_test_acc}')

if __name__ == '__main__':
    args = utils.get_parser()
    seed = args.init_seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.multiprocessing.set_sharing_strategy('file_system')

    utils.setup_logging(args)
    main(args)
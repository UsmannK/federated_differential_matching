import torch
import logging
import utils
import datasets

def train(models, args, net_dataidx_map):
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
        train_acc = utils.compute_accuracy(model, train_dl)
        test_acc = utils.compute_accuracy(model, test_dl)
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
        train_acc = utils.compute_accuracy(model, train_dl)
        test_acc = utils.compute_accuracy(model, test_dl)
        train_accs.append(train_acc)
        test_accs.append(test_acc)

        logging.debug('')
        logging.debug(f'Model {model_id}')
        logging.debug(f'Last 10 retraining epoch losses: {loss_strings[-10:]}')
        logging.debug(f'Local Training accuracy: {train_acc}')
        logging.debug(f'Local Test accuracy: {test_acc}')
    return train_accs, test_accs
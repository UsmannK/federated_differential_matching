# Federated Differential Matching

This repo contains a codebase designed to experiment around adding new matching methods to FedMA. The main file is `main.py` and it should be run using the options specified in `utils.py`. The matching code itself is in `diff_match.py`

In `main.py` the loop of importance is in the `run_diff_match` function. It does a layer-wise matching run across N models similar to FedMA. The `--skip-retraining` option skips retraining between layers. It sends the matching to `run_diff_match` which is where you can implement various matching methods. Currently there is code for learning the matching matrix via SGD, which is not very viable.

What has not worked:
- Using SGD to learn a permutation matrix for the layers
- Using an Optimal Transport solver to match layers between heterogeneously trained networks.

What is promising:

- Using an Optimal Transport solver (like the POT package) to match layers between homogeneously trained networks
    - This works but is not much better than FedAvg. I suggest trying to add an OT component _to_ FedAvg and see what that looks like.

To see how to use an OT solver to match networks, see: https://github.com/sidak/otfusion

-----

Example of a 1-shot merge of 5x VGG-9 homogeneously trained models (CIFAR10):

command: `python main.py --logdir logs/cifar_test --dataset cifar10 --datadir data/cifar10 --n_nets 5 --dump_intermediate_models --partition homo --lr 0.01 --epochs 25 --model_type vgg9 --skip_eval --skip_retraining --match_all_layers --log_stderr`

To use MNIST and MLP change the `--dataset` and `model_type` as per `utils.py`

```
Model 0
Last 10 retraining epoch losses: ['1.26606', '1.21634', '1.16008', '1.10885', '1.08537', '1.02599', '0.98416', '0.94997', '0.90053', '0.85081']
Local Training accuracy: 0.7349
Local Test accuracy: 0.7033
Best Test accuracy: 0.7033

Model 1
Last 10 retraining epoch losses: ['1.28904', '1.21895', '1.16897', '1.12750', '1.06396', '1.03258', '0.98671', '0.93217', '0.89650', '0.86378']
Local Training accuracy: 0.7366
Local Test accuracy: 0.7037
Best Test accuracy: 0.7037

Model 2
Last 10 retraining epoch losses: ['1.27366', '1.25162', '1.20062', '1.12890', '1.09396', '1.04567', '1.00820', '0.96150', '0.94080', '0.89861']
Local Training accuracy: 0.7354
Local Test accuracy: 0.6982
Best Test accuracy: 0.6982

Model 3
Last 10 retraining epoch losses: ['1.27082', '1.22383', '1.15883', '1.11653', '1.06384', '1.02371', '0.99501', '0.94721', '0.90649', '0.88344']
Local Training accuracy: 0.7184
Local Test accuracy: 0.6936
Best Test accuracy: 0.6936

Model 4
Last 10 retraining epoch losses: ['1.26593', '1.22068', '1.18205', '1.13190', '1.09292', '1.02878', '0.99948', '0.95713', '0.90597', '0.89311']
Local Training accuracy: 0.7108
Local Test accuracy: 0.6838
Best Test accuracy: 0.6838

****** Diff matching ******** 
Diff matching (Train acc): 0.63376
Diff matching (Test acc): 0.6511
```

Trained to .7, matched and merged at 0.65 with no retraining
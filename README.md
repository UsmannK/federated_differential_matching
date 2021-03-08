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
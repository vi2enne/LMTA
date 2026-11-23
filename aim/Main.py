import os
os.environ.setdefault("TF32", "0")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")

import argparse
import json
import random
import time
from pathlib import Path

import networkx as nx
import numpy as np
import torch

from Agent import Agent
from Env import Env
from param_sets import param_sets


def set_seed_all(seed: int, deterministic: bool = True, num_torch_threads: int = 1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if num_torch_threads is not None and num_torch_threads > 0:
        try:
            torch.set_num_threads(num_torch_threads)
        except Exception:
            pass

    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        except Exception:
            pass
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass


def generate_job_name(params):
    return (f"LMTA-AIM_n{params['n']}_b{params['budget']}_T{params['T']}"
            f"_sg{params['num_subgoals']}_sim{params['num_simulations']}_seed{params['seed']}")


def create_model(params):
    num_nodes = params['n']
    g = nx.erdos_renyi_graph(num_nodes, p=min(1.0, 5.0 / num_nodes), directed=True, seed=params.get('seed', 42))
    env = Env(graph=g, budget=params['budget'], T=params['T'])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return Agent(env=env, config=params).to(device)


def run_training(params):
    job_name = generate_job_name(params)
    set_seed_all(params.get('seed', 42))
    model = create_model(params)
    model._job_name = job_name
    print(f"===== TRAINING: {job_name} =====", flush=True)
    model.train_model(
        num_epochs=params['num_epoch'],
        games_per_epoch=params['games_per_epoch'],
        train_steps_per_epoch=params['train_steps_per_epoch'],
        validation_games=params['validation_games']
    )
    return Path("models") / f"{job_name}_best_model.pth"


def evaluate_checkpoint(params, checkpoint):
    model = create_model(params)
    model.load_state_dict(torch.load(checkpoint, map_location=next(model.parameters()).device))
    rewards = model.evaluate(params['num_test'], split='heldout')
    output = dict(domain='AIM', num_nodes=params['test_n'], T=params['T'], budget=params['test_budget'],
                  seed=params['seed'], num_simulations=params['num_simulations'], checkpoint=str(checkpoint),
                  config=params, rewards=rewards, mean_reward=float(np.mean(rewards)),
                  instance_seed_keys=[[params['seed'], params['test_namespace'], i] for i in range(len(rewards))])
    tmp = Path("heldout.json.tmp")
    tmp.write_text(json.dumps(output, indent=2) + "\n")
    os.replace(tmp, 'heldout.json')
    print(f"Held-out mean={np.mean(rewards):.4f} episodes={len(rewards)}", flush=True)


def run_experiment(params, checkpoint=None):
    if params['test_namespace'] < 2:
        raise ValueError('Test namespace must be independent of training and validation')
    if params['test_budget'] != params['budget']:
        raise ValueError('Intensity options are defined relative to the training budget')
    if min(params[k] for k in ('n', 'test_n', 'T', 'num_epoch', 'games_per_epoch',
                               'num_simulations', 'validation_games', 'num_test')) < 1:
        raise ValueError("Counts must be positive")
    if checkpoint is None:
        checkpoint = run_training(params)
    evaluate_checkpoint(params, checkpoint)


def build_arg_parser():
    p = argparse.ArgumentParser(description="LMTA on adaptive influence maximization.")
    for key, value in param_sets[0].items():
        if key == 'index':
            continue
        p.add_argument(f'--{key}', type=type(value), default=None)
    p.add_argument('--checkpoint', help='Evaluate an existing checkpoint instead of training')
    p.add_argument('--run_dir', help='Directory for models, results and heldout.json')
    return p


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    params = dict(param_sets[0])
    params.update({k: v for k, v in vars(args).items() if v is not None and k in params})
    checkpoint = Path(args.checkpoint).resolve() if args.checkpoint else None
    run_dir = Path(args.run_dir) if args.run_dir else Path('runs') / generate_job_name(params)
    run_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(run_dir)
    set_seed_all(params['seed'])
    start = time.time()
    run_experiment(params, checkpoint)
    print(f"Finished in {(time.time() - start) / 60:.2f} minutes.", flush=True)

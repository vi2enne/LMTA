import os
os.environ.setdefault("TF32", "0")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")

import argparse
import json
from pathlib import Path
import random
import numpy as np
import torch

from Agent import MuZeroAgent
from Env import Env


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
        if torch.cuda.is_available():
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            if hasattr(torch.backends.cuda, 'enable_cudnn_sdp'):
                torch.backends.cuda.enable_cudnn_sdp(False)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        except Exception:
            pass
        torch.use_deterministic_algorithms(True)


DEFAULT_CONFIG = {
    'seed': 42,
    'test_namespace': 2,

    'num_subgoals': 32,
    'action_space_size': 32,

    'budget': 60,
    'T': 10,
    'node_dim': 7,
    'num_nodes_train': 500,
    'num_nodes_test': 500,

    'num_simulations': 250,
    'discount': 0.997,
    'pb_c_base': 19652,
    'pb_c_init': 1.5,
    'root_dirichlet_alpha': 0.2,
    'root_exploration_fraction': 0.20,
    'search_prior_mix': 0.5,
    'search_temperature': 1.0,

    'num_epochs': 50,
    'games_per_epoch': 24,
    'train_per_epoch': 32,
    'batch_size': 8,
    'buffer_size': 1000,
    'num_unroll_steps': 5,
    'td_steps': 10,

    'lr_hl': 1e-3,
    'budget_entropy_beta': 0.02,
    'budget_policy_warmup_epochs': 3,

    'embed_dim': 128,
    'll_buffer_capacity': 10000,

    'pull_loss_weight': 0.10,
    'hl_push_weight': 0.10,
    'order_loss_weight': 0.10,
    'slowness_margin': 0.10,
    'consistency_weight': 1.0,

    'mi_critic_hidden_dim': 128,
    'margin_predictor_hidden_dim': 64,
    'margin_predictor_dropout': 0.1,
    'slowness_margin_min': 0.01,
    'slowness_margin_max': 2.0,
    'hl_xi': 1.0,
    'hl_margin_l2': 1e-4,

    'option_max_multiplier': 3.0,
    'option_sigma_min': 0.5,
    'option_sigma_frac': 0.1,
    'option_adjust_nats': 3.0,

    'num_test': 50,
    'validation_games': 50,
    'validation_interval': 1,
    'rollout_batch_size': 24,
    'output_dir': None,
    'checkpoint': None,
}


def generate_job_name(params):
    return (f"LMTA-SOP-n{params['num_nodes_train']}-sg{params['num_subgoals']}-"
            f"b{params['budget']}-T{params['T']}-sims{params['num_simulations']}")


def build_arg_parser():
    p = argparse.ArgumentParser(description="LMTA on the stochastic orienteering problem.")
    for key, value in DEFAULT_CONFIG.items():
        if key in ('output_dir', 'checkpoint'):
            continue
        p.add_argument(f'--{key}', type=type(value), default=None)
    p.add_argument('--output_dir', help='Fresh output directory for one training seed')
    p.add_argument('--checkpoint', help='Evaluate an existing checkpoint instead of training')
    return p


def main(config):
    test_namespace = config['test_namespace']
    if test_namespace < 2:
        raise ValueError('Test namespace must be independent of training and validation')
    if config['action_space_size'] != config['num_subgoals']:
        raise ValueError('One tree action is required for every subgoal')
    seed = int(config['seed'])
    job = generate_job_name(config)
    path = Path(config['output_dir'] or Path('runs') / f'{job}_seed{seed}')
    path.mkdir(parents=True, exist_ok=True)
    config['output_dir'] = str(path.resolve())
    training = not config['checkpoint']
    if training and any((path / name).exists() for name in ('learning.csv', 'best.pth', 'final.pth', 'heldout.json')):
        raise FileExistsError(f'Choose a fresh training output directory: {path}')
    if (path / 'heldout.json').exists():
        raise FileExistsError(f'Choose a fresh evaluation output directory: {path}')
    set_seed_all(seed)
    chosen = path / 'best.pth' if training else Path(config['checkpoint'])
    env = Env.from_seed(config['num_nodes_train'], config['budget'], config['T'], seed, 0, 0)
    agent = MuZeroAgent(env, config)
    print(f"SOP {job} seed={seed} device={agent.device} output={path}", flush=True)
    if training:
        agent.train(job)
    agent.load_state_dict(torch.load(chosen, map_location='cpu', weights_only=True))
    rewards = agent.sample_instances(config['num_test'], test_namespace)
    result = dict(domain='SOP', num_nodes=config['num_nodes_test'], T=config['T'], budget=config['budget'],
                  seed=seed, num_simulations=config['num_simulations'], checkpoint=str(chosen.resolve()),
                  config=config, instance_seed_keys=[[seed, test_namespace, i] for i in range(len(rewards))],
                  rewards=rewards, hl_updates=agent.hl_learning_steps, mean_reward=float(np.mean(rewards)))
    with (path / 'heldout.json').open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False)
    print(f"[FINAL-EVAL] SOP held-out average reward: {result['mean_reward']:.6f}", flush=True)


if __name__ == '__main__':
    cfg = dict(DEFAULT_CONFIG)
    args = build_arg_parser().parse_args()
    cfg.update({k: v for k, v in vars(args).items() if v is not None})
    main(cfg)

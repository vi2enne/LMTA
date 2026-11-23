import random

import networkx as nx
import numpy as np
import torch

from Env import Env
from latent_search import search_network


def search(agent, root_latents, max_depth, training):
    return search_network(agent, agent.config, root_latents, max_depth, training)


@torch.no_grad()
def evaluate_games(agent, num_episodes, split, batch_size=16):
    cfg = agent.config
    n = cfg['test_n'] if split == 'heldout' else cfg['n']
    k = cfg['test_budget'] if split == 'heldout' else cfg['budget']
    horizon = cfg['T']
    device = agent.subgoal_embeddings.weight.device
    if not hasattr(agent, '_evaluation_graphs'):
        agent._evaluation_graphs = {}
    scores = []
    for first in range(0, num_episodes, batch_size):
        envs, generators = [], []
        for index in range(first, min(num_episodes, first + batch_size)):
            graph_seed = agent.episode_seed(split, index)
            cascade_seed = agent.episode_seed(split, index, 'cascade')
            budget_seed = agent.episode_seed(split, index, 'budget')
            key = n, k, horizon, graph_seed
            env = agent._evaluation_graphs.get(key)
            if env is None:
                graph = nx.erdos_renyi_graph(n, min(1.0, 5.0 / n), directed=True, seed=graph_seed)
                env = Env(graph, budget=k, T=horizon, rng=random.Random(cascade_seed))
                agent._evaluation_graphs[key] = env
            env.reset()
            env.rng.seed(cascade_seed)
            envs.append(env)
            generators.append(torch.Generator(device=device).manual_seed(budget_seed))
        adj = torch.as_tensor(np.stack([e.A for e in envs]), device=device)
        done = [False] * len(envs)
        totals = np.zeros(len(envs), dtype=np.int64)
        for day in range(horizon):
            active = [i for i, e in enumerate(envs) if not done[i] and e.budget_r and e.state[:, 0].any()]
            actions = [[] for _ in envs]
            if active:
                states = torch.as_tensor(np.stack([envs[i].state for i in active]), device=device)
                matrices = adj[active]
                context = [agent.context(envs[i].budget_r, horizon - day, k) for i in active]
                latents = agent.representation_net(states, matrices, context)
                visits, _ = search(agent, latents, horizon - day, training=False)
                ids = torch.as_tensor(visits.argmax(1), device=device)
                goals = agent.subgoal_embeddings(ids)
                logits = agent.budget_raw_logits(latents, goals, ids, [envs[i].budget_r for i in active],
                                                 [horizon - day] * len(active))
                remaining = torch.tensor([envs[i].budget_r for i in active], device=device)
                feasible = torch.arange(logits.shape[-1], device=device) <= remaining[:, None]
                probs = logits.masked_fill(~feasible, -torch.inf).softmax(-1)
                budgets = torch.cat([torch.multinomial(p, 1, generator=generators[i])
                                     for i, p in zip(active, probs)]).tolist()
                selections = agent.select_seed_batch(states, matrices, goals, budgets, epsilon=0.0)
                for i, selected in zip(active, selections):
                    actions[i] = selected
            for i, env in enumerate(envs):
                if not done[i]:
                    env.state, reward, done[i], _ = env.step(env.state, actions[i])
                    env.budget_r -= len(actions[i])
                    totals[i] += reward
            if all(done):
                break
        scores.extend(totals.tolist())
    return scores

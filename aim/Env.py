import random

import networkx as nx
import numpy as np


class Env:

    def __init__(self, graph, budget=20, T=10, p_active=1, weighted=True, rng=None):
        self.rng = random if rng is None else rng
        self.graph = graph
        self.n = len(graph)
        self.nodes = list(range(self.n))
        self.weighted = weighted
        self.T = T
        self.budget = budget
        self.p_active = p_active
        self.neighbors = [tuple(graph.successors(v)) for v in self.nodes]
        self.probability = [1 / degree if degree else 0.0 for _, degree in graph.in_degree(self.nodes)]
        if weighted:
            for u, v in graph.edges():
                graph[u][v]['weight'] = self.probability[v]
        self.A = nx.to_numpy_array(graph, nodelist=self.nodes, dtype=np.float32)
        self.reset()

    def reset(self):
        self.state = np.zeros((self.n, 3), dtype=np.float32)
        self.state[:, 0] = 1
        self.budget_r = self.budget

    def step(self, state, action):
        next_state = state.copy()
        if len(action) > self.budget_r or len(set(action)) != len(action) or any(
            v < 0 or v >= self.n or not state[v, 0] for v in action
        ):
            raise ValueError("illegal seed selection")
        seeds = self.activate_seeds(action)
        self.set_state_index(next_state, seeds, 1)
        sources = self.get_nodes_type(next_state, "active")
        successful = {}
        for v in sources:
            for u in self.neighbors[v]:
                if next_state[u, 0] and self.rng.random() < self.probability[u]:
                    successful.setdefault(u, []).append(v)
        activated = list(successful)
        positions = {v: i for i, v in enumerate(action)}
        seed_set = set(seeds)
        self.primitive_rewards = [float(v in seed_set) for v in action]
        if action:
            for parents in successful.values():
                share = 1.0 / len(parents)
                for v in parents:
                    self.primitive_rewards[positions.get(v, -1)] += share
        self.set_state_index(next_state, sources, 2)
        self.set_state_index(next_state, activated, 1)
        reward = len(seeds) + len(activated)
        done = (not np.any(next_state[:, 0]) or
                (self.budget_r == len(action) and not activated))
        return next_state, reward, done, reward == 0

    def set_state_index(self, state, nodes, state_index):
        state[nodes, :] = 0
        state[nodes, state_index] = 1

    def activate_seeds(self, seeds):
        if self.p_active == 1:
            return list(seeds)
        return [v for v in seeds if self.rng.random() < self.p_active]

    def get_nodes_type(self, state, type):
        index = {"inactive": 0, "active": 1, "removed": 2}[type]
        return np.flatnonzero(state[:, index]).tolist()

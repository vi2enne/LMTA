import numpy as np


class Env:
    def __init__(self, loc, initial_profit=None, budget=20, T=10, max_daily_distance=1.5,
                 distance_penalty=5, start_node=0, rng=None):
        self.loc = np.asarray(loc)
        self.n = len(self.loc)
        self.T = T
        self.budget = budget
        self.max_daily_distance = max_daily_distance
        self.distance_penalty = distance_penalty
        self.start_node = start_node
        self.rng = np.random if rng is None else rng
        self.policy_rng = self.rng
        self.initial_profit = (self.rng.uniform(0.5, 1.0, self.n) if initial_profit is None
                               else np.asarray(initial_profit))
        self.profit_factor = 0.05
        self.profit_min = 0.3
        self.eta = 0.1
        self.dist = np.sqrt(((self.loc[:, None] - self.loc[None, :]) ** 2).sum(axis=2))
        self.distance_scale = max(float(self.dist.max()), 1e-12)
        self.normalized_loc = self.loc / max(float(self.loc.max()), 1e-12)
        self.reset()

    @classmethod
    def from_seed(cls, n, budget, T, seed, split, episode):
        rng = np.random.default_rng(np.random.SeedSequence([seed, split, episode]))
        env = cls(rng.uniform(0, 100, (n, 2)), budget=budget, T=T, rng=rng)
        env.policy_rng = np.random.default_rng(np.random.SeedSequence([seed, split, episode, 1]))
        return env

    def reset(self):
        self.t = 1
        self.profit = self.initial_profit.copy()
        self.profit[self.start_node] = 0
        self.remaining_budget = self.budget
        self.current_node = self.start_node

    def get_state(self):
        state = np.column_stack((self.normalized_loc, self.profit)).astype(np.float32)
        context = np.empty((self.n, 4), dtype=np.float32)
        context[:, 0] = self.dist[self.current_node] / self.distance_scale
        context[:, 1] = self.dist[self.start_node] / self.distance_scale
        context[:, 2] = (self.T - self.t + 1) / self.T
        context[:, 3] = self.remaining_budget / self.budget
        return np.column_stack((state, context))

    def get_not_visited(self):
        return np.flatnonzero(self.profit > 0)

    def get_context(self):
        return np.array([self.current_node, self.start_node])

    def cal_total_distance(self, action, context=None):
        current, depot = self.get_context() if context is None else context
        route = [current, *action]
        if self.t == self.T:
            route.append(depot)
        return self.dist[route[:-1], route[1:]].sum() / self.distance_scale

    def cal_reward(self, nodes):
        return self.profit[nodes].sum() - self.eta * max(len(nodes) - 1, 0) ** 2

    def reward_by_simulation(self, state, context, action):
        distance = self.cal_total_distance(action, context)
        profit = state[action, 2].sum() - self.eta * max(len(action) - 1, 0) ** 2
        return profit - self.distance_penalty * max(distance - self.max_daily_distance, 0)

    def step(self, action):
        if self.t > self.T or len(action) > self.remaining_budget:
            raise ValueError("Day or visit budget exceeded")
        action = list(action)
        if (len(set(action)) != len(action) or
                any(not isinstance(i, (int, np.integer)) or i < 0 or i >= self.n or self.profit[i] <= 0
                    for i in action)):
            raise ValueError("A tour can only visit distinct, unvisited cities")
        reward = self.cal_reward(action) - self.distance_penalty * max(
            self.cal_total_distance(action) - self.max_daily_distance, 0)
        self.remaining_budget -= len(action)
        self.profit[action] = 0
        if action:
            self.current_node = action[-1]
        shocks = self.profit_factor * self.rng.uniform(-1, 1, self.n)
        alive = self.profit > 0
        self.profit[alive] = np.maximum(self.profit[alive] + shocks[alive], self.profit_min)
        self.t += 1
        return self.get_state(), float(reward), self.t > self.T

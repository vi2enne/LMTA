from collections import deque, namedtuple
from dataclasses import dataclass
import random


Transition_sub = namedtuple(
    'Transition_sub', ['state', 'adj', 'option', 'action', 'next_state', 'context', 'next_context', 'bellman'])


@dataclass
class BellmanTarget:
    reward: float
    next_discount: float
    next_state: object = None
    next_goal: object = None

    def passive_day(self, gain, gamma):
        self.reward += self.next_discount * gain
        self.next_discount *= gamma

    def link(self, state, goal):
        self.next_state = state
        self.next_goal = goal


class ReplayBuffer_sub:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, *transition):
        self.memory.append(Transition_sub(*transition))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size) if len(self.memory) >= batch_size else None

    def __len__(self):
        return len(self.memory)


class Game:
    def __init__(self, num_subgoals, discount, adj):
        self.num_subgoals = num_subgoals
        self.discount = discount
        self.adj = adj
        self.state_history = []
        self.action_history = []
        self.reward_history = []
        self.budget_history = []
        self.remaining_history = []
        self.root_values = []
        self.child_visits = []
        self.duration_history = []
        self.learned_budget = False

    def observed_value(self, index, steps):
        stop = min(index + steps, len(self.reward_history))
        continuation = self.root_values[stop] if stop < len(self.reward_history) else 0.0
        for reward in reversed(self.reward_history[index:stop]):
            continuation = reward + self.discount * continuation
        return continuation


class ReplayBuffer:
    def __init__(self, capacity, batch_size):
        self.batch_size = batch_size
        self.buffer = deque(maxlen=capacity)

    def save_game(self, game):
        if game.action_history:
            self.buffer.append(game)

    def sample_batch(self):
        games = random.choices(self.buffer, k=self.batch_size)
        return [(g, random.randrange(len(g.action_history))) for g in games]

    def __len__(self):
        return len(self.buffer)

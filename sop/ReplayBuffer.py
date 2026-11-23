from collections import deque
from dataclasses import dataclass, field
import random


@dataclass
class Game:
    states: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    budgets: list = field(default_factory=list)
    remaining: list = field(default_factory=list)
    durations: list = field(default_factory=list)
    rewards: list = field(default_factory=list)
    policies: list = field(default_factory=list)
    values: list = field(default_factory=list)
    actor_enabled: list = field(default_factory=list)


class ReplayBuffer:
    def __init__(self, config):
        self.buffer = deque(maxlen=config['buffer_size'])
        self.batch_size = config['batch_size']

    def sample_batch(self):
        games = random.choices(self.buffer, k=self.batch_size)
        return [(game, random.randrange(len(game.actions))) for game in games]

    def __len__(self):
        return len(self.buffer)

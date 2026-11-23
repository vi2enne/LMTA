import csv
import os
import random
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from Env import Env
from Network import ActionNetwork, MuZeroHLNetworks
from ReplayBuffer import Game, ReplayBuffer
from Search import search, support_mean


def route_increments(env, route):
    dist, depot, last = env.dist, env.start_node, env.t == env.T
    path = [env.current_node, *route] + ([depot] if last else [])
    u, v = np.asarray(path[:-1], dtype=np.intp), np.asarray(path[1:], dtype=np.intp)
    inc = dist[u] + dist[v] - dist[u, v][:, None]
    if not last:
        inc = np.vstack([inc, dist[path[-1]][None]])
    pos = inc.argmin(0)
    add = inc[pos, np.arange(env.n)]
    return add / env.distance_scale, pos


def one_step_rewards(env, state, route, length, add):
    k = len(route)
    eta = env.eta * (k * k - max(k - 1, 0) ** 2)
    excess = np.maximum(length + add - env.max_daily_distance, 0) - max(length - env.max_daily_distance, 0)
    return np.where(state[:, 2] > 0, state[:, 2] - eta - env.distance_penalty * excess, 0).astype(np.float32)


class MuZeroAgent:
    def __init__(self, env, config, rank=0):
        self.env, self.config = env, config
        self.device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
        self.hl_networks = MuZeroHLNetworks(config, env).to(self.device)
        self.ll_action_net = ActionNetwork(config['node_dim'], config['embed_dim']).to(self.device)
        with torch.no_grad():
            self.ll_action_net.q_net[-1].weight.zero_()
            self.ll_action_net.q_net[-1].bias.zero_()
        self.hl_optimizer = torch.optim.Adam(self.hl_networks.parameters(), lr=config['lr_hl'], weight_decay=1e-5)
        self.replay_buffer = ReplayBuffer(config)
        self.ll_buffer = deque(maxlen=config['ll_buffer_capacity'])
        self.hl_learning_steps = 0
        self.collected_frames = 0

    def _get_hl_model(self):
        return self.hl_networks

    def tensor(self, x, dtype=None):
        return torch.as_tensor(np.asarray(x), dtype=dtype, device=self.device)

    def representation(self, state):
        return F.normalize(self.hl_networks.representation_net(state)[1], dim=-1)

    def dynamics_reward(self, h, z):
        return self.hl_networks.dynamics_net(h, z), self.hl_networks.reward_net(h, z)

    def budget_logits(self, h, z, remaining, goal_ids):
        cfg = self.config
        logits = self.hl_networks.budget_head(h, z)
        ids = torch.as_tensor(np.asarray(goal_ids), dtype=torch.long, device=self.device).reshape(-1)
        rem = remaining.to(torch.float32).reshape(-1)
        share = torch.full_like(rem, cfg['budget'] / cfg['T'])
        mu = torch.minimum(self.hl_networks.budget_head.option_multipliers[ids] * share, rem)
        sigma = (cfg['option_sigma_frac'] * share).clamp_min(cfg['option_sigma_min'])
        support = torch.arange(logits.shape[1], device=self.device, dtype=torch.float32)
        prior = -(support[None, :] - mu[:, None]).square() / (2 * sigma[:, None].square())
        logits = prior + cfg['option_adjust_nats'] * torch.tanh(logits)
        illegal = torch.arange(logits.shape[1], device=self.device)[None, :] > remaining[:, None]
        return logits.masked_fill(illegal, -1e9)

    @torch.no_grad()
    def play_tours(self, envs, states, budgets, training):
        count = len(envs)
        tours = [[] for _ in envs]
        work = [s.copy() for s in states]
        previous = np.zeros(count)
        macro_returns = np.zeros(count)
        lengths = [float(e.dist[e.current_node, e.start_node]) / e.distance_scale if e.t == e.T else 0.0
                   for e in envs]
        geo = [route_increments(e, []) for e in envs]
        onestep = [one_step_rewards(e, w, [], l, g[0]) for e, w, l, g in zip(envs, work, lengths, geo)]
        for w, g in zip(work, geo):
            w[:, 3] = g[0]
        for step in range(int(max(budgets, default=0))):
            active = [i for i in range(count) if step < budgets[i] and np.any(work[i][:, 2] > 0)]
            if not active:
                break
            visited = self.tensor([work[i][:, 2] for i in active], torch.float32) <= 0
            scores = self.tensor(np.stack([onestep[i] for i in active]), torch.float32)
            selected = dict(zip(active, scores.masked_fill(visited, -torch.inf).argmax(1).tolist()))
            for i in active:
                city, env = selected[i], envs[i]
                tours[i].insert(int(geo[i][1][city]), city)
                lengths[i] += float(geo[i][0][city])
                value = env.reward_by_simulation(states[i], env.get_context(), tours[i])
                reward = value - previous[i]
                previous[i] = value
                macro_returns[i] += self.config['discount'] ** step * reward
                next_state = work[i].copy()
                next_state[city, 2] = 0
                next_state[:, 6] = (env.remaining_budget - step - 1) / env.budget
                geo[i] = route_increments(env, tours[i])
                onestep[i] = one_step_rewards(env, next_state, tours[i], lengths[i], geo[i][0])
                next_state[:, 3] = geo[i][0]
                if training:
                    self.ll_buffer.append((work[i], next_state))
                work[i] = next_state
        return tours, macro_returns

    @torch.no_grad()
    def rollout(self, envs, epoch=None):
        self.hl_networks.eval()
        cfg = self.config
        training = epoch is not None
        n = len(envs)
        games = [Game() for e in envs]
        totals = np.zeros(n)
        warmup = training and epoch < cfg['budget_policy_warmup_epochs']
        for day in range(envs[0].T):
            states = [e.get_state() for e in envs]
            h = self.representation(self.tensor(states, torch.float32))
            policy, value = search(self, h, envs[0].T - day, training)
            if training:
                probs = policy ** (1 / cfg['search_temperature'])
                probs /= probs.sum(1, keepdims=True)
                goals = np.array([np.random.choice(len(p), p=p) for p in probs])
            else:
                goals = policy.argmax(1)
            remaining = np.array([e.remaining_budget for e in envs])
            z = self.hl_networks.subgoal_embeddings(self.tensor(goals))
            logits = self.budget_logits(h, z, self.tensor(remaining), goals)
            if training:
                budgets = torch.distributions.Categorical(logits=logits).sample().cpu().numpy()
            else:
                probs = logits.double().softmax(1).cpu().numpy()
                budgets = np.array([e.policy_rng.choice(len(p), p=p / p.sum())
                                    for e, p in zip(envs, probs)])
            tours, macro = self.play_tours(envs, states, budgets, training)
            for i, env in enumerate(envs):
                _, reward, _ = env.step(tours[i])
                totals[i] += reward
                if training:
                    game = games[i]
                    game.states.append(states[i])
                    game.actions.append(goals[i])
                    game.budgets.append(budgets[i])
                    game.remaining.append(remaining[i])
                    game.durations.append(len(tours[i]))
                    game.rewards.append(float(macro[i] if tours[i] else reward))
                    game.policies.append(policy[i])
                    game.values.append(value[i])
                    game.actor_enabled.append(not warmup)
                    self.collected_frames += len(tours[i])
        if training:
            self.replay_buffer.buffer.extend(games)
        return totals.tolist()

    def sample_instances(self, count, split, first=0, epoch=None):
        cfg = self.config
        size = cfg['num_nodes_train'] if split in (0, 1) else cfg['num_nodes_test']
        rewards = []
        for start in range(0, count, cfg['rollout_batch_size']):
            envs = [Env.from_seed(size, cfg['budget'], cfg['T'], cfg['seed'], split, first + i)
                    for i in range(start, min(count, start + cfg['rollout_batch_size']))]
            rewards.extend(self.rollout(envs, epoch))
        return rewards

    def update_hl_weights(self):
        cfg = self.config
        model = self.hl_networks.train()
        batch = self.replay_buffer.sample_batch()
        root = self.representation(self.tensor([g.states[i] for g, i in batch], torch.float32))
        first_ids = self.tensor([g.actions[i] for g, i in batch], torch.long)
        first_z = model.subgoal_embeddings(first_ids)
        duration = self.tensor([g.durations[i] for g, i in batch])
        dist = torch.linalg.vector_norm(root - model.dynamics_net(root, first_z), dim=1)
        margin = model.margin_predictor(first_z, cfg['slowness_margin_min'], cfg['slowness_margin_max'])
        score = dist + cfg['hl_xi'] * margin
        if len(score) > 1:
            a, b = torch.triu_indices(len(score), len(score), offset=1, device=self.device)
            sign = (duration[a] - duration[b]).sign()
            rank = F.relu(1 - sign * (score[a] - score[b])).mean()
        else:
            rank = score.new_tensor(1.)
        margin_l2 = sum(p.square().sum() for p in model.margin_predictor.parameters())
        loss = (cfg['hl_push_weight'] * F.relu(margin - dist).mean() +
                cfg['order_loss_weight'] * rank + cfg['hl_margin_l2'] * margin_l2)
        if self.ll_buffer:
            micro = random.choices(self.ll_buffer, k=cfg['batch_size'])
            states = [row[0] for row in micro] + [row[1] for row in micro]
            before, after = self.representation(self.tensor(states, torch.float32)).chunk(2)
            delta = (before - after).norm(dim=-1)
            loss = loss + cfg['pull_loss_weight'] * F.relu(delta - cfg['slowness_margin']).square().mean()

        if self.replay_buffer.buffer[-1].actor_enabled[0]:
            recent = list(self.replay_buffer.buffer)[-cfg['games_per_epoch']:]
            draws = cfg['batch_size'] * min(cfg['num_unroll_steps'], cfg['T'])
            samples = [(g, random.randrange(len(g.actions))) for g in random.choices(recent, k=draws)]
            with torch.no_grad():
                sub_h = self.representation(self.tensor([g.states[i] for g, i in samples], torch.float32))
                sub_z = model.subgoal_embeddings(self.tensor([g.actions[i] for g, i in samples]))
            rem = self.tensor([g.remaining[i] for g, i in samples])
            taken = self.tensor([g.budgets[i] for g, i in samples])
            dist_b = torch.distributions.Categorical(logits=self.budget_logits(
                sub_h, sub_z, rem, [g.actions[i] for g, i in samples]))
            critic = model.budget_critic(torch.cat((sub_h, sub_z), dim=-1)).squeeze(-1)
            returns = self.tensor([g.rewards[i] + cfg['discount'] *
                                   (g.values[i+1] if i+1 < len(g.values) else 0.)
                                   for g, i in samples], torch.float32)
            advantage = returns - critic
            loss = loss + (advantage.square() - dist_b.log_prob(taken) * advantage.detach()
                           - cfg['budget_entropy_beta'] * dist_b.entropy()).mean()

        h = root
        unroll = cfg['num_unroll_steps']
        def value_target(g, i):
            end = min(i + cfg['td_steps'], len(g.rewards))
            value = g.values[end] if end < len(g.rewards) else 0.
            for r in reversed(g.rewards[i:end]):
                value = r + cfg['discount'] * value
            return value
        pred_loss = root.new_zeros(())
        consistency = root.new_zeros(())
        for depth in range(unroll + 1):
            valid = [(row, g, i + depth) for row, (g, i) in enumerate(batch)
                     if i + depth <= len(g.actions)]
            if not valid:
                break
            rows = self.tensor([row for row, _, _ in valid])
            latent = h.index_select(0, rows)
            p, v = model.prediction_net(latent)
            target_v = self.tensor([value_target(g, i) for _, g, i in valid], torch.float32)
            pred_loss += F.mse_loss(support_mean(v), target_v, reduction='sum')
            nonterminal = [j for j, (_, g, i) in enumerate(valid) if i < len(g.actions)]
            if not nonterminal:
                break
            idx = self.tensor(nonterminal)
            current = [valid[j] for j in nonterminal]
            target_p = self.tensor([g.policies[i] for _, g, i in current], torch.float32)
            pred_loss += F.cross_entropy(p[idx], target_p, reduction='sum')
            if depth == unroll:
                break
            z = model.subgoal_embeddings(self.tensor([g.actions[i] for _, g, i in current]))
            next_h, reward = self.dynamics_reward(latent[idx], z)
            target_r = self.tensor([g.rewards[i] for _, g, i in current], torch.float32)
            pred_loss += F.mse_loss(reward.squeeze(-1), target_r, reduction='sum')
            real = [j for j, (_, g, i) in enumerate(current) if i + 1 < len(g.states)]
            if real:
                with torch.no_grad():
                    target = self.representation(self.tensor([current[j][1].states[current[j][2] + 1]
                                                              for j in real], torch.float32))
                consistency += (1 - F.cosine_similarity(next_h[real], target, dim=-1)).sum()
            h = h.index_copy(0, rows[idx], next_h)
        loss = loss + (pred_loss + cfg['consistency_weight'] * consistency) / len(batch)
        self.hl_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        budget_params = list(model.budget_head.parameters()) + list(model.budget_critic.parameters())
        budget_ids = {id(p) for p in budget_params}
        torch.nn.utils.clip_grad_norm_(budget_params, 5, error_if_nonfinite=True)
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if id(p) not in budget_ids], 5,
                                       error_if_nonfinite=True)
        self.hl_optimizer.step()
        self.hl_learning_steps += 1
        return loss.item()

    def train(self, job_name):
        cfg = self.config
        output = Path(cfg['output_dir'])
        metrics = output / 'learning.csv'
        best = -np.inf
        with metrics.open('x', newline='') as stream:
            log = csv.writer(stream)
            log.writerow(['epoch', 'frames', 'selfplay_reward', 'validation_reward', 'hl_updates', 'seconds'])
            for epoch in range(cfg['num_epochs']):
                start = time.monotonic()
                train = self.sample_instances(cfg['games_per_epoch'], 0, epoch * cfg['games_per_epoch'], epoch)
                hl_losses = []
                for _ in range(cfg['train_per_epoch']):
                    if len(self.replay_buffer) >= cfg['batch_size']:
                        hl_losses.append(self.update_hl_weights())
                val = None
                if (epoch + 1) % cfg['validation_interval'] == 0 or epoch + 1 == cfg['num_epochs']:
                    val = float(np.mean(self.sample_instances(cfg['validation_games'], 1)))
                    if val > best:
                        best = val
                        self.save(output / 'best.pth')
                elapsed = time.monotonic() - start
                log.writerow([epoch + 1, self.collected_frames, np.mean(train), val, self.hl_learning_steps,
                              round(elapsed, 2)])
                stream.flush()
                print(f"epoch={epoch+1} frames={self.collected_frames} selfplay={np.mean(train):.3f} "
                      f"validation={val} hl_steps={self.hl_learning_steps} "
                      f"hl_loss={np.mean(hl_losses) if hl_losses else None} seconds={elapsed:.2f}", flush=True)
        if not self.hl_learning_steps:
            raise RuntimeError('Training finished without high-level updates. Check replay and batch sizes.')
        self.save(output / 'final.pth')

    def save(self, path):
        tmp = path.with_suffix('.tmp')
        torch.save(self.state_dict(), tmp)
        os.replace(tmp, path)

    def state_dict(self):
        return dict(hl_networks=self.hl_networks.state_dict(), ll_action_net=self.ll_action_net.state_dict())

    def load_state_dict(self, state):
        self.hl_networks.load_state_dict(state['hl_networks'])
        self.ll_action_net.load_state_dict(state['ll_action_net'])

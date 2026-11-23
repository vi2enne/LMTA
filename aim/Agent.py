import os
import time
import random
import platform
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from Env import Env
from Network import (
    RepresentationNetwork, PredictionNetwork, DynamicsNetwork, RewardNetwork,
    BudgetHead, LowLevelPolicyNetwork, ActionEmbedder, MI_Critic_Close, MI_Critic_Far, SubgoalMarginPredictor
)
from ReplayBuffer import BellmanTarget, ReplayBuffer, ReplayBuffer_sub, Game


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent(nn.Module):
    def __init__(self, env, config):
        super(Agent, self).__init__()
        self.env = env
        self.config = dict(config)
        config = self.config

        self.representation_net = RepresentationNetwork(embed_dim=config['embed_dim']).to(device)
        self.prediction_net = PredictionNetwork(
            embed_dim=config['embed_dim'], action_space_size=config['num_subgoals'], value_scale=config['n']
        ).to(device)
        self.dynamics_net = DynamicsNetwork(embed_dim=config['embed_dim']).to(device)
        self.reward_net = RewardNetwork(embed_dim=config['embed_dim'], reward_scale=config['budget']).to(device)
        self.budget_head = BudgetHead(
            embed_dim=config['embed_dim'], max_budget=env.budget,
            option_multipliers=torch.linspace(0.0, config['option_max_multiplier'], config['num_subgoals'])
        ).to(device)
        self.budget_critic = nn.Sequential(nn.Linear(2 * config["embed_dim"], 256),
                                           nn.ReLU(), nn.Linear(256, 1)).to(device)
        self.subgoal_embeddings = nn.Embedding(config['num_subgoals'], config['embed_dim']).to(device)

        self.action_embedder = ActionEmbedder(
            max_nodes=config['n'],
            output_dim=config['action_embedding_dim'],
            hidden_dim=config['mi_critic_hidden_dim'] // 2
        ).to(device)
        self.mi_critic_close = MI_Critic_Close(
            latent_dim=config['embed_dim'],
            subgoal_embed_dim=config['embed_dim'],
            action_embedding_dim=config['action_embedding_dim'],
            hidden_dim=config['mi_critic_hidden_dim']
        ).to(device)
        self.mi_critic_far = MI_Critic_Far(
            latent_dim=config['embed_dim'],
            hidden_dim=config['mi_critic_hidden_dim']
        ).to(device)
        self.margin_predictor = SubgoalMarginPredictor(
            embed_dim=config['embed_dim'],
            hidden_dim=config['margin_predictor_hidden_dim'],
            dropout_rate=config['margin_predictor_dropout']
        ).to(device)

        self.ll_action_net = LowLevelPolicyNetwork(
            embed_dim=config['embed_dim'], subgoal_embed_dim=config['embed_dim'], reach_hops=config['ll_reach_hops']
        ).to(device)
        self.ll_action_net_target = LowLevelPolicyNetwork(
            embed_dim=config['embed_dim'], subgoal_embed_dim=config['embed_dim'], reach_hops=config['ll_reach_hops']
        ).to(device)
        self.ll_action_net_target.load_state_dict(self.ll_action_net.state_dict())

        self.hl_params = (
            list(self.representation_net.parameters()) +
            list(self.prediction_net.parameters()) +
            list(self.dynamics_net.parameters()) +
            list(self.reward_net.parameters()) +
            list(self.budget_head.parameters()) +
            list(self.budget_critic.parameters()) +
            list(self.subgoal_embeddings.parameters()) +
            list(self.action_embedder.parameters()) +
            list(self.mi_critic_close.parameters()) +
            list(self.mi_critic_far.parameters()) +
            list(self.margin_predictor.parameters())
        )
        self.hl_optimizer = torch.optim.Adam(self.hl_params, lr=config['lr_hl'], weight_decay=1e-5)
        self.ll_optimizer = torch.optim.Adam(self.ll_action_net.parameters(), lr=config['lr_ll'], weight_decay=1e-5)

        self.hl_replay_buffer = ReplayBuffer(capacity=config['hl_buffer_capacity'], batch_size=config['batch_size'])
        self.ll_replay_buffer = ReplayBuffer_sub(config['ll_buffer_capacity'])

        self.ll_epsilon = config['ll_epsilon_start']
        self.ll_learning_steps = 0
        self.true_frames_cum = 0

    def context(self, remaining, days, capacity=None):
        capacity = self.config['budget'] if capacity is None else capacity
        return remaining / max(1, capacity), days / self.config['T']

    def budget_raw_logits(self, latent, goals, goal_ids, remaining, days):
        logits = self.budget_head(latent, goals)
        dev = logits.device
        ids = torch.as_tensor(goal_ids, dtype=torch.long, device=dev).reshape(-1)
        rem = torch.as_tensor(remaining, dtype=torch.float32, device=dev).reshape(-1)
        days = torch.as_tensor(days, dtype=torch.float32, device=dev).reshape(-1).clamp_min(1.0)
        share = rem / days
        mu = torch.minimum(self.budget_head.option_multipliers[ids] * share, rem)
        sigma = torch.clamp(self.config['option_sigma_frac'] * share, min=self.config['option_sigma_min'])
        support = torch.arange(logits.shape[-1], device=dev, dtype=torch.float32)
        prior = -(support[None, :] - mu[:, None]).square() / (2 * sigma[:, None].square())
        return prior + self.config['option_adjust_nats'] * torch.tanh(logits)

    def train_model(self, num_epochs=50, games_per_epoch=16, train_steps_per_epoch=32, validation_games=50):
        os.makedirs("results", exist_ok=True)
        best_test_reward = -float('inf')
        best_epoch = -1
        job_name = getattr(self, "_job_name", "aim")
        csv_path = f"results/{job_name}.csv"
        with open(csv_path, "w") as f:
            f.write("frames,reward\n")
        self._print_setup_info()
        train_wall_start = time.time()
        total_used_seeds = 0
        hl_updates_total = 0

        for epoch in range(num_epochs):
            self.train()
            epoch_start_time = time.time()
            warmup = epoch < self.config['warmup_epochs']
            training_graphs = [
                nx.erdos_renyi_graph(
                    self.config['n'],
                    p=min(1.0, 5.0 / self.config['n']),
                    directed=True,
                    seed=self.episode_seed('train', epoch * games_per_epoch + i)
                )
                for i in range(games_per_epoch)
            ]
            epoch_rewards, used_seeds_this_epoch = self.collect_games(training_graphs, warmup)
            total_used_seeds += int(used_seeds_this_epoch)

            hl_losses, ll_losses = [], []
            for step in range(train_steps_per_epoch):
                hl_loss = self.update_hl_weights()
                if hl_loss is not None:
                    hl_updates_total += 1
                    hl_losses.append(hl_loss)
                ll_loss = self.update_ll_weights()
                if ll_loss:
                    ll_losses.append(ll_loss)

            test_rewards = self.evaluate(num_episodes=validation_games, split='validation')
            avg_test_reward = float(np.mean(test_rewards))
            self.true_frames_cum += used_seeds_this_epoch
            with open(csv_path, "a") as f:
                f.write(f"{self.true_frames_cum},{avg_test_reward:.6f}\n")

            if avg_test_reward > best_test_reward:
                best_test_reward = avg_test_reward
                best_epoch = epoch + 1
                if hasattr(self, '_job_name'):
                    os.makedirs("models", exist_ok=True)
                    self.save_weights(f"models/{self._job_name}_best_model.pth")

            print(f"epoch={epoch + 1} self_play={np.mean(epoch_rewards):.2f} validation={avg_test_reward:.2f} "
                  f"best={best_test_reward:.2f}@{best_epoch} "
                  f"hl_loss={np.mean([d['total'] for d in hl_losses]) if hl_losses else float('nan'):.4f} "
                  f"ll_loss={np.mean(ll_losses) if ll_losses else float('nan'):.4f} "
                  f"seconds={time.time() - epoch_start_time:.1f}", flush=True)

        if hasattr(self, '_job_name'):
            os.makedirs("models", exist_ok=True)
            self.save_weights(f"models/{self._job_name}_final_model.pth")
        print(f"Training seconds={time.time() - train_wall_start:.2f} frames={total_used_seeds} "
              f"HL_updates={hl_updates_total} LL_updates={self.ll_learning_steps} "
              f"best_epoch={best_epoch} best_validation={best_test_reward:.2f}", flush=True)
        if hl_updates_total == 0 or self.ll_learning_steps == 0:
            raise RuntimeError("Training completed without both HL and LL optimizer updates")

    def save_weights(self, path):
        torch.save(self.state_dict(), path + ".tmp")
        os.replace(path + ".tmp", path)

    def _print_setup_info(self):
        print(f"OS: {platform.system()} {platform.release()} | Python: {platform.python_version()} | "
              f"PyTorch: {torch.__version__} | device: {device}", flush=True)
        print(f"Seed: {self.config.get('seed')} | n={self.config['n']} | T={self.env.T} | Budget={self.env.budget} | "
              f"num_simulations={self.config['num_simulations']}", flush=True)

    @torch.no_grad()
    def collect_games(self, graphs, warmup):
        from aim_search import search
        envs = [Env(g, budget=self.config['budget'], T=self.config['T']) for g in graphs]
        games = [Game(self.config['num_subgoals'], self.config['discount'], env.A) for env in envs]
        for g in games:
            g.learned_budget = not warmup
        episode_rewards = np.zeros(len(games))
        pending = [None] * len(games)
        active = list(range(len(games)))
        all_adjs = torch.as_tensor(np.stack([e.A for e in envs]), device=device)
        for t in range(self.config['T']):
            if not active:
                break
            env_batch = [envs[i] for i in active]
            state = torch.as_tensor(np.stack([env.state for env in env_batch]), device=device)
            adj = all_adjs[active]
            days = self.config['T'] - t
            context = [self.context(e.budget_r, days, e.budget) for e in env_batch]
            h = self.representation_net(state, adj, context)
            visits, root_values = search(self, h, self.config['T'] - t, training=True)
            pi = visits / visits.sum(axis=1, keepdims=True)
            z_ids = [np.random.choice(self.config['num_subgoals'], p=row) for row in pi]
            z = self.subgoal_embeddings(torch.as_tensor(z_ids, dtype=torch.long, device=device))
            remaining = [env.budget_r for env in env_batch]
            logits = self.budget_raw_logits(h, z, z_ids, remaining, [days] * len(remaining))
            legal = torch.arange(logits.shape[-1], device=device)[None, :] <= torch.tensor(remaining, device=device)[:, None]
            budgets = Categorical(logits=logits.masked_fill(~legal, -torch.inf)).sample().tolist()
            actions = self.select_seed_batch(state, adj, z, budgets, self.ll_epsilon)
            next_active = []
            for j, idx in enumerate(active):
                env, game, action = envs[idx], games[idx], actions[j]
                next_state, raw_reward, done, _ = env.step(env.state, action)
                episode_rewards[idx] += raw_reward
                game.state_history.append(env.state)
                game.action_history.append(z_ids[j])
                game.budget_history.append(budgets[j])
                game.remaining_history.append(env.budget_r)
                game.duration_history.append(len(action))
                game.child_visits.append(pi[j])
                game.root_values.append(float(root_values[j]))
                macro = sum(self.config['discount'] ** step * r for step, r in enumerate(env.primitive_rewards))
                game.reward_history.append(macro if action else raw_reward)
                pending[idx] = self.store_ll_experience(env.state, action, z_ids[j], env.primitive_rewards,
                                                        next_state, days, env, pending[idx], raw_reward)
                env.state = next_state
                env.budget_r -= len(action)
                if not done:
                    next_active.append(idx)
            active = next_active
        for g in games:
            self.hl_replay_buffer.save_game(g)
        return episode_rewards.tolist(), sum(sum(g.duration_history) for g in games)

    @torch.no_grad()
    def select_seed_batch(self, states, adjs, goals, budgets, epsilon=0.0):
        states = states.clone()
        actions = [[] for _ in budgets]
        counts = states[:, :, 0].sum(-1).to(torch.long).tolist()
        limits = [min(k, int(n)) for k, n in zip(budgets, counts)]
        for step in range(max(limits, default=0)):
            rows = [j for j, limit in enumerate(limits) if step < limit]
            s = states[rows]
            legal = s[:, :, 0].bool()
            q = self.ll_action_net(s, adjs[rows], goals[rows])[:, :-1]
            chosen = q.masked_fill(~legal, -torch.inf).argmax(1)
            if epsilon:
                explore = torch.rand(len(rows), device=device) < epsilon
                random_nodes = torch.multinomial(legal.float(), 1).squeeze(1)
                chosen = torch.where(explore, random_nodes, chosen)
            states[rows, chosen, 0] = 0
            states[rows, chosen, 1] = 1
            for row, act in zip(rows, chosen.tolist()):
                actions[row].append(act)
        return actions

    def update_hl_weights(self):
        batch_size = self.config['batch_size']
        if len(self.hl_replay_buffer) < batch_size or len(self.ll_replay_buffer) < batch_size:
            return None
        samples = self.hl_replay_buffer.sample_batch()
        def tensor(data, dtype=torch.float32):
            return torch.as_tensor(np.asarray(data), dtype=dtype, device=device)

        latent = self.representation_net(
            tensor([g.state_history[i] for g, i in samples]), tensor([g.adj for g, _ in samples]),
            [self.context(g.remaining_history[i], self.config['T'] - i) for g, i in samples]
        )
        z = self.subgoal_embeddings(tensor([g.action_history[i] for g, i in samples], torch.long))
        latent_next = self.dynamics_net(latent, z)
        distance = (latent - latent_next).norm(dim=-1)
        margin = self.margin_predictor(z, self.config['slowness_margin_min'], self.config['slowness_margin_max'])
        score = distance + self.config['hl_xi'] * margin
        duration = tensor([g.duration_history[i] for g, i in samples])
        perm = torch.randperm(batch_size, device=device)
        label = (duration - duration[perm]).sign()
        order = F.relu(1 - label * (score - score[perm])).mean()
        push = (margin - distance).relu().mean()

        primitive = self.ll_replay_buffer.sample(batch_size)
        adj = tensor([r.adj for r in primitive])
        before = self.representation_net(tensor([r.state for r in primitive]), adj, [r.context for r in primitive])
        after = self.representation_net(tensor([r.next_state for r in primitive]), adj, [r.next_context for r in primitive])
        cap = ((before - after).norm(dim=-1) - self.config['slowness_margin']).relu().square().mean()

        loss = (self.config['hl_rank_weight'] * order + self.config['hl_push_weight'] * push
                + self.config['slowness_loss_coeff'] * cap
                + self.config['hl_margin_param_l2'] * sum(p.square().sum() for p in self.margin_predictor.parameters()))

        budget_loss = latent.new_zeros(())
        recent = list(self.hl_replay_buffer.buffer)[-self.config['games_per_epoch']:]
        g_samples = []
        if recent[-1].learned_budget:
            draws = batch_size * min(self.config['num_unroll_steps'], self.config['T'])
            g_samples = [(g, random.randrange(len(g.action_history))) for g in random.choices(recent, k=draws)]
            with torch.no_grad():
                budget_h = self.representation_net(
                    tensor([g.state_history[i] for g, i in g_samples]), tensor([g.adj for g, _ in g_samples]),
                    [self.context(g.remaining_history[i], self.config['T'] - i) for g, i in g_samples]
                )
                budget_z = self.subgoal_embeddings(tensor([g.action_history[i] for g, i in g_samples], torch.long))
        if g_samples:
            logits = self.budget_raw_logits(budget_h, budget_z, [g.action_history[i] for g, i in g_samples],
                                            [g.remaining_history[i] for g, i in g_samples],
                                            [self.config['T'] - i for _, i in g_samples])
            remaining = tensor([g.remaining_history[i] for g, i in g_samples], torch.long)
            legal = torch.arange(logits.size(-1), device=device)[None, :] <= remaining[:, None]
            dist = Categorical(logits=logits.masked_fill(~legal, torch.finfo(logits.dtype).min))
            taken = tensor([g.budget_history[i] for g, i in g_samples], torch.long)
            target = tensor([g.reward_history[i] + self.config['discount'] *
                             (g.root_values[i + 1] if i + 1 < len(g.root_values) else 0)
                             for g, i in g_samples])
            critic = self.budget_critic(torch.cat([budget_h, budget_z], dim=-1)).squeeze(-1)
            advantage = target - critic
            budget_loss = (advantage.square() - dist.log_prob(taken) * advantage.detach()
                           - self.config['budget_entropy_beta'] * dist.entropy()).mean()
            loss = loss + budget_loss

        v_total, p_total, r_total, c_total = (latent.new_zeros(()) for _ in range(4))
        active_samples = samples
        h = latent
        for depth in range(self.config['num_unroll_steps'] + 1):
            logits, value_logits = self.prediction_net(h)
            values = value_logits.squeeze(-1)
            scale = self.prediction_net.value_scale
            v_targets = tensor([g.observed_value(i, self.config['td_steps']) for g, i in active_samples])
            v_total = v_total + F.mse_loss(values / scale, v_targets / scale, reduction='sum') / batch_size
            continuing = [j for j, (g, i) in enumerate(active_samples) if i < len(g.action_history)]
            if not continuing:
                break
            current = [active_samples[j] for j in continuing]
            policies = tensor([g.child_visits[i] for g, i in current])
            p_total = p_total - (policies * logits[continuing].log_softmax(-1)).sum() / batch_size
            if depth == self.config['num_unroll_steps']:
                break
            z_u = self.subgoal_embeddings(tensor([g.action_history[i] for g, i in current], torch.long))
            h_u = h[continuing]
            reward_logits = self.reward_net(h_u, z_u)
            scale = self.reward_net.reward_scale
            r_scalar = tensor([g.reward_history[i] for g, i in current])
            r_total = r_total + F.mse_loss(reward_logits.squeeze(-1) / scale, r_scalar / scale,
                                           reduction='sum') / batch_size
            h = self.dynamics_net(h_u, z_u)
            active_samples = [(g, i + 1) for g, i in current]
            real = [j for j, (g, i) in enumerate(active_samples) if i < len(g.state_history)]
            if real:
                with torch.no_grad():
                    target = self.representation_net(
                        tensor([active_samples[j][0].state_history[active_samples[j][1]] for j in real]),
                        tensor([active_samples[j][0].adj for j in real]),
                        [self.context(active_samples[j][0].remaining_history[active_samples[j][1]],
                                      self.config['T'] - active_samples[j][1]) for j in real])
                c_total = c_total + (1 - F.cosine_similarity(h[real], target, dim=-1)).sum() / batch_size
        loss = loss + p_total + v_total + r_total + self.config['consistency_weight'] * c_total
        self.hl_optimizer.zero_grad()
        loss.backward()
        budget_params = list(self.budget_head.parameters()) + list(self.budget_critic.parameters())
        budget_ids = {id(p) for p in budget_params}
        torch.nn.utils.clip_grad_norm_(budget_params, 5.0, error_if_nonfinite=True)
        torch.nn.utils.clip_grad_norm_([p for p in self.hl_params if id(p) not in budget_ids], 5.0,
                                       error_if_nonfinite=True)
        self.hl_optimizer.step()
        return dict(total=loss.item(), policy=p_total.item(), value=v_total.item(), reward=r_total.item(),
                    budget=budget_loss.item(), hl_hinge=push.item(), hl_rank=order.item(), slow_close=cap.item(),
                    consistency=c_total.item())

    def update_ll_weights(self):
        if len(self.ll_replay_buffer) < self.config['batch_size']:
            return None
        batch = self.ll_replay_buffer.sample(self.config['batch_size'])

        def tensor(data, dtype=torch.float32):
            return torch.as_tensor(np.asarray(data), dtype=dtype, device=device)
        goals = self.subgoal_embeddings(tensor([t.option for t in batch], torch.long)).detach()
        adjacency = tensor([t.adj for t in batch])
        q_preds = self.ll_action_net(tensor([t.state for t in batch]), adjacency, goals)
        q_preds_for_actions = q_preds.gather(1, tensor([t.action for t in batch], torch.long)[:, None]).squeeze(1)
        targets = [t.bellman for t in batch]
        with torch.no_grad():
            target_q = tensor([target.reward for target in targets])
            continuing = [i for i, target in enumerate(targets) if target.next_state is not None]
            if continuing:
                link = [targets[i] for i in continuing]
                s_next = tensor([target.next_state for target in link])
                next_goals = self.subgoal_embeddings(tensor([target.next_goal for target in link], torch.long))
                q_next = self.ll_action_net_target(s_next, adjacency[continuing], next_goals)[:, :-1]
                legal = s_next[:, :, 0].bool()
                best = q_next.masked_fill(~legal, -torch.inf).amax(1)
                target_q[continuing] += tensor([target.next_discount for target in link]) * best

        loss = F.mse_loss(q_preds_for_actions, target_q)
        self.ll_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.ll_action_net.parameters(), max_norm=5.0, error_if_nonfinite=True)
        self.ll_optimizer.step()
        self.ll_learning_steps += 1
        if self.ll_learning_steps % self.config['ll_target_update_freq'] == 0:
            self.ll_action_net_target.load_state_dict(self.ll_action_net.state_dict())
        self.ll_epsilon = max(self.config['ll_epsilon_min'], self.ll_epsilon * self.config['ll_epsilon_decay'])
        return loss.item()

    def store_ll_experience(self, state, action, subgoal_id, rewards, next_state, days, env, pending, raw_reward):
        gamma = self.config['discount']
        if not action:
            if pending is not None:
                pending.passive_day(raw_reward, gamma)
            elif raw_reward:
                raise ValueError("Passive gain before the first seed")
            return pending
        temp = state
        for i, seed in enumerate(action):
            done = i == len(action) - 1
            if done:
                following = next_state
            else:
                following = temp.copy()
                env.set_state_index(following, [seed], 1)
            if pending is not None:
                pending.link(temp, subgoal_id)
            pending = target = BellmanTarget(float(rewards[i]), gamma)
            self.ll_replay_buffer.push(temp, env.A, subgoal_id, seed, following,
                                       self.context(env.budget_r - i, days, env.budget),
                                       self.context(env.budget_r - i - 1, days - int(done), env.budget), target)
            temp = following
        return pending

    def episode_seed(self, split, index, stream='graph'):
        namespace = {'train': 0, 'validation': 1, 'heldout': self.config.get('test_namespace', 2)}[split]
        key = {'graph': (), 'cascade': (0,), 'budget': (1,)}[stream]
        seq = np.random.SeedSequence([int(self.config['seed']), namespace, int(index)], spawn_key=key)
        return int(seq.generate_state(1, dtype=np.uint64)[0])

    @torch.no_grad()
    def evaluate(self, num_episodes=50, split='validation'):
        from aim_search import evaluate_games
        previous_mode = self.training
        self.eval()
        try:
            return evaluate_games(self, num_episodes, split)
        finally:
            self.train(previous_mode)

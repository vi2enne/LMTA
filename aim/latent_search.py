import numpy as np
import torch


def support_mean(logits):
    if logits.ndim == 1:
        return logits
    if logits.shape[-1] == 1:
        return logits.squeeze(-1)
    radius = logits.shape[-1] // 2
    support = torch.arange(-radius, radius + 1, device=logits.device, dtype=logits.dtype)
    return logits.softmax(-1) @ support


@torch.no_grad()
def search_network(model, cfg, hidden, horizon, training):
    if horizon <= 0 or cfg['num_simulations'] <= 0:
        raise ValueError("search requires positive horizon and simulation count")
    def index(array):
        return torch.as_tensor(array, dtype=torch.long, device=hidden.device)
    batch, dim = hidden.shape
    rows = np.arange(batch)
    sims, nz = cfg['num_simulations'], cfg['num_subgoals']
    discount = cfg['discount']
    shape = (batch, sims + 1, nz)
    priors = np.zeros(shape, np.float32)
    visits = np.zeros(shape, np.int32)
    returns = np.zeros(shape, np.float64)
    children = np.full(shape, -1, np.int32)
    rewards = np.zeros((batch, sims + 1), np.float32)
    hidden_bank = hidden.new_empty(batch, sims + 1, dim)
    hidden_bank[:, 0] = hidden
    mix = float(cfg['search_prior_mix'])
    p0, _ = model.prediction_net(hidden)
    priors[:, 0] = p0.softmax(-1).cpu().numpy()
    priors[:, 0] = (1 - mix) * priors[:, 0] + mix / nz
    decode_reward = getattr(model, "decode_reward", support_mean)
    root_sum = np.zeros(batch, dtype=np.float64)
    if training:
        frac = cfg['root_exploration_fraction']
        noise = np.random.dirichlet([cfg['root_dirichlet_alpha']] * nz, size=batch)
        priors[:, 0] = (1 - frac) * priors[:, 0] + frac * noise

    low = np.full(batch, np.inf)
    high = np.full(batch, -np.inf)
    for sim in range(1, sims + 1):
        nodes = np.zeros(batch, dtype=np.intp)
        active = rows
        path = []
        frontier = []
        leaf_value = np.zeros(batch, dtype=np.float64)
        for depth in range(horizon):
            parents = nodes[active]
            v = visits[active, parents]
            n = 1 + v.sum(1, keepdims=True) + (parents[:, None] != 0)
            q = returns[active, parents] / np.maximum(v, 1)
            norm = high[active] > low[active]
            if norm.any():
                q[norm] = ((q[norm] - low[active[norm], None]) /
                           (high[active[norm], None] - low[active[norm], None]))
            total = v.sum(1)
            sibling = np.where(total > 0, (q * v).sum(1) / np.maximum(total, 1), 0.0)
            q = np.where(v > 0, q, sibling[:, None])
            c = cfg['pb_c_init'] + np.log1p(n / cfg['pb_c_base'])
            score = q + priors[active, parents] * c * np.sqrt(n) / (1 + v)
            actions = nz - 1 - score[:, ::-1].argmax(1)
            next_nodes = children[active, parents, actions]
            new = next_nodes < 0
            if new.any():
                r, par, act = active[new], parents[new], actions[new]
                next_nodes[new] = sim
                children[r, par, act] = sim
                frontier.append((r, par, act, depth + 1 < horizon))
            path.append((active, parents, actions, next_nodes))
            nodes[active] = next_nodes
            active = active[~new]
            if not len(active):
                break

        if frontier:
            r = np.concatenate([x[0] for x in frontier])
            par = np.concatenate([x[1] for x in frontier])
            act = np.concatenate([x[2] for x in frontier])
            nonterminal = np.concatenate([np.full(len(x[0]), x[3]) for x in frontier])
            parent = hidden_bank[index(r), index(par)]
            emb = model.subgoal_embeddings(index(act))
            next_hidden = model.dynamics_net(parent, emb)
            rew = model.reward_net(parent, emb)
            policy, value = model.prediction_net(next_hidden)
            prediction = torch.cat((policy.softmax(-1), decode_reward(rew)[:, None],
                                    support_mean(value)[:, None]), -1).cpu().numpy()
            hidden_bank[index(r), sim] = next_hidden
            priors[r, sim] = (1 - mix) * prediction[:, :nz] + mix / nz
            rewards[r, sim] = prediction[:, nz]
            leaf_value[r] = np.where(nonterminal, prediction[:, nz + 1], 0)

        for r, par, act, child in reversed(path):
            leaf_value[r] = rewards[r, child] + discount * leaf_value[r]
            returns[r, par, act] += leaf_value[r]
            visits[r, par, act] += 1
            q = returns[r, par, act] / visits[r, par, act]
            low[r] = np.minimum(low[r], q)
            high[r] = np.maximum(high[r], q)
        root_sum += leaf_value

    return visits[:, 0].copy(), root_sum / sims

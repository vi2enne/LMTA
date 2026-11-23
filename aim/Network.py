import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class GNNLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x, adj):
        if x.dim() == 2:
            x = x.unsqueeze(0)
        batch_size, n_nodes, _ = x.size()

        adj = adj.to(x.device)
        if adj.dim() == 2:
            adj_b = adj.unsqueeze(0).expand(batch_size, adj.size(0), adj.size(1))
        elif adj.dim() == 3:
            adj_b = adj
        else:
            raise ValueError("GNNLayer adj shape error: adj must be 2D or 3D.")

        if adj_b.shape[1] != n_nodes or adj_b.shape[2] != n_nodes:
            adj_b = adj_b[:, :n_nodes, :n_nodes]

        deg = adj_b.sum(dim=-1, keepdim=True).clamp_min(1.0)
        adj_norm = adj_b / deg

        neighbors = torch.bmm(adj_norm, x)
        aggregated = x + neighbors
        out = F.relu(self.linear(aggregated))
        return out

class RepresentationNetwork(nn.Module):
    def __init__(self, node_feature_size=3, embed_dim=64):
        super(RepresentationNetwork, self).__init__()
        self.gnn1 = GNNLayer(node_feature_size, embed_dim)
        self.gnn2 = GNNLayer(embed_dim, embed_dim)
        self.fc_latent = nn.Linear(embed_dim, embed_dim)
        self.fc_context = nn.Linear(2, embed_dim, bias=False)
        self.ln = nn.LayerNorm(embed_dim)

    def forward(self, state, adj, context):
        if state.dim() == 2:
            state = state.unsqueeze(0)
        x = F.relu(self.gnn1(state, adj))
        x = F.relu(self.gnn2(x, adj))
        graph_embed = x.mean(dim=1)
        s_latent = self.fc_latent(graph_embed)
        context = torch.as_tensor(context, dtype=state.dtype, device=state.device).reshape(-1, 2)
        if context.shape[0] != state.shape[0]:
            raise ValueError("one resource context is required per graph")
        s_latent = s_latent + self.fc_context(context)
        s_latent = self.ln(s_latent)
        s_latent = F.normalize(s_latent, p=2, dim=1)
        return s_latent

class PredictionNetwork(nn.Module):
    def __init__(self, embed_dim, action_space_size, value_scale, hidden_dim=256):
        super().__init__()
        self.register_buffer("value_scale", torch.tensor(float(value_scale)))
        self.policy_head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, action_space_size)
        )
        self.value_head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, latent_state):
        value = self.value_head(latent_state) * self.value_scale
        return self.policy_head(latent_state), value

class DynamicsNetwork(nn.Module):
    def __init__(self, embed_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim)
        )
    def forward(self, latent_state, subgoal_embedding):
        x = torch.cat([latent_state, subgoal_embedding], dim=-1)
        next_latent_state = self.net(x)
        return F.normalize(next_latent_state, p=2, dim=1)

class RewardNetwork(nn.Module):
    def __init__(self, embed_dim, reward_scale, hidden_dim=256):
        super().__init__()
        self.register_buffer("reward_scale", torch.tensor(float(reward_scale)))
        self.net = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, latent_state, subgoal_embedding):
        x = torch.cat([latent_state, subgoal_embedding], dim=-1)
        return self.net(x) * self.reward_scale

class BudgetHead(nn.Module):
    def __init__(self, embed_dim, max_budget, option_multipliers, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, max_budget + 1)
        )
        self.register_buffer("option_multipliers", torch.as_tensor(option_multipliers, dtype=torch.float32))
        with torch.no_grad():
            self.net[-1].weight.zero_()
            self.net[-1].bias.zero_()
    def forward(self, latent_state, subgoal_embedding):
        x = torch.cat([latent_state, subgoal_embedding], dim=-1)
        return self.net(x)

class ActionEmbedder(nn.Module):
    def __init__(self, max_nodes, output_dim, hidden_dim=64):
        super().__init__()
        self.max_nodes = max_nodes
        self.fc1 = nn.Linear(max_nodes, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, actions_binary_vectors):
        emb = F.relu(self.fc1(actions_binary_vectors.float()))
        emb = F.relu(self.fc2(emb))
        return emb

class MI_Critic_Close(nn.Module):
    def __init__(self, latent_dim, subgoal_embed_dim, action_embedding_dim, hidden_dim):
        super().__init__()
        input_size = 2 * latent_dim + subgoal_embed_dim + action_embedding_dim
        self.fc1 = nn.Linear(input_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, phi_s_ll_t, phi_s_ll_tplus1, z_hl_vector_active, a_ll_action_embedded):
        combined = torch.cat([phi_s_ll_t, phi_s_ll_tplus1, z_hl_vector_active, a_ll_action_embedded], dim=1)
        x = F.relu(self.fc1(combined))
        x = F.relu(self.fc2(x))
        scalar_output = self.fc3(x)
        return scalar_output

class MI_Critic_Far(nn.Module):
    def __init__(self, latent_dim, hidden_dim):
        super().__init__()
        input_size = 2 * latent_dim
        self.fc1 = nn.Linear(input_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, phi_s_hl_t, phi_g_s_hl_z_hl):
        combined = torch.cat([phi_s_hl_t, phi_g_s_hl_z_hl], dim=1)
        x = F.relu(self.fc1(combined))
        x = F.relu(self.fc2(x))
        scalar_output = self.fc3(x)
        return scalar_output

class SubgoalMarginPredictor(nn.Module):
    def __init__(self, embed_dim, hidden_dim=None, dropout_rate=0.1):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = embed_dim // 2

        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, subgoal_embeddings, min_margin=0.1, max_margin=2.0):
        raw_margins = self.net(subgoal_embeddings).squeeze(-1)
        margins = min_margin + (max_margin - min_margin) * torch.sigmoid(raw_margins)
        return margins

class LowLevelPolicyNetwork(nn.Module):
    def __init__(self, node_feature_size=3, embed_dim=64, subgoal_embed_dim=64, reach_hops=8):
        super(LowLevelPolicyNetwork, self).__init__()
        self.node_feature_size = node_feature_size
        self.embed_dim = embed_dim
        self.subgoal_embed_dim = subgoal_embed_dim
        self.q1 = nn.Linear(self.node_feature_size, self.embed_dim)
        self.q2 = nn.Linear(self.embed_dim, self.embed_dim)
        self.q3 = nn.Linear(self.embed_dim * 2, self.embed_dim)
        self.two_hop_weight = nn.Parameter(torch.zeros(self.embed_dim))
        self.reach_weight = nn.Parameter(torch.zeros(reach_hops, self.embed_dim))
        self.fc_a = nn.Sequential(
            nn.Linear(self.embed_dim + self.subgoal_embed_dim, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.fc_p = nn.Sequential(
            nn.Linear(self.embed_dim + self.subgoal_embed_dim, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, xv, adj, subgoal_embedding, mask=None):
        unbatched = xv.ndim == 2
        if unbatched:
            xv = xv.unsqueeze(0)
        hv = F.relu(self.q1(xv))
        mu1 = F.relu(self.q2(torch.matmul(adj, hv)))
        pre = self.q3(torch.cat((hv, mu1), -1))
        inactive = xv[..., :1]
        pre = pre + torch.matmul(adj, inactive * torch.matmul(adj, inactive)) * self.two_hop_weight
        v = inactive
        for h in range(self.reach_weight.shape[0]):
            v = torch.matmul(adj, inactive * v) if h else torch.matmul(adj, inactive)
            pre = pre + torch.log1p(v) * self.reach_weight[h]
        mu = F.relu(pre)
        goals = subgoal_embedding.reshape(xv.size(0), 1, -1)
        node_input = torch.cat((mu, goals.expand(-1, xv.size(1), -1)), -1)
        null_input = torch.cat((mu.mean(1), goals.squeeze(1)), -1)
        q = torch.cat((self.fc_a(node_input).squeeze(-1), self.fc_p(null_input)), -1)
        if mask is not None:
            q = q.masked_fill(mask, -torch.inf)
        return q.squeeze(0) if unbatched else q

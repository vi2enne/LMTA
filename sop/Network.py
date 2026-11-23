import torch
import torch.nn as nn
import torch.nn.functional as F
from graph_encoder import GraphAttentionEncoder


class RepresentationNetwork(nn.Module):
    def __init__(self, node_dim, embed_dim):
        super().__init__()
        self.first = nn.Linear(node_dim, embed_dim)
        self.second = nn.Linear(embed_dim, embed_dim)
        self.projection = nn.Linear(embed_dim, embed_dim)

    def forward(self, state):
        features = state
        for layer in (self.first, self.second):
            neighbors = (features.sum(dim=1, keepdim=True) - features) / max(features.shape[1]-1, 1)
            features = F.relu(layer(features + neighbors))
        return features, F.normalize(self.projection(features.mean(dim=1)), dim=-1)


class PredictionNetwork(nn.Module):
    def __init__(self, embed_dim, action_space_size, hidden_dim=128):
        super().__init__()
        self.policy_head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_space_size)
        )
        self.value_head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, latent_state):
        policy_logits = self.policy_head(latent_state)
        value_support = self.value_head(latent_state)
        return policy_logits, value_support


class DynamicsNetwork(nn.Module):
    def __init__(self, embed_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim)
        )

    def forward(self, latent_state, subgoal_embedding):
        x = torch.cat([latent_state, subgoal_embedding], dim=-1)
        next_latent_state = self.net(x)
        next_latent_state = F.normalize(next_latent_state, p=2, dim=1)
        return next_latent_state


class RewardNetwork(nn.Module):
    def __init__(self, embed_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, latent_state, subgoal_embedding):
        x = torch.cat([latent_state, subgoal_embedding], dim=-1)
        return self.net(x)


class BudgetHead(nn.Module):
    def __init__(self, embed_dim, max_budget, option_multipliers, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, max_budget + 1)
        )
        self.register_buffer('option_multipliers', torch.as_tensor(option_multipliers, dtype=torch.float32))
        with torch.no_grad():
            self.net[-1].weight.zero_()
            self.net[-1].bias.zero_()

    def forward(self, latent_state, subgoal_embedding):
        x = torch.cat([latent_state, subgoal_embedding], dim=-1)
        return self.net(x)


class ActionNetwork(nn.Module):
    def __init__(self, node_dim, embed_dim, context_dim=2, hidden_dim=128):
        super().__init__()
        self.node_dim = node_dim
        self.embed_dim = embed_dim

        self.encoder = GraphAttentionEncoder(
            n_heads=4,
            embed_dim=embed_dim,
            n_layers=2,
            node_dim=node_dim
        )

        q_net_input_dim = embed_dim + (embed_dim * context_dim) + embed_dim
        self.q_net = nn.Sequential(
            nn.Linear(q_net_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        q_null_net_input_dim = embed_dim + (embed_dim * context_dim) + embed_dim
        self.q_null_net = nn.Sequential(
            nn.Linear(q_null_net_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state, context, subgoal_embedding, mask=None):
        batch_size = state.shape[0]
        num_nodes = state.shape[1]

        node_embeddings, graph_embedding = self.encoder(state)

        context_indices = context.long().unsqueeze(-1).expand(-1, -1, self.embed_dim)
        context_embedding = torch.gather(node_embeddings, 1, context_indices)
        context_embedding_flat = context_embedding.view(batch_size, -1)

        subgoal_emb_expanded = subgoal_embedding.unsqueeze(1).expand(-1, num_nodes, -1)
        context_emb_expanded = context_embedding_flat.unsqueeze(1).expand(-1, num_nodes, -1)

        q_input = torch.cat([node_embeddings, context_emb_expanded, subgoal_emb_expanded], dim=-1)
        q_null_input = torch.cat([graph_embedding, context_embedding_flat, subgoal_embedding], dim=-1)
        q_values = self.q_net(q_input)

        q_value_null = self.q_null_net(q_null_input).unsqueeze(1)

        q_values = torch.cat([q_values, q_value_null], dim=1)

        if mask is not None:
            q_values[mask] = -float('inf')

        return q_values.squeeze(-1)


class MI_Critic_Close(nn.Module):
    def __init__(self, latent_dim, subgoal_embed_dim, hidden_dim):
        super().__init__()
        input_size = latent_dim + latent_dim + subgoal_embed_dim + latent_dim
        self.fc1 = nn.Linear(input_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, phi_s_ll_t, phi_s_ll_tplus1, z_hl_vector_active, a_ll_node_embedded):
        combined = torch.cat([phi_s_ll_t, phi_s_ll_tplus1, z_hl_vector_active, a_ll_node_embedded], dim=1)
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
        hidden_dim = hidden_dim or embed_dim // 2
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


class MuZeroHLNetworks(nn.Module):
    def __init__(self, config, env):
        super().__init__()
        self.config = config
        self.representation_net = RepresentationNetwork(config['node_dim'], config['embed_dim'])
        self.prediction_net = PredictionNetwork(embed_dim=config['embed_dim'], action_space_size=config['action_space_size'])
        self.dynamics_net = DynamicsNetwork(embed_dim=config['embed_dim'])
        self.reward_net = RewardNetwork(embed_dim=config['embed_dim'])
        self.budget_head = BudgetHead(
            embed_dim=config['embed_dim'], max_budget=env.budget,
            option_multipliers=torch.linspace(0.0, config['option_max_multiplier'], config['num_subgoals']))
        self.budget_critic = nn.Sequential(
            nn.Linear(2 * config['embed_dim'], 128), nn.ReLU(), nn.Linear(128, 1)
        )
        self.subgoal_embeddings = nn.Embedding(config['num_subgoals'], config['embed_dim'])

        self.mi_critic_close = MI_Critic_Close(
            latent_dim=config['embed_dim'],
            subgoal_embed_dim=config['embed_dim'],
            hidden_dim=config['mi_critic_hidden_dim']
        )
        self.mi_critic_far = MI_Critic_Far(
            latent_dim=config['embed_dim'],
            hidden_dim=config['mi_critic_hidden_dim']
        )
        self.margin_predictor = SubgoalMarginPredictor(
            embed_dim=config['embed_dim'],
            hidden_dim=config.get('margin_predictor_hidden_dim', config['embed_dim'] // 2),
            dropout_rate=config.get('margin_predictor_dropout', 0.1)
        )

    def decode_reward(self, logits):
        return logits.squeeze(-1)

# File name : model.py
# Author : Ted Song (original) / extended for CS30707 project

import copy
import os
import math
import random
from collections import deque

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


def _default_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SnakeEncoder(nn.Module):
    """Conv encoder mapping (grid, dir_onehot) -> z in R^{z_dim}.

    Mirrors the CURL DMControl encoder (Appendix A): 4 conv layers + MLP +
    LayerNorm + tanh. The 4-d direction one-hot is concatenated into the MLP
    so spatial features and the agent's heading stay coupled.

    in_channels: 3 for a single-frame board, 3*k for a k-frame stack (placeholder).
    """

    def __init__(self, in_channels: int = 3, board_size: int = 20,
                 z_dim: int = 50, num_filters: int = 32,
                 num_conv_layers: int = 4, dir_dim: int = 4,
                 hidden_units: int = 256):
        super().__init__()
        self.in_channels = in_channels
        self.z_dim = z_dim
        self.dir_dim = dir_dim

        layers = [nn.Conv2d(in_channels, num_filters, kernel_size=3, stride=1, padding=1),
                  nn.ReLU(inplace=True)]
        for _ in range(num_conv_layers - 1):
            layers += [nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1),
                       nn.ReLU(inplace=True)]
        self.conv = nn.Sequential(*layers)

        conv_out = num_filters * board_size * board_size
        self.mlp = nn.Sequential(
            nn.Linear(conv_out + dir_dim, hidden_units),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_units, z_dim),
        )
        self.ln = nn.LayerNorm(z_dim)

    def forward(self, grid: torch.Tensor, dir_onehot: torch.Tensor) -> torch.Tensor:
        if grid.dim() == 3:
            grid = grid.unsqueeze(0)
            dir_onehot = dir_onehot.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False
        h = self.conv(grid)
        h = h.flatten(start_dim=1)
        h = torch.cat([h, dir_onehot], dim=1)
        z = self.mlp(h)
        z = self.ln(z)
        z = torch.tanh(z)
        if squeeze:
            z = z.squeeze(0)
        return z


class DQN:
    """Original windowed-state DQN (input 85, output 3). Kept for baseline."""

    def __init__(self, episode, input_size, output_size, device=None):
        self.device = device or _default_device()
        self.epsilon_start = 0.95
        self.epsilon_end = 1e-20
        self.epsilon_decay = episode
        self.gamma = 0.95
        self.lr = 1e-3
        self.batch_size = 256
        self.input_size = input_size
        self.output_size = output_size
        self.model = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_size),
        ).to(self.device)

        self.optimizer = optim.Adam(params=self.model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        self.steps_done = 0
        self.deque_size = int(1e5)
        self.epi_for_memory = deque(maxlen=self.deque_size)
        self.epsilon_threshold = 0.95

    def print_eps(self):
        return self.epsilon_threshold

    def decay_epsilon(self):
        self.epsilon_threshold = self.epsilon_end + (
            self.epsilon_start - self.epsilon_end
        ) * np.exp(-1.0 * self.steps_done / self.epsilon_decay)
        self.steps_done += 8

    def memorize(self, state, action, reward, next_state):
        if len(self.epi_for_memory) >= self.deque_size:
            for _ in range(len(self.epi_for_memory) - self.deque_size):
                self.epi_for_memory.popleft()
        self.epi_for_memory.append(
            (
                state.detach().cpu(),
                action.detach().cpu() if torch.is_tensor(action) else torch.LongTensor([[int(action)]]),
                torch.FloatTensor([reward]),
                next_state.detach().cpu(),
            )
        )

    def select_action(self, state):
        if torch.rand(1)[0] > self.epsilon_threshold:
            with torch.no_grad():
                s = state.to(self.device)
                return torch.argmax(self.model(s).data).view(1, 1).cpu()
        else:
            return torch.LongTensor([[random.randrange(self.output_size)]])

    def optimize_model(self, load_data=False):
        if len(self.epi_for_memory) < self.batch_size:
            return
        if load_data:
            batch = random.sample(
                list(np.array(pd.read_pickle(f"{os.getcwd()}/human_data.pkl"))),
                self.batch_size,
            )
        else:
            batch = random.sample(self.epi_for_memory, self.batch_size)
        states, actions, rewards, next_states = zip(*batch)

        states = torch.cat(states).reshape(self.batch_size, self.input_size).to(self.device)
        next_states = torch.cat(next_states).reshape(self.batch_size, self.input_size).to(self.device)
        actions = torch.cat(actions).to(self.device)
        rewards = torch.cat(rewards).to(self.device)

        current_q = self.model(states).gather(1, actions)
        max_next_q = self.model(next_states).detach().max(1)[0]
        expected_q = rewards + (self.gamma * max_next_q)

        self.optimizer.zero_grad()
        loss = self.criterion(current_q.squeeze(), expected_q)
        loss.backward()
        self.optimizer.step()

    def save(self, name):
        save_dir = os.path.join(os.getcwd(), "save")
        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(save_dir, f"{name}.pt"))

    def load(self, name):
        path = os.path.join(os.getcwd(), "save", f"{name}.pt")
        self.model.load_state_dict(torch.load(path, map_location=self.device))


class ZDQN:
    """DQN that takes raw (grid, dir) -> encoder -> z -> Q-head.

    Reward gradients flow into the encoder so the embedding becomes
    task-relevant. The encoder typically starts from a contrastive checkpoint.
    """

    def __init__(self, episode, z_dim: int, output_size: int = 3,
                 encoder=None, device=None,
                 lr_head: float = 1e-3, lr_encoder: float = 1e-4,
                 batch_size: int = 256, buffer_size: int = int(5e4),
                 finetune_encoder: bool = True,
                 epsilon_decay_mult: float = 3.0,
                 epsilon_start: float = 0.95):
        self.device = device or _default_device()
        assert encoder is not None, "ZDQN requires an encoder"
        self.encoder = encoder.to(self.device)
        self.finetune_encoder = finetune_encoder
        if not finetune_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
            self.encoder.eval()
        else:
            self.encoder.train()

        self.epsilon_start = epsilon_start
        self.epsilon_end = 1e-20
        # Multiply by epsilon_decay_mult (default 3) so the conv encoder
        # has 3x longer exploration window than the hand-crafted-state DQN.
        self.epsilon_decay = episode * epsilon_decay_mult
        self.gamma = 0.95
        self.batch_size = batch_size
        self.z_dim = z_dim
        self.output_size = output_size

        self.head = nn.Sequential(
            nn.Linear(z_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_size),
        ).to(self.device)

        param_groups = [{"params": self.head.parameters(), "lr": lr_head}]
        if finetune_encoder:
            param_groups.append({"params": self.encoder.parameters(), "lr": lr_encoder})
        self.optimizer = optim.Adam(param_groups)
        self.criterion = nn.MSELoss()
        self.steps_done = 0
        self.deque_size = buffer_size
        self.epi_for_memory = deque(maxlen=self.deque_size)
        self.epsilon_threshold = epsilon_start

        # Optional joint-CURL hooks; populated by attach_curl().
        self.curl = None
        self.curl_aug_fn = None
        self.curl_weight = 1.0

    def attach_curl(self, curl_module, aug_fn, curl_weight: float = 1.0,
                    lr_curl: float = 1e-3) -> None:
        """Wire a CURLModule into the optimizer for joint training.

        The query encoder inside `curl_module` must be the same object as
        `self.encoder` so gradients from both losses flow into one set of
        encoder weights (paper §4 "shared representations").
        """
        assert self.finetune_encoder, "Joint CURL requires a trainable encoder."
        assert curl_module.q_enc is self.encoder, (
            "CURLModule.q_enc must be the ZDQN encoder instance.")
        self.curl = curl_module.to(self.device)
        self.curl_aug_fn = aug_fn
        self.curl_weight = float(curl_weight)
        # Add the bilinear W (and any other curl-only params) to the optimizer.
        self.optimizer.add_param_group({"params": [self.curl.W], "lr": lr_curl})

    @torch.no_grad()
    def _embed_eval(self, grid, dir_onehot):
        was_training = self.encoder.training
        self.encoder.eval()
        g = grid.to(self.device)
        d = dir_onehot.to(self.device)
        if g.dim() == 3:
            g = g.unsqueeze(0)
            d = d.unsqueeze(0)
            z = self.encoder(g, d).squeeze(0)
        else:
            z = self.encoder(g, d)
        if was_training:
            self.encoder.train()
        return z

    def decay_epsilon(self):
        self.epsilon_threshold = self.epsilon_end + (
            self.epsilon_start - self.epsilon_end
        ) * np.exp(-1.0 * self.steps_done / self.epsilon_decay)
        self.steps_done += 8

    def memorize(self, grid, dir_onehot, action, reward, next_grid, next_dir, done):
        self.epi_for_memory.append(
            (
                grid.detach().cpu(),
                dir_onehot.detach().cpu(),
                action.detach().cpu() if torch.is_tensor(action) else torch.LongTensor([[int(action)]]),
                torch.FloatTensor([reward]),
                next_grid.detach().cpu(),
                next_dir.detach().cpu(),
                torch.tensor([bool(done)], dtype=torch.bool),
            )
        )

    def select_action(self, grid, dir_onehot):
        if torch.rand(1)[0] > self.epsilon_threshold:
            z = self._embed_eval(grid, dir_onehot)
            with torch.no_grad():
                if z.dim() == 1:
                    z = z.unsqueeze(0)
                return torch.argmax(self.head(z).data).view(1, 1).cpu()
        return torch.LongTensor([[random.randrange(self.output_size)]])

    def optimize_model(self):
        """One gradient step. Returns dict with 'q_loss' and (if attached) 'curl_loss'."""
        if len(self.epi_for_memory) < self.batch_size:
            return None
        batch = random.sample(self.epi_for_memory, self.batch_size)
        grids, dirs, actions, rewards, next_grids, next_dirs, dones = zip(*batch)

        grids = torch.stack(grids).to(self.device)
        dirs = torch.stack(dirs).to(self.device)
        next_grids = torch.stack(next_grids).to(self.device)
        next_dirs = torch.stack(next_dirs).to(self.device)
        actions = torch.cat(actions).to(self.device)
        rewards = torch.cat(rewards).to(self.device)
        dones = torch.cat(dones).to(self.device)

        if self.finetune_encoder:
            self.encoder.train()
        z = self.encoder(grids, dirs)
        current_q = self.head(z).gather(1, actions)

        with torch.no_grad():
            self.encoder.eval()
            z_next = self.encoder(next_grids, next_dirs)
            max_next_q = self.head(z_next).max(1)[0]
            max_next_q = max_next_q * (~dones).float()
            if self.finetune_encoder:
                self.encoder.train()

        expected_q = rewards + self.gamma * max_next_q
        q_loss = self.criterion(current_q.squeeze(), expected_q)

        curl_loss = None
        if self.curl is not None:
            # Two independent augmentations of the same minibatch (paper §4.7).
            x_q = self.curl_aug_fn(grids)
            x_k = self.curl_aug_fn(grids)
            z_q = self.curl.encode_query(x_q, dirs)
            z_k = self.curl.encode_key(x_k, dirs)
            curl_loss = self.curl.info_nce_loss(z_q, z_k)
            total = q_loss + self.curl_weight * curl_loss
        else:
            total = q_loss

        self.optimizer.zero_grad()
        total.backward()
        all_params = [p for g in self.optimizer.param_groups for p in g['params']]
        nn.utils.clip_grad_norm_(all_params, max_norm=10.0)
        self.optimizer.step()

        if self.curl is not None:
            self.curl.update_key_encoder()

        out = {"q_loss": float(q_loss.item())}
        if curl_loss is not None:
            out["curl_loss"] = float(curl_loss.item())
        return out

    def save(self, name):
        save_dir = os.path.join(os.getcwd(), "save")
        os.makedirs(save_dir, exist_ok=True)
        torch.save(
            {"head": self.head.state_dict(),
             "encoder": self.encoder.state_dict(),
             "z_dim": self.z_dim},
            os.path.join(save_dir, f"{name}.pt"),
        )

    def load(self, name):
        path = os.path.join(os.getcwd(), "save", f"{name}.pt")
        ckpt = torch.load(path, map_location=self.device)
        self.head.load_state_dict(ckpt["head"])
        self.encoder.load_state_dict(ckpt["encoder"])


class ZDQN2(ZDQN):
    """ZDQN + Target Network + Double DQN (van Hasselt et al., 2016).

    Two improvements over ZDQN:
    - Target network: Q-targets use a periodically-synced frozen copy of the
      online network, so the training target doesn't shift every step.
    - Double DQN: the online net selects the greedy action; the target net
      evaluates it. Decouples selection from evaluation, reducing Q overestimation.

    Default z_dim raised to 128 to match the larger representational demand.
    """

    def __init__(self, episode, z_dim: int = 128, output_size: int = 3,
                 encoder=None, device=None,
                 lr_head: float = 1e-3, lr_encoder: float = 1e-4,
                 batch_size: int = 256, buffer_size: int = int(5e4),
                 finetune_encoder: bool = True,
                 epsilon_decay_mult: float = 3.0,
                 epsilon_start: float = 0.95,
                 target_update_freq: int = 1000):
        super().__init__(episode, z_dim=z_dim, output_size=output_size,
                         encoder=encoder, device=device,
                         lr_head=lr_head, lr_encoder=lr_encoder,
                         batch_size=batch_size, buffer_size=buffer_size,
                         finetune_encoder=finetune_encoder,
                         epsilon_decay_mult=epsilon_decay_mult,
                         epsilon_start=epsilon_start)
        self.target_encoder = copy.deepcopy(self.encoder)
        self.target_head = copy.deepcopy(self.head)
        for p in self.target_encoder.parameters():
            p.requires_grad = False
        for p in self.target_head.parameters():
            p.requires_grad = False
        self.target_update_freq = target_update_freq
        self.update_steps = 0

    def _sync_target(self):
        self.target_encoder.load_state_dict(self.encoder.state_dict())
        self.target_head.load_state_dict(self.head.state_dict())

    def optimize_model(self):
        if len(self.epi_for_memory) < self.batch_size:
            return None
        batch = random.sample(self.epi_for_memory, self.batch_size)
        grids, dirs, actions, rewards, next_grids, next_dirs, dones = zip(*batch)

        grids = torch.stack(grids).to(self.device)
        dirs = torch.stack(dirs).to(self.device)
        next_grids = torch.stack(next_grids).to(self.device)
        next_dirs = torch.stack(next_dirs).to(self.device)
        actions = torch.cat(actions).to(self.device)
        rewards = torch.cat(rewards).to(self.device)
        dones = torch.cat(dones).to(self.device)

        if self.finetune_encoder:
            self.encoder.train()
        z = self.encoder(grids, dirs)
        current_q = self.head(z).gather(1, actions)

        with torch.no_grad():
            self.encoder.eval()
            # Double DQN: online net picks best action, target net evaluates it
            z_next_online = self.encoder(next_grids, next_dirs)
            best_actions = self.head(z_next_online).argmax(dim=1, keepdim=True)
            z_next_target = self.target_encoder(next_grids, next_dirs)
            next_q = self.target_head(z_next_target).gather(1, best_actions).squeeze(1)
            next_q = next_q * (~dones).float()
            if self.finetune_encoder:
                self.encoder.train()

        expected_q = rewards + self.gamma * next_q
        q_loss = self.criterion(current_q.squeeze(), expected_q)

        curl_loss = None
        if self.curl is not None:
            x_q = self.curl_aug_fn(grids)
            x_k = self.curl_aug_fn(grids)
            z_q = self.curl.encode_query(x_q, dirs)
            z_k = self.curl.encode_key(x_k, dirs)
            curl_loss = self.curl.info_nce_loss(z_q, z_k)
            total = q_loss + self.curl_weight * curl_loss
        else:
            total = q_loss

        self.optimizer.zero_grad()
        total.backward()
        all_params = [p for g in self.optimizer.param_groups for p in g['params']]
        nn.utils.clip_grad_norm_(all_params, max_norm=10.0)
        self.optimizer.step()

        if self.curl is not None:
            self.curl.update_key_encoder()

        self.update_steps += 1
        if self.update_steps % self.target_update_freq == 0:
            self._sync_target()

        out = {"q_loss": float(q_loss.item())}
        if curl_loss is not None:
            out["curl_loss"] = float(curl_loss.item())
        return out

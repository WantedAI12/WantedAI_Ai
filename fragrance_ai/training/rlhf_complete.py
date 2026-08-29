# fragrance_ai/training/rlhf_complete.py
"""
Complete RLHF Implementation with REINFORCE and PPO
Includes proper buffer management, GAE, and minibatch updates
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from collections import deque
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class Experience:
    """Single experience for replay buffer"""
    state: torch.Tensor
    action: int
    log_prob: float
    reward: float
    next_state: torch.Tensor
    done: bool
    value: Optional[float] = None


class RLBuffer:
    """Experience replay buffer for PPO"""

    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity

    def add(self, experience: Experience):
        """Add experience to buffer"""
        self.buffer.append(experience)

    def get_batch(self) -> Dict[str, torch.Tensor]:
        """Get all experiences as batch tensors"""
        if not self.buffer:
            return {}

        states = torch.cat([exp.state for exp in self.buffer])
        actions = torch.tensor([exp.action for exp in self.buffer], dtype=torch.long)
        log_probs = torch.tensor([exp.log_prob for exp in self.buffer], dtype=torch.float32)
        rewards = torch.tensor([exp.reward for exp in self.buffer], dtype=torch.float32)
        dones = torch.tensor([exp.done for exp in self.buffer], dtype=torch.float32)

        values = None
        if self.buffer[0].value is not None:
            values = torch.tensor([exp.value for exp in self.buffer], dtype=torch.float32)

        return {
            'states': states,
            'actions': actions,
            'log_probs': log_probs,
            'rewards': rewards,
            'dones': dones,
            'values': values
        }

    def clear(self):
        """Clear buffer"""
        self.buffer.clear()

    def __len__(self):
        return len(self.buffer)


# ============================================================================
# Neural Networks
# ============================================================================

class PolicyNetwork(nn.Module):
    """Actor network for action probability distribution"""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        logits = self.network(state)
        return torch.softmax(logits, dim=-1)


class ValueNetwork(nn.Module):
    """Critic network for state value estimation"""

    def __init__(self, state_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)


# ============================================================================
# REINFORCE Implementation (Minimal)
# ============================================================================

class REINFORCEAgent:
    """Minimal REINFORCE implementation"""

    def __init__(self, state_dim: int, action_dim: int, lr: float = 0.001):
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Policy network
        self.policy_net = PolicyNetwork(state_dim, action_dim)
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=lr)

        # Episode storage
        self.saved_log_probs = []
        self.saved_rewards = []
        self.last_state = None
        self.last_saved_actions = []

    def select_action(self, state: torch.Tensor) -> Tuple[int, float]:
        """Select action using policy network"""
        probs = self.policy_net(state)
        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.item(), log_prob

    def update_policy_with_feedback(
        self,
        chosen_id: str,
        options: List[Dict],
        state: Optional[torch.Tensor] = None,
        saved_actions: Optional[List] = None,
        rating: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Update policy based on user feedback (REINFORCE)

        Args:
            chosen_id: ID of chosen option
            options: List of presented options
            state: State tensor (uses last_state if None)
            saved_actions: Saved (action, log_prob) tuples
            rating: User rating (1-5)

        Returns:
            Training metrics
        """
        # Use saved state/actions if not provided
        if state is None:
            state = self.last_state
        if saved_actions is None:
            saved_actions = self.last_saved_actions

        # Calculate reward based on feedback
        if rating is not None:
            # Normalize rating: 1→-1, 3→0, 5→1
            reward = (rating - 3) / 2.0
        else:
            # Binary reward: chosen=+1, not chosen=0
            reward = 1.0 if any(opt["id"] == chosen_id for opt in options) else 0.0

        # Calculate REINFORCE loss
        loss = 0.0
        for action, log_prob in saved_actions:
            loss += -log_prob * reward

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)

        # Optimizer step
        self.optimizer.step()

        # Log the update in JSON format (Acceptance Criteria)
        import json
        logger.info(json.dumps({
            "event": "rl_update",
            "algorithm": "REINFORCE",
            "loss": round(loss.item(), 4),
            "reward": round(float(reward), 4),
            "entropy": 0.0,  # REINFORCE doesn't track entropy
            "clip_frac": 0.0  # N/A for REINFORCE
        }))

        return {
            "algorithm": "REINFORCE",
            "loss": float(loss.item()),
            "reward": float(reward),
            "entropy": 0.0,
            "clip_frac": 0.0,
            "chosen_id": chosen_id
        }


# ============================================================================
# PPO Implementation (Complete)
# ============================================================================

class PPOAgent:
    """Complete PPO implementation with Actor-Critic"""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        batch_size: int = 64,
        ppo_epochs: int = 4
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Hyperparameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.batch_size = batch_size
        self.ppo_epochs = ppo_epochs

        # Networks
        self.policy_net = PolicyNetwork(state_dim, action_dim)
        self.value_net = ValueNetwork(state_dim)

        # Optimizers
        self.optimizer = optim.Adam(
            list(self.policy_net.parameters()) + list(self.value_net.parameters()),
            lr=lr
        )

        # Buffer
        self.buffer = RLBuffer()

        # Tracking
        self.last_state = None
        self.last_saved_actions = []
        self.last_values = []

    def select_action(self, state: torch.Tensor) -> Tuple[int, float, float]:
        """Select action and estimate value"""
        with torch.no_grad():
            probs = self.policy_net(state)
            value = self.value_net(state)

        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.item(), log_prob.item(), value.item()

    def compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Generalized Advantage Estimation (GAE)

        Returns:
            advantages: GAE advantages
            returns: Discounted returns
        """
        batch_size = len(rewards)
        advantages = torch.zeros(batch_size)

        # Calculate GAE
        last_gae = 0
        for t in reversed(range(batch_size)):
            if t == batch_size - 1:
                next_value = 0
                next_done = 1
            else:
                next_value = values[t + 1]
                next_done = dones[t + 1]

            delta = rewards[t] + self.gamma * next_value * (1 - next_done) - values[t]
            advantages[t] = delta + self.gamma * self.gae_lambda * (1 - next_done) * last_gae
            last_gae = advantages[t]

        returns = advantages + values

        # Standardize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages, returns

    def update_policy_with_feedback(
        self,
        chosen_id: str,
        options: List[Dict],
        state: Optional[torch.Tensor] = None,
        saved_actions: Optional[List] = None,
        rating: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Update policy using PPO algorithm

        Args:
            chosen_id: ID of chosen option
            options: List of presented options
            state: State tensor
            saved_actions: Saved (action, log_prob) tuples
            rating: User rating (1-5)

        Returns:
            Training metrics
        """
        # Use saved state/actions if not provided
        if state is None:
            state = self.last_state
        if saved_actions is None:
            saved_actions = self.last_saved_actions

        # Calculate reward
        if rating is not None:
            reward = (rating - 3) / 2.0  # Normalize: 1→-1, 3→0, 5→1
        else:
            reward = 1.0 if any(opt["id"] == chosen_id for opt in options) else 0.0

        # Add experiences to buffer
        for i, (action, log_prob) in enumerate(saved_actions):
            # Calculate individual rewards (chosen option gets full reward)
            option_reward = reward if options[i]["id"] == chosen_id else reward * 0.1

            # Get value estimate if available
            value = self.last_values[i] if self.last_values else 0.5

            experience = Experience(
                state=state,
                action=action.item() if torch.is_tensor(action) else action,
                log_prob=log_prob.item() if torch.is_tensor(log_prob) else log_prob,
                reward=option_reward,
                next_state=state,  # Simplified: same state
                done=True,  # Single-step episodes
                value=value
            )
            self.buffer.add(experience)

        # Check if we have enough data for update
        if len(self.buffer) < self.batch_size:
            return {
                "algorithm": "PPO",
                "status": "buffering",
                "buffer_size": len(self.buffer),
                "batch_size": self.batch_size
            }

        # Get batch data
        batch = self.buffer.get_batch()
        states = batch['states']
        actions = batch['actions']
        old_log_probs = batch['log_probs']
        rewards = batch['rewards']
        dones = batch['dones']
        old_values = batch['values'] if batch['values'] is not None else torch.zeros_like(rewards)

        # Compute GAE
        advantages, returns = self.compute_gae(rewards, old_values, dones)

        # PPO update loop
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        total_clip_frac = 0

        for epoch in range(self.ppo_epochs):
            # Mini-batch updates
            for start_idx in range(0, len(states), self.batch_size):
                end_idx = min(start_idx + self.batch_size, len(states))

                # Get mini-batch
                mb_states = states[start_idx:end_idx]
                mb_actions = actions[start_idx:end_idx]
                mb_old_log_probs = old_log_probs[start_idx:end_idx]
                mb_advantages = advantages[start_idx:end_idx]
                mb_returns = returns[start_idx:end_idx]

                # Forward pass
                probs = self.policy_net(mb_states)
                values = self.value_net(mb_states).squeeze()

                # Calculate new log probs and entropy
                dist = Categorical(probs)
                new_log_probs = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()

                # PPO clipped objective
                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                clipped_ratio = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps)

                # Calculate clip_frac (fraction of ratios that were clipped)
                clip_frac = torch.mean((torch.abs(ratio - 1.0) > self.clip_eps).float())

                policy_loss = -torch.min(
                    ratio * mb_advantages,
                    clipped_ratio * mb_advantages
                ).mean()

                # Value loss (MSE)
                value_loss = nn.functional.mse_loss(values, mb_returns)

                # Total loss
                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    - self.entropy_coef * entropy
                )

                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    list(self.policy_net.parameters()) + list(self.value_net.parameters()),
                    self.max_grad_norm
                )

                # Optimizer step
                self.optimizer.step()

                # Track losses
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                total_clip_frac += clip_frac.item()

        # Clear buffer after update
        self.buffer.clear()

        # Calculate averages
        num_updates = self.ppo_epochs * (len(states) // self.batch_size + 1)
        avg_policy_loss = total_policy_loss / num_updates
        avg_value_loss = total_value_loss / num_updates
        avg_entropy = total_entropy / num_updates
        avg_clip_frac = total_clip_frac / num_updates

        # Log update in JSON format (Acceptance Criteria)
        import json
        logger.info(json.dumps({
            "event": "rl_update",
            "algorithm": "PPO",
            "loss": round(avg_policy_loss + avg_value_loss, 4),
            "policy_loss": round(avg_policy_loss, 4),
            "value_loss": round(avg_value_loss, 4),
            "reward": round(float(reward), 4),
            "entropy": round(avg_entropy, 4),
            "clip_frac": round(avg_clip_frac, 4)
        }))

        return {
            "algorithm": "PPO",
            "loss": avg_policy_loss + avg_value_loss,
            "policy_loss": avg_policy_loss,
            "value_loss": avg_value_loss,
            "entropy": avg_entropy,
            "clip_frac": avg_clip_frac,
            "reward": float(reward),
            "chosen_id": chosen_id
        }

    def generate_variations(
        self,
        state: torch.Tensor,
        num_options: int = 3
    ) -> Tuple[List[Dict], torch.Tensor, List[Tuple]]:
        """
        Generate variation options

        Returns:
            options: List of option dictionaries
            state: State tensor
            saved_actions: List of (action, log_prob) tuples
        """
        options = []
        saved_actions = []
        values = []

        with torch.no_grad():
            probs = self.policy_net(state)
            value = self.value_net(state)

        dist = Categorical(probs)

        for i in range(num_options):
            action = dist.sample()
            log_prob = dist.log_prob(action)

            options.append({
                "id": f"option_{i}",
                "action": action.item(),
                "log_prob": log_prob.item()
            })

            saved_actions.append((action, log_prob))
            values.append(value.item())

        # Save for later use
        self.last_state = state
        self.last_saved_actions = saved_actions
        self.last_values = values

        return options, state, saved_actions


# ============================================================================
# Unified RLHF Engine
# ============================================================================

class RLHFEngine:
    """Unified engine supporting both REINFORCE and PPO"""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        algorithm: str = "PPO",
        **kwargs
    ):
        """
        Initialize RLHF engine

        Args:
            state_dim: State dimension
            action_dim: Action dimension
            algorithm: "REINFORCE" or "PPO"
            **kwargs: Algorithm-specific parameters
        """
        self.algorithm = algorithm.upper()

        if self.algorithm == "REINFORCE":
            # Filter kwargs for REINFORCE (only lr is accepted)
            reinforce_kwargs = {k: v for k, v in kwargs.items() if k in ['lr']}
            self.agent = REINFORCEAgent(state_dim, action_dim, **reinforce_kwargs)
        elif self.algorithm == "PPO":
            self.agent = PPOAgent(state_dim, action_dim, **kwargs)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        logger.info(f"Initialized RLHF engine with {self.algorithm}")

    def update_policy_with_feedback(self, *args, **kwargs):
        """Delegate to specific algorithm"""
        return self.agent.update_policy_with_feedback(*args, **kwargs)

    def save_model(self, path: str):
        """Save model weights"""
        if self.algorithm == "PPO":
            torch.save({
                'policy_net': self.agent.policy_net.state_dict(),
                'value_net': self.agent.value_net.state_dict(),
                'optimizer': self.agent.optimizer.state_dict()
            }, path)
        else:
            torch.save({
                'policy_net': self.agent.policy_net.state_dict(),
                'optimizer': self.agent.optimizer.state_dict()
            }, path)
        logger.info(f"Model saved to {path}")

    def load_model(self, path: str):
        """Load model weights"""
        checkpoint = torch.load(path)
        self.agent.policy_net.load_state_dict(checkpoint['policy_net'])

        if self.algorithm == "PPO" and 'value_net' in checkpoint:
            self.agent.value_net.load_state_dict(checkpoint['value_net'])

        self.agent.optimizer.load_state_dict(checkpoint['optimizer'])
        logger.info(f"Model loaded from {path}")


# ============================================================================
# Export
# ============================================================================

__all__ = [
    'RLHFEngine',
    'REINFORCEAgent',
    'PPOAgent',
    'RLBuffer',
    'Experience'
]
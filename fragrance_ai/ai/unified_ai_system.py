"""
Unified AI System for Fragrance Generation
통합 AI 시스템 - 모든 AI 컴포넌트를 하나로 통합

Components:
- Deep Learning: UniversalFragranceGenerator
- MOGA Optimization: Multi-Objective Genetic Algorithm
- Reinforcement Learning: PPO/REINFORCE with RLHF
- Evolution Service: DNA-based fragrance evolution
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging
import sqlite3
import os
from datetime import datetime

# DEAP imports for MOGA
from deap import base, creator, tools, algorithms
import random

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration Classes
# ============================================================================

@dataclass
class UnifiedAIConfig:
    """통합 AI 시스템 설정"""
    # Deep Learning Configuration
    dl_embedding_dim: int = 768
    dl_num_layers: int = 6
    dl_num_heads: int = 8
    dl_dropout: float = 0.1
    dl_max_length: int = 100

    # MOGA Configuration
    moga_population_size: int = 100
    moga_generations: int = 50
    moga_mutation_rate: float = 0.1
    moga_crossover_rate: float = 0.7

    # RL Configuration
    rl_algorithm: str = "PPO"  # "PPO" or "REINFORCE"
    rl_state_dim: int = 20
    rl_action_dim: int = 12
    rl_learning_rate: float = 3e-4
    rl_gamma: float = 0.99
    rl_gae_lambda: float = 0.95
    rl_clip_epsilon: float = 0.2

    # System Configuration
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    verbose: bool = False


# ============================================================================
# 1. Deep Learning Component - Fragrance Generator
# ============================================================================

class FragranceGeneratorDL(nn.Module):
    """
    딥러닝 기반 향수 생성기
    Transformer 아키텍처를 사용한 조건부 생성 모델
    """

    def __init__(self, config: UnifiedAIConfig):
        super().__init__()
        self.config = config

        # Note embedding layers
        self.note_embedding = nn.Embedding(1000, config.dl_embedding_dim)  # 1000 different notes
        self.concentration_embedding = nn.Embedding(100, config.dl_embedding_dim)  # 100 concentration levels
        self.position_embedding = nn.Embedding(config.dl_max_length, config.dl_embedding_dim)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.dl_embedding_dim,
            nhead=config.dl_num_heads,
            dim_feedforward=config.dl_embedding_dim * 4,
            dropout=config.dl_dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, config.dl_num_layers)

        # Output heads
        self.note_predictor = nn.Linear(config.dl_embedding_dim, 1000)
        self.concentration_predictor = nn.Linear(config.dl_embedding_dim, 100)

        # Layer normalization
        self.layer_norm = nn.LayerNorm(config.dl_embedding_dim)

    def forward(self, note_ids: torch.Tensor, conc_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass

        Args:
            note_ids: Note IDs tensor (batch, seq_len)
            conc_ids: Concentration IDs tensor (batch, seq_len)

        Returns:
            Dictionary with predictions
        """
        batch_size, seq_len = note_ids.shape

        # Create embeddings
        note_emb = self.note_embedding(note_ids)
        conc_emb = self.concentration_embedding(conc_ids)

        # Position embeddings
        pos_ids = torch.arange(seq_len, device=note_ids.device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.position_embedding(pos_ids)

        # Combine embeddings
        combined = note_emb + conc_emb + pos_emb

        # Transformer processing
        transformed = self.transformer(combined)
        transformed = self.layer_norm(transformed)

        # Predictions
        note_logits = self.note_predictor(transformed)
        conc_logits = self.concentration_predictor(transformed)

        return {
            'note_logits': note_logits,
            'concentration_logits': conc_logits,
            'hidden_states': transformed
        }

    def generate(self, seed_notes: Optional[torch.Tensor] = None, max_length: int = 20) -> Dict[str, torch.Tensor]:
        """
        Generate a new fragrance formula

        Args:
            seed_notes: Optional starting notes
            max_length: Maximum formula length

        Returns:
            Generated formula dictionary
        """
        if seed_notes is None:
            # Start with a random note
            seed_notes = torch.randint(1, 100, (1, 1), device=self.config.device)
            seed_concs = torch.randint(1, 20, (1, 1), device=self.config.device)
        else:
            seed_concs = torch.randint(1, 20, seed_notes.shape, device=self.config.device)

        generated_notes = seed_notes
        generated_concs = seed_concs

        for _ in range(max_length - seed_notes.size(1)):
            with torch.no_grad():
                output = self.forward(generated_notes, generated_concs)

                # Sample next note
                note_probs = F.softmax(output['note_logits'][:, -1, :], dim=-1)
                next_note = torch.multinomial(note_probs, 1)

                # Sample next concentration
                conc_probs = F.softmax(output['concentration_logits'][:, -1, :], dim=-1)
                next_conc = torch.multinomial(conc_probs, 1)

                # Append to sequence
                generated_notes = torch.cat([generated_notes, next_note], dim=1)
                generated_concs = torch.cat([generated_concs, next_conc], dim=1)

                # Stop if we generate an END token (note_id = 0)
                if next_note.item() == 0:
                    break

        return {
            'notes': generated_notes,
            'concentrations': generated_concs
        }


# ============================================================================
# 2. MOGA Component - Multi-Objective Optimization
# ============================================================================

class MOGAOptimizer:
    """
    Multi-Objective Genetic Algorithm for fragrance optimization
    다목적 유전 알고리즘 최적화
    """

    def __init__(self, config: UnifiedAIConfig):
        self.config = config
        self.setup_deap()
        self.setup_database()

    def setup_deap(self):
        """Setup DEAP framework for MOGA"""
        # Create fitness and individual classes
        if not hasattr(creator, "FitnessMulti"):
            creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0, 1.0))  # Quality, Cost, Stability
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMulti)

        # Setup toolbox
        self.toolbox = base.Toolbox()

        # Gene: concentration of each ingredient (0-100)
        self.toolbox.register("attr_float", random.uniform, 0, 100)

        # Individual: list of 20 ingredient concentrations
        self.toolbox.register("individual", tools.initRepeat, creator.Individual,
                            self.toolbox.attr_float, n=20)

        # Population
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        # Genetic operators
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=10, indpb=self.config.moga_mutation_rate)
        self.toolbox.register("select", tools.selNSGA2)

        # Evaluation function
        self.toolbox.register("evaluate", self.evaluate_fragrance)

    def setup_database(self):
        """Setup ingredient database"""
        package_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        db_path = os.path.join(package_root, "data", "moga_ingredients.db")

        if not os.path.exists(db_path):
            # Create simple database
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            # Create table
            cursor.execute("""
                CREATE TABLE ingredients (
                    id INTEGER PRIMARY KEY,
                    name TEXT,
                    category TEXT,
                    price_per_kg REAL,
                    volatility REAL,
                    odor_threshold REAL
                )
            """)

            # Insert sample ingredients
            ingredients = [
                ("Bergamot", "top", 120, 85, 0.15),
                ("Lemon", "top", 80, 90, 0.10),
                ("Rose", "heart", 5000, 45, 0.05),
                ("Jasmine", "heart", 4000, 40, 0.02),
                ("Sandalwood", "base", 800, 15, 0.08),
                ("Vanilla", "base", 600, 10, 0.001),
                ("Musk", "base", 1500, 5, 0.0001)
            ]

            cursor.executemany(
                "INSERT INTO ingredients (name, category, price_per_kg, volatility, odor_threshold) VALUES (?, ?, ?, ?, ?)",
                ingredients
            )

            conn.commit()
            conn.close()

        # Load ingredients - only select the columns we need
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT id, name, category, price_per_kg, volatility, odor_threshold FROM ingredients")
        self.ingredients = cursor.fetchall()
        conn.close()

    def evaluate_fragrance(self, individual):
        """
        Evaluate a fragrance formula

        Args:
            individual: List of ingredient concentrations

        Returns:
            Tuple of (quality, cost, stability)
        """
        # Normalize concentrations
        total = sum(individual[:len(self.ingredients)])
        if total == 0:
            return 0, float('inf'), 0

        normalized = [c / total * 100 for c in individual[:len(self.ingredients)]]

        # Calculate objectives
        quality = 0
        cost = 0
        stability = 0

        for i, (ing_id, name, category, price, volatility, threshold) in enumerate(self.ingredients):
            if i >= len(normalized):
                break

            conc = normalized[i]

            # Quality: based on balance and threshold
            if category == "top" and 20 <= conc <= 30:
                quality += 10
            elif category == "heart" and 30 <= conc <= 50:
                quality += 15
            elif category == "base" and 20 <= conc <= 40:
                quality += 12

            # Cost calculation
            cost += (conc / 100) * price

            # Stability: inverse of volatility weighted by concentration
            stability += (conc / 100) * (100 - volatility)

        return quality, cost, stability

    def optimize(self, num_generations: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Run MOGA optimization

        Args:
            num_generations: Number of generations to run

        Returns:
            List of Pareto-optimal solutions
        """
        if num_generations is None:
            num_generations = self.config.moga_generations

        # Create initial population
        population = self.toolbox.population(n=self.config.moga_population_size)

        # Evaluate initial population
        fitnesses = map(self.toolbox.evaluate, population)
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit

        # Run evolution
        for gen in range(num_generations):
            # Select parents
            offspring = self.toolbox.select(population, len(population))
            offspring = list(map(self.toolbox.clone, offspring))

            # Apply crossover and mutation
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < self.config.moga_crossover_rate:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if random.random() < self.config.moga_mutation_rate:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values

            # Evaluate offspring with invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # Replace population
            population[:] = offspring

        # Get Pareto front
        pareto_front = tools.sortNondominated(population, len(population), first_front_only=True)[0]

        # Format results
        results = []
        for ind in pareto_front:
            results.append({
                'formula': ind[:len(self.ingredients)],
                'quality': ind.fitness.values[0],
                'cost': ind.fitness.values[1],
                'stability': ind.fitness.values[2]
            })

        return results


# ============================================================================
# 3. Reinforcement Learning Component - PPO/REINFORCE
# ============================================================================

class RLAgent(nn.Module):
    """
    Reinforcement Learning Agent for fragrance evolution
    PPO/REINFORCE implementation
    """

    def __init__(self, config: UnifiedAIConfig):
        super().__init__()
        self.config = config

        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(config.rl_state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, config.rl_action_dim)
        )

        # Value network (for PPO)
        if config.rl_algorithm == "PPO":
            self.value_net = nn.Sequential(
                nn.Linear(config.rl_state_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 1)
            )

        # Optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=config.rl_learning_rate)

        # Memory for episodes
        self.memory = []

    def select_action(self, state: torch.Tensor) -> Tuple[int, torch.Tensor, Optional[torch.Tensor]]:
        """
        Select action using current policy

        Args:
            state: Current state tensor

        Returns:
            Tuple of (action, log_prob, value)
        """
        # Get action probabilities
        logits = self.policy_net(state)
        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)

        # Sample action
        action = dist.sample()
        log_prob = dist.log_prob(action)

        # Get value if using PPO
        value = None
        if self.config.rl_algorithm == "PPO":
            value = self.value_net(state)

        return action.item(), log_prob, value

    def update_reinforce(self, rewards: List[float], log_probs: List[torch.Tensor]):
        """
        Update policy using REINFORCE algorithm

        Args:
            rewards: List of rewards
            log_probs: List of log probabilities
        """
        # Calculate returns
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + self.config.rl_gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns)

        # Normalize returns
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Calculate loss
        policy_loss = []
        for log_prob, R in zip(log_probs, returns):
            policy_loss.append(-log_prob * R)

        # Update
        self.optimizer.zero_grad()
        loss = torch.stack(policy_loss).sum()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def update_ppo(self, states: torch.Tensor, actions: torch.Tensor,
                   rewards: torch.Tensor, old_log_probs: torch.Tensor,
                   advantages: torch.Tensor, returns: torch.Tensor):
        """
        Update policy using PPO algorithm

        Args:
            states: State tensor
            actions: Action tensor
            rewards: Reward tensor
            old_log_probs: Old log probabilities
            advantages: Advantage estimates
            returns: Return estimates
        """
        # Get current policy values
        logits = self.policy_net(states)
        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)

        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()

        values = self.value_net(states).squeeze()

        # Calculate ratio for PPO
        ratio = torch.exp(log_probs - old_log_probs)

        # Clipped surrogate objective
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.config.rl_clip_epsilon,
                          1 + self.config.rl_clip_epsilon) * advantages

        # Calculate losses
        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss = F.mse_loss(values, returns)
        entropy_loss = -entropy.mean()

        # Total loss
        loss = policy_loss + 0.5 * value_loss + 0.01 * entropy_loss

        # Update
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)
        self.optimizer.step()

        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy_loss.item()
        }


# ============================================================================
# 4. Unified AI System - Main Integration
# ============================================================================

class UnifiedFragranceAI:
    """
    통합 AI 시스템
    모든 AI 컴포넌트를 통합하여 향수 생성 및 최적화
    """

    def __init__(self, config: Optional[UnifiedAIConfig] = None):
        if config is None:
            config = UnifiedAIConfig()

        self.config = config

        # Set random seeds
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        random.seed(config.seed)

        # Initialize components
        self.dl_generator = FragranceGeneratorDL(config).to(config.device)
        self.moga_optimizer = MOGAOptimizer(config)
        self.rl_agent = RLAgent(config).to(config.device)

        logger.info("Unified AI System initialized successfully")

    def generate_with_dl(self, seed_notes: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        Generate fragrance using deep learning

        Args:
            seed_notes: Optional starting notes

        Returns:
            Generated formula
        """
        if seed_notes:
            seed_tensor = torch.tensor([seed_notes], device=self.config.device)
        else:
            seed_tensor = None

        self.dl_generator.eval()
        with torch.no_grad():
            result = self.dl_generator.generate(seed_tensor)

        # Convert to readable format
        formula = {
            'notes': result['notes'].cpu().numpy().tolist(),
            'concentrations': result['concentrations'].cpu().numpy().tolist()
        }

        return formula

    def optimize_with_moga(self, constraints: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Optimize fragrance using MOGA

        Args:
            constraints: Optional constraints for optimization

        Returns:
            Pareto-optimal solutions
        """
        results = self.moga_optimizer.optimize()

        # Apply constraints if provided
        if constraints:
            filtered_results = []
            for result in results:
                if constraints.get('max_cost') and result['cost'] > constraints['max_cost']:
                    continue
                if constraints.get('min_quality') and result['quality'] < constraints['min_quality']:
                    continue
                filtered_results.append(result)
            results = filtered_results

        return results

    def evolve_with_rl(self, current_formula: List[float],
                       user_feedback: float) -> Dict[str, Any]:
        """
        Evolve fragrance using reinforcement learning

        Args:
            current_formula: Current formula (as state)
            user_feedback: User rating (1-5)

        Returns:
            Evolved formula
        """
        # Convert formula to state
        if len(current_formula) >= self.config.rl_state_dim:
            state = torch.tensor(current_formula[:self.config.rl_state_dim],
                               dtype=torch.float32, device=self.config.device)
        else:
            # Pad if necessary
            padded_formula = current_formula + [0.0] * (self.config.rl_state_dim - len(current_formula))
            state = torch.tensor(padded_formula,
                               dtype=torch.float32, device=self.config.device)

        state = state.unsqueeze(0)  # Add batch dimension

        # Select action
        action, log_prob, value = self.rl_agent.select_action(state)

        # Apply action to formula
        evolved_formula = self.apply_evolution_action(current_formula, action)

        # Store experience for learning
        reward = (user_feedback - 3) / 2  # Normalize feedback to [-1, 1]
        self.rl_agent.memory.append((state, action, reward, log_prob, value))

        # Update policy if enough experience
        if len(self.rl_agent.memory) >= 32:
            self.train_rl_agent()

        return {
            'evolved_formula': evolved_formula,
            'action_taken': action,
            'expected_improvement': reward
        }

    def apply_evolution_action(self, formula: List[float], action: int) -> List[float]:
        """
        Apply evolution action to formula

        Args:
            formula: Current formula
            action: Action index

        Returns:
            Modified formula
        """
        evolved = formula.copy()

        # Define actions
        if action == 0:  # Increase top notes
            for i in range(min(3, len(evolved))):
                evolved[i] *= 1.2
        elif action == 1:  # Decrease top notes
            for i in range(min(3, len(evolved))):
                evolved[i] *= 0.8
        elif action == 2:  # Increase heart notes
            for i in range(3, min(7, len(evolved))):
                evolved[i] *= 1.2
        elif action == 3:  # Decrease heart notes
            for i in range(3, min(7, len(evolved))):
                evolved[i] *= 0.8
        elif action == 4:  # Increase base notes
            for i in range(7, min(len(evolved), 10)):
                evolved[i] *= 1.2
        elif action == 5:  # Decrease base notes
            for i in range(7, min(len(evolved), 10)):
                evolved[i] *= 0.8
        elif action == 6:  # Add complexity
            for i in range(len(evolved)):
                evolved[i] += random.uniform(-5, 5)
        elif action == 7:  # Simplify
            for i in range(len(evolved)):
                if evolved[i] < 10:
                    evolved[i] = 0
        elif action == 8:  # Boost intensity
            evolved = [c * 1.1 for c in evolved]
        elif action == 9:  # Reduce intensity
            evolved = [c * 0.9 for c in evolved]
        elif action == 10:  # Random mutation
            idx = random.randint(0, len(evolved) - 1)
            evolved[idx] += random.uniform(-10, 10)
        elif action == 11:  # Reset component
            idx = random.randint(0, len(evolved) - 1)
            evolved[idx] = random.uniform(0, 30)

        # Normalize to 100%
        total = sum(evolved)
        if total > 0:
            evolved = [(c / total) * 100 for c in evolved]

        return evolved

    def train_rl_agent(self):
        """Train RL agent using collected experience"""
        if len(self.rl_agent.memory) == 0:
            return

        # Extract experience
        states, actions, rewards, log_probs, values = zip(*self.rl_agent.memory)

        if self.config.rl_algorithm == "REINFORCE":
            loss = self.rl_agent.update_reinforce(rewards, log_probs)
            logger.info(f"REINFORCE training loss: {loss:.4f}")
        else:  # PPO
            # Convert to tensors
            states = torch.cat(states)
            actions = torch.tensor(actions, device=self.config.device)
            rewards = torch.tensor(rewards, device=self.config.device, dtype=torch.float32)
            old_log_probs = torch.stack(log_probs).detach()

            # Calculate advantages and returns
            values = torch.cat(values).squeeze() if values[0] is not None else torch.zeros_like(rewards)
            advantages = rewards - values.detach()
            returns = rewards

            # Update
            losses = self.rl_agent.update_ppo(states, actions, rewards,
                                             old_log_probs, advantages, returns)
            logger.info(f"PPO losses - Policy: {losses['policy_loss']:.4f}, "
                       f"Value: {losses['value_loss']:.4f}, "
                       f"Entropy: {losses['entropy']:.4f}")

        # Clear memory
        self.rl_agent.memory = []

    def generate_complete_fragrance(self, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate complete fragrance using all AI components

        Args:
            requirements: User requirements

        Returns:
            Complete fragrance formula with optimization
        """
        result = {
            'timestamp': datetime.now().isoformat(),
            'requirements': requirements
        }

        # Step 1: Generate initial formula with DL
        logger.info("Step 1: Generating initial formula with Deep Learning")
        dl_formula = self.generate_with_dl()
        result['dl_formula'] = dl_formula

        # Step 2: Optimize with MOGA
        logger.info("Step 2: Optimizing with Multi-Objective Genetic Algorithm")
        moga_results = self.optimize_with_moga(requirements.get('constraints'))
        if moga_results:
            result['moga_optimized'] = moga_results[0]  # Best solution

        # Step 3: Prepare for RL evolution (simulate user feedback)
        logger.info("Step 3: Preparing for Reinforcement Learning evolution")
        if 'formula' in requirements:
            current = requirements['formula']
        else:
            # Make sure we have a list, not a nested list
            if 'concentrations' in dl_formula:
                conc = dl_formula['concentrations']
                # If it's a nested list (from batch processing), take the first item
                if isinstance(conc, list) and len(conc) > 0 and isinstance(conc[0], list):
                    current = conc[0]
                else:
                    current = conc
            else:
                current = [10] * 20

        # Simulate evolution with default feedback
        rl_result = self.evolve_with_rl(current, requirements.get('user_rating', 3.5))
        result['rl_evolved'] = rl_result

        # Combine all results
        result['final_formula'] = {
            'method': 'AI_Unified_System',
            'components': {
                'deep_learning': dl_formula,
                'moga_optimization': moga_results[0] if moga_results else None,
                'rl_evolution': rl_result
            }
        }

        logger.info("Complete fragrance generation finished")
        return result


# ============================================================================
# Utility Functions
# ============================================================================

def create_unified_ai_system(config: Optional[Dict[str, Any]] = None) -> UnifiedFragranceAI:
    """
    Factory function to create unified AI system

    Args:
        config: Optional configuration dictionary

    Returns:
        UnifiedFragranceAI instance
    """
    if config:
        ai_config = UnifiedAIConfig(**config)
    else:
        ai_config = UnifiedAIConfig()

    return UnifiedFragranceAI(ai_config)


def test_unified_system():
    """Test the unified AI system"""
    logger.info("Testing Unified AI System")

    # Create system
    ai_system = create_unified_ai_system()

    # Test DL generation
    logger.info("Testing Deep Learning generation...")
    dl_result = ai_system.generate_with_dl()
    logger.info(f"DL Result: {len(dl_result['notes'][0])} notes generated")

    # Test MOGA optimization
    logger.info("Testing MOGA optimization...")
    moga_results = ai_system.optimize_with_moga({'max_cost': 1000})
    logger.info(f"MOGA Result: {len(moga_results)} Pareto-optimal solutions found")

    # Test RL evolution
    logger.info("Testing RL evolution...")
    test_formula = [10.0] * 20
    rl_result = ai_system.evolve_with_rl(test_formula, 4.0)
    logger.info(f"RL Result: Formula evolved with action {rl_result['action_taken']}")

    # Test complete generation
    logger.info("Testing complete generation pipeline...")
    requirements = {
        'style': 'fresh',
        'season': 'summer',
        'constraints': {'max_cost': 500},
        'user_rating': 4.5
    }
    complete_result = ai_system.generate_complete_fragrance(requirements)
    logger.info(f"Complete generation successful: {complete_result['final_formula']['method']}")

    return True


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Run test
    success = test_unified_system()

    if success:
        logger.info("All tests passed successfully!")
    else:
        logger.error("Some tests failed")

    # Example usage
    logger.info("\n" + "="*60)
    logger.info("Example: Creating a summer fresh fragrance")

    ai = create_unified_ai_system()
    result = ai.generate_complete_fragrance({
        'style': 'fresh',
        'season': 'summer',
        'constraints': {
            'max_cost': 300,
            'min_quality': 50
        }
    })

    logger.info(f"Final formula created: {result['final_formula']['method']}")

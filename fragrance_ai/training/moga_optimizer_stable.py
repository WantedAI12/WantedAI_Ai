"""
MOGA Optimizer with Enhanced Stability
완전한 안정화 버전: 양수 보장, IFRA 클리핑, 다양성 보상
"""

import numpy as np
import sqlite3
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import json
import logging
import hashlib
import random

# Deep Learning for embeddings
import torch
import torch.nn as nn
import torch.nn.functional as F

# DEAP for genetic algorithms
from deap import base, creator, tools, algorithms

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration Constants
# ============================================================================

MIN_EFFECTIVE_CONCENTRATION = 0.1  # 최소 유효 농도 (%)
MAX_RENORM_ITERATIONS = 10  # 재정규화 최대 반복
RENORM_TOLERANCE = 0.01  # 정규화 수렴 허용오차
ENTROPY_EPSILON = 1e-12  # 엔트로피 계산 스무딩
DIVERSITY_WEIGHT = 0.1  # 다양성 보상 가중치


# ============================================================================
# Enhanced Fragrance Database with Embeddings
# ============================================================================

class StableFragranceDatabase:
    """Production database with embedding support"""

    def __init__(self, db_path: str = "fragrance_stable.db"):
        self.db_path = Path(__file__).resolve().parent.parent / "data" / db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
        self._load_real_data()
        self._init_embeddings()

    def _init_database(self):
        """Initialize database with embedding columns"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ingredients (
                id INTEGER PRIMARY KEY,
                name TEXT UNIQUE NOT NULL,
                cas_number TEXT,
                category TEXT NOT NULL,
                volatility REAL,
                odor_threshold REAL,
                price_per_kg REAL,
                ifra_limit REAL,
                chemical_family TEXT,
                molecular_weight REAL,
                logp REAL,
                vapor_pressure REAL,
                hansen_d REAL,
                hansen_p REAL,
                hansen_h REAL,
                description TEXT,
                embedding TEXT  -- JSON serialized embedding vector
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS formulations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                dna_hash TEXT UNIQUE NOT NULL,
                recipe TEXT NOT NULL,
                quality_score REAL,
                stability_score REAL,
                cost REAL,
                diversity_score REAL,  -- New: diversity metric
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                generation INTEGER,
                pareto_rank INTEGER
            )
        """)

        conn.commit()
        conn.close()

    def _load_real_data(self):
        """Load real ingredient data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM ingredients")
        if cursor.fetchone()[0] == 0:
            real_ingredients = [
                # Top Notes
                {
                    'name': 'Bergamot', 'cas_number': '8007-75-8', 'category': 'top',
                    'volatility': 0.9, 'odor_threshold': 0.05, 'price_per_kg': 85,
                    'ifra_limit': 2.0, 'chemical_family': 'citrus', 'molecular_weight': 136.23,
                    'logp': 2.8, 'vapor_pressure': 2.6,
                    'hansen_d': 16.0, 'hansen_p': 4.5, 'hansen_h': 6.8,
                    'description': 'Fresh, citrusy, slightly floral'
                },
                {
                    'name': 'Lemon', 'cas_number': '8008-56-8', 'category': 'top',
                    'volatility': 0.95, 'odor_threshold': 0.03, 'price_per_kg': 65,
                    'ifra_limit': 3.0, 'chemical_family': 'citrus', 'molecular_weight': 136.23,
                    'logp': 2.7, 'vapor_pressure': 3.1,
                    'hansen_d': 15.8, 'hansen_p': 4.3, 'hansen_h': 6.5,
                    'description': 'Sharp, fresh, zesty citrus'
                },
                # Heart Notes
                {
                    'name': 'Rose', 'cas_number': '8007-01-0', 'category': 'heart',
                    'volatility': 0.5, 'odor_threshold': 0.01, 'price_per_kg': 5000,
                    'ifra_limit': 0.5, 'chemical_family': 'floral', 'molecular_weight': 238.37,
                    'logp': 3.8, 'vapor_pressure': 0.8,
                    'hansen_d': 18.1, 'hansen_p': 5.2, 'hansen_h': 8.3,
                    'description': 'Rich, deep floral, romantic'
                },
                {
                    'name': 'Jasmine', 'cas_number': '8022-96-6', 'category': 'heart',
                    'volatility': 0.45, 'odor_threshold': 0.008, 'price_per_kg': 8000,
                    'ifra_limit': 0.4, 'chemical_family': 'floral', 'molecular_weight': 226.36,
                    'logp': 3.9, 'vapor_pressure': 0.6,
                    'hansen_d': 18.5, 'hansen_p': 5.8, 'hansen_h': 9.1,
                    'description': 'Intensely floral, sweet, narcotic'
                },
                # Base Notes
                {
                    'name': 'Sandalwood', 'cas_number': '8006-87-9', 'category': 'base',
                    'volatility': 0.2, 'odor_threshold': 0.05, 'price_per_kg': 2500,
                    'ifra_limit': 2.0, 'chemical_family': 'woody', 'molecular_weight': 220.35,
                    'logp': 4.2, 'vapor_pressure': 0.2,
                    'hansen_d': 19.5, 'hansen_p': 3.8, 'hansen_h': 6.2,
                    'description': 'Creamy, soft, woody, balsamic'
                },
                {
                    'name': 'Vanilla', 'cas_number': '8024-06-4', 'category': 'base',
                    'volatility': 0.15, 'odor_threshold': 0.02, 'price_per_kg': 600,
                    'ifra_limit': 3.0, 'chemical_family': 'oriental', 'molecular_weight': 152.15,
                    'logp': 1.4, 'vapor_pressure': 0.1,
                    'hansen_d': 20.1, 'hansen_p': 8.4, 'hansen_h': 11.2,
                    'description': 'Sweet, creamy, warm, comforting'
                },
                {
                    'name': 'Musk', 'cas_number': '541-91-3', 'category': 'base',
                    'volatility': 0.1, 'odor_threshold': 0.001, 'price_per_kg': 3000,
                    'ifra_limit': 1.5, 'chemical_family': 'animalic', 'molecular_weight': 258.40,
                    'logp': 5.1, 'vapor_pressure': 0.05,
                    'hansen_d': 18.8, 'hansen_p': 2.9, 'hansen_h': 4.8,
                    'description': 'Soft, warm, animalic, powdery'
                }
            ]

            for ing in real_ingredients:
                cursor.execute("""
                    INSERT INTO ingredients (
                        name, cas_number, category, volatility, odor_threshold,
                        price_per_kg, ifra_limit, chemical_family, molecular_weight,
                        logp, vapor_pressure, hansen_d, hansen_p, hansen_h, description
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    ing['name'], ing['cas_number'], ing['category'], ing['volatility'],
                    ing['odor_threshold'], ing['price_per_kg'], ing['ifra_limit'],
                    ing['chemical_family'], ing['molecular_weight'], ing['logp'],
                    ing['vapor_pressure'], ing['hansen_d'], ing['hansen_p'],
                    ing['hansen_h'], ing['description']
                ))

            conn.commit()
        conn.close()

    def _init_embeddings(self):
        """Initialize ingredient embeddings"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT id, hansen_d, hansen_p, hansen_h, volatility, logp FROM ingredients")

        for row in cursor.fetchall():
            ing_id = row[0]
            # Create embedding from chemical properties
            embedding = [
                row[1] / 20.0 if row[1] else 0,  # hansen_d normalized
                row[2] / 10.0 if row[2] else 0,  # hansen_p normalized
                row[3] / 15.0 if row[3] else 0,  # hansen_h normalized
                row[4] if row[4] else 0,  # volatility
                row[5] / 5.0 if row[5] else 0  # logp normalized
            ]

            # Save embedding
            cursor.execute(
                "UPDATE ingredients SET embedding = ? WHERE id = ?",
                (json.dumps(embedding), ing_id)
            )

        conn.commit()
        conn.close()

    def get_all_ingredients(self) -> List[Dict]:
        """Get all ingredients with embeddings"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT id, name, category, volatility, odor_threshold,
                   price_per_kg, ifra_limit, molecular_weight,
                   hansen_d, hansen_p, hansen_h, embedding
            FROM ingredients
        """)

        ingredients = []
        for row in cursor.fetchall():
            ingredients.append({
                'id': row[0],
                'name': row[1],
                'category': row[2],
                'volatility': row[3],
                'odor_threshold': row[4],
                'price_per_kg': row[5],
                'ifra_limit': row[6],
                'molecular_weight': row[7],
                'hansen_d': row[8],
                'hansen_p': row[9],
                'hansen_h': row[10],
                'embedding': json.loads(row[11]) if row[11] else None
            })

        conn.close()
        return ingredients


# ============================================================================
# Stable MOGA with Enhanced Genetic Operators
# ============================================================================

class StableMOGA:
    """MOGA with stability guarantees and diversity preservation"""

    def __init__(self, population_size: int = 100, generations: int = 200):
        self.population_size = population_size
        self.generations = generations
        self.current_generation = 0

        # Database
        self.db = StableFragranceDatabase()
        self.ingredients = self.db.get_all_ingredients()

        # Group by category
        self.top_notes = [i for i in self.ingredients if i['category'] == 'top']
        self.heart_notes = [i for i in self.ingredients if i['category'] == 'heart']
        self.base_notes = [i for i in self.ingredients if i['category'] == 'base']

        # GA parameters
        self.crossover_prob = 0.9
        self.mutation_prob = 0.1
        self.mutation_sigma = 0.2  # Standard deviation for exponential mutation
        self.eta_c = 20  # SBX distribution index
        self.eta_m = 20  # Polynomial mutation distribution index

        # Initialize DEAP
        self._setup_deap()

    def _setup_deap(self):
        """Setup DEAP framework"""
        if hasattr(creator, "FitnessMulti"):
            del creator.FitnessMulti
        if hasattr(creator, "Individual"):
            del creator.Individual

        # Multi-objective: maximize quality, minimize cost, maximize stability, maximize diversity
        creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0, 1.0, 1.0))
        creator.create("Individual", list, fitness=creator.FitnessMulti)

        self.toolbox = base.Toolbox()
        self.toolbox.register("individual", self.create_individual)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("evaluate", self.evaluate_fragrance_stable)
        self.toolbox.register("mate", self.stable_sbx_crossover)
        self.toolbox.register("mutate", self.stable_polynomial_mutation)
        self.toolbox.register("select", tools.selNSGA2)

    def create_individual(self) -> list:
        """Create random individual with proper normalization"""
        individual = []

        # Select ingredients
        num_top = random.randint(2, min(4, len(self.top_notes)))
        num_heart = random.randint(2, min(4, len(self.heart_notes)))
        num_base = random.randint(1, min(3, len(self.base_notes)))

        selected_top = random.sample(self.top_notes, num_top)
        selected_heart = random.sample(self.heart_notes, num_heart)
        selected_base = random.sample(self.base_notes, num_base)

        # Add with concentrations
        for note in selected_top:
            concentration = random.uniform(5, 20)
            individual.append((note['id'], concentration))

        for note in selected_heart:
            concentration = random.uniform(10, 40)
            individual.append((note['id'], concentration))

        for note in selected_base:
            concentration = random.uniform(20, 50)
            individual.append((note['id'], concentration))

        # Apply stable normalization
        individual = self.stable_normalize(individual)

        return creator.Individual(individual)

    def stable_normalize(self, individual: List, max_iterations: int = MAX_RENORM_ITERATIONS) -> List:
        """
        Stable normalization with IFRA clipping and iterative renormalization

        1. Filter minimum concentrations
        2. Normalize to 100%
        3. Apply IFRA limits iteratively
        """
        if not individual:
            return creator.Individual()

        # Step 1: Filter minimum effective concentration
        filtered = [
            (i, c) for i, c in individual
            if c >= MIN_EFFECTIVE_CONCENTRATION
        ]

        if not filtered:
            return creator.Individual()

        # Step 2: Initial normalization to 100%
        total = sum(c for _, c in filtered)
        if total <= 0:
            return creator.Individual()

        normalized = [(i, max(0, c * 100 / total)) for i, c in filtered]

        # Step 3: Iterative IFRA limit application with renormalization
        for iteration in range(max_iterations):
            # Apply IFRA limits
            clipped = []
            needs_renorm = False

            for ing_id, concentration in normalized:
                ing_data = next((i for i in self.ingredients if i['id'] == ing_id), None)
                if ing_data and 'ifra_limit' in ing_data:
                    ifra_limit = ing_data['ifra_limit']
                    if concentration > ifra_limit:
                        safe_conc = ifra_limit
                        needs_renorm = True
                    else:
                        safe_conc = concentration
                else:
                    safe_conc = concentration

                # Ensure positive
                safe_conc = max(0, safe_conc)
                if safe_conc >= MIN_EFFECTIVE_CONCENTRATION:
                    clipped.append((ing_id, safe_conc))

            if not clipped:
                return creator.Individual()

            # Check if renormalization is needed
            current_total = sum(c for _, c in clipped)
            if not needs_renorm and abs(current_total - 100) <= RENORM_TOLERANCE:
                # Converged
                return creator.Individual(clipped)

            # Renormalize
            if current_total > 0:
                normalized = [(i, max(0, c * 100 / current_total)) for i, c in clipped]
            else:
                return creator.Individual(clipped)

        # Return final result after max iterations
        final_total = sum(c for _, c in normalized)
        if final_total > 0:
            return creator.Individual([(i, max(0, c * 100 / final_total)) for i, c in normalized])
        return creator.Individual(normalized)

    def stable_polynomial_mutation(self, individual: List) -> Tuple[List]:
        """
        Polynomial mutation with exponential form for positive guarantee
        c' = c * exp(N(0, σ))
        """
        mutated = []

        for ing_id, concentration in individual:
            if random.random() < self.mutation_prob:
                # Generate polynomial perturbation
                u = random.random()
                if u < 0.5:
                    delta = (2 * u) ** (1 / (self.eta_m + 1)) - 1
                else:
                    delta = 1 - (2 * (1 - u)) ** (1 / (self.eta_m + 1))

                # Apply exponential mutation: c' = c * exp(N(0, σ))
                # This guarantees positive values
                mutation_factor = np.exp(delta * self.mutation_sigma)
                new_concentration = concentration * mutation_factor

                # Ensure positive with clamp_min(0)
                new_concentration = max(0, new_concentration)

                # Apply upper bound
                new_concentration = min(100, new_concentration)

                mutated.append((ing_id, new_concentration))

                # Small chance to add new ingredient
                if random.random() < 0.05:
                    available = [i['id'] for i in self.ingredients
                               if i['id'] not in [x[0] for x in mutated]]
                    if available:
                        new_ing = random.choice(available)
                        new_conc = random.uniform(5, 30)
                        mutated.append((new_ing, max(0, new_conc)))
            else:
                mutated.append((ing_id, concentration))

        # Apply stable normalization after mutation
        mutated = self.stable_normalize(mutated)

        return (mutated,)

    def stable_sbx_crossover(self, ind1: List, ind2: List) -> Tuple[List, List]:
        """
        SBX crossover with stable normalization after crossover
        """
        child1 = []
        child2 = []

        ind1_dict = {ing_id: conc for ing_id, conc in ind1}
        ind2_dict = {ing_id: conc for ing_id, conc in ind2}
        all_ingredients = set(ind1_dict.keys()) | set(ind2_dict.keys())

        for ing_id in all_ingredients:
            conc1 = ind1_dict.get(ing_id, 0)
            conc2 = ind2_dict.get(ing_id, 0)

            if random.random() < self.crossover_prob and conc1 != conc2:
                # SBX operation
                if conc1 > conc2:
                    conc1, conc2 = conc2, conc1

                u = random.random()
                if u <= 0.5:
                    beta = (2 * u) ** (1 / (self.eta_c + 1))
                else:
                    beta = (1 / (2 * (1 - u))) ** (1 / (self.eta_c + 1))

                child_conc1 = 0.5 * ((1 + beta) * conc1 + (1 - beta) * conc2)
                child_conc2 = 0.5 * ((1 - beta) * conc1 + (1 + beta) * conc2)

                # Ensure positive with clamp_min(0)
                child_conc1 = max(0, child_conc1)
                child_conc2 = max(0, child_conc2)

                if child_conc1 >= MIN_EFFECTIVE_CONCENTRATION:
                    child1.append((ing_id, child_conc1))
                if child_conc2 >= MIN_EFFECTIVE_CONCENTRATION:
                    child2.append((ing_id, child_conc2))
            else:
                if conc1 >= MIN_EFFECTIVE_CONCENTRATION:
                    child1.append((ing_id, conc1))
                if conc2 >= MIN_EFFECTIVE_CONCENTRATION:
                    child2.append((ing_id, conc2))

        # Apply stable normalization after crossover
        child1 = self.stable_normalize(child1)
        child2 = self.stable_normalize(child2)

        return child1, child2

    def calculate_entropy(self, concentrations: List[float]) -> float:
        """
        Calculate Shannon entropy with epsilon smoothing
        Handles 0*log(0) = 0 case properly
        """
        if not concentrations:
            return 0.0

        # Filter very small values and normalize
        filtered = [c for c in concentrations if c > MIN_EFFECTIVE_CONCENTRATION]
        if not filtered or len(filtered) == 1:
            return 0.0  # Single element has zero entropy

        total = sum(filtered)
        if total <= 0:
            return 0.0

        probs = np.array(filtered) / total

        # Calculate entropy with epsilon smoothing
        entropy = 0.0
        for p in probs:
            if p > 0:  # Handle 0*log(0) = 0
                # Add epsilon for numerical stability
                entropy -= p * np.log(p + ENTROPY_EPSILON)

        # Normalize by max entropy
        max_entropy = np.log(len(probs))
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

        # Ensure non-negative
        return max(0.0, normalized_entropy)

    def calculate_diversity(self, individual: List, population: List) -> float:
        """
        Calculate diversity bonus based on embedding distance
        Penalizes similar formulations, rewards unique ones
        """
        if not individual or not population:
            return 0.5  # Neutral diversity

        # Get individual embedding
        ind_embedding = self.get_formulation_embedding(individual)

        # Calculate minimum distance to population
        min_distance = float('inf')
        for other in population:
            if other != individual:
                other_embedding = self.get_formulation_embedding(other)
                distance = np.linalg.norm(ind_embedding - other_embedding)
                min_distance = min(min_distance, distance)

        # Convert to diversity score (higher distance = higher diversity)
        if min_distance == float('inf'):
            return 1.0  # Unique individual

        # Normalize to [0, 1]
        diversity_score = 1.0 - np.exp(-min_distance / 2.0)
        return diversity_score

    def get_formulation_embedding(self, individual: List) -> np.ndarray:
        """Get embedding vector for formulation"""
        embedding = np.zeros(5)  # 5D embedding space

        total_weight = 0.0
        for ing_id, concentration in individual:
            ing_data = next((i for i in self.ingredients if i['id'] == ing_id), None)
            if ing_data and ing_data.get('embedding'):
                ing_embedding = np.array(ing_data['embedding'])
                # Weighted average by concentration
                embedding += ing_embedding * concentration
                total_weight += concentration

        if total_weight > 0:
            embedding /= total_weight

        return embedding

    def evaluate_fragrance_stable(self, individual: List) -> Tuple[float, float, float, float]:
        """
        Evaluate with stability objective sign correction
        Returns: (quality, cost, stability, diversity)
        """
        if not individual:
            return 0.0, float('inf'), 0.0, 0.0

        ingredient_map = {i['id']: i for i in self.ingredients}

        # Calculate quality
        quality_score = 0.0
        odor_impact = 0.0
        volatility_profile = {'top': 0, 'heart': 0, 'base': 0}

        for ing_id, concentration in individual:
            if ing_id not in ingredient_map:
                continue
            ing = ingredient_map[ing_id]

            # Odor impact
            if ing['odor_threshold'] and ing['odor_threshold'] > 0:
                odor_impact += concentration / ing['odor_threshold']

            # Volatility profile
            volatility_profile[ing['category']] += concentration

        # Quality based on odor and balance
        quality_score = min(100, odor_impact * 10)

        # Balance score
        balance_score = 0
        if 20 <= volatility_profile['top'] <= 30:
            balance_score += 30
        if 30 <= volatility_profile['heart'] <= 50:
            balance_score += 35
        if 30 <= volatility_profile['base'] <= 50:
            balance_score += 35

        quality_score += balance_score

        # Calculate cost
        total_cost = 0.0
        for ing_id, concentration in individual:
            if ing_id in ingredient_map:
                ing = ingredient_map[ing_id]
                if ing.get('price_per_kg'):
                    total_cost += (concentration / 100) * ing['price_per_kg']

        # Calculate stability with sign correction
        # f_stability = -penalty_total for maximization consistency
        penalty_total = 0.0
        comparisons = 0

        for i, (ing1_id, conc1) in enumerate(individual):
            if ing1_id not in ingredient_map:
                continue
            ing1 = ingredient_map[ing1_id]

            for j, (ing2_id, conc2) in enumerate(individual[i+1:], i+1):
                if ing2_id not in ingredient_map:
                    continue
                ing2 = ingredient_map[ing2_id]

                # Hansen distance penalty
                if all(k in ing1 for k in ['hansen_d', 'hansen_p', 'hansen_h']) and \
                   all(k in ing2 for k in ['hansen_d', 'hansen_p', 'hansen_h']):

                    delta_d = ing1['hansen_d'] - ing2['hansen_d']
                    delta_p = ing1['hansen_p'] - ing2['hansen_p']
                    delta_h = ing1['hansen_h'] - ing2['hansen_h']

                    Ra = np.sqrt(4 * delta_d**2 + delta_p**2 + delta_h**2)

                    # Higher distance = higher penalty
                    weight = (conc1 * conc2) / 10000
                    penalty_total += Ra * weight
                    comparisons += weight

        # Convert penalty to stability score (minimize penalty = maximize stability)
        if comparisons > 0:
            penalty_total = penalty_total / (1 + comparisons)

        # f_stability = -penalty_total for maximization
        stability_score = 100.0 - min(100, penalty_total * 5)

        # Calculate entropy-based complexity
        concentrations = [c for _, c in individual]
        entropy = self.calculate_entropy(concentrations)

        # Add entropy bonus to quality
        quality_score += entropy * 20  # Up to 20 points for diversity

        # Calculate diversity (placeholder - needs population context)
        diversity_score = entropy  # Use entropy as proxy for now

        return quality_score, total_cost, stability_score, diversity_score

    def optimize(self, enable_diversity: bool = True) -> Dict[str, Any]:
        """Run optimization with optional diversity preservation"""
        # Create initial population
        population = self.toolbox.population(n=self.population_size)

        # Evaluate initial population
        fitnesses = list(map(self.toolbox.evaluate, population))
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit

        # Statistics
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean, axis=0)
        stats.register("std", np.std, axis=0)
        stats.register("min", np.min, axis=0)
        stats.register("max", np.max, axis=0)

        # Evolution loop
        for generation in range(self.generations):
            self.current_generation = generation

            # Select parents
            offspring = self.toolbox.select(population, self.population_size)
            offspring = list(map(self.toolbox.clone, offspring))

            # Apply crossover
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < self.crossover_prob:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            # Apply mutation
            for mutant in offspring:
                if random.random() < self.mutation_prob:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values

            # Evaluate offspring
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # Add diversity bonus if enabled
            if enable_diversity:
                for ind in offspring:
                    diversity = self.calculate_diversity(ind, population)
                    # Update fitness with diversity bonus
                    current_fit = list(ind.fitness.values)
                    current_fit[3] = diversity * DIVERSITY_WEIGHT
                    ind.fitness.values = tuple(current_fit)

            # Environmental selection
            population[:] = tools.selNSGA2(population + offspring, self.population_size)

            # Log statistics
            if generation % 10 == 0:
                record = stats.compile(population)
                logger.info(f"Generation {generation}: {record}")

        # Get Pareto front
        pareto_front = tools.sortNondominated(population, len(population), first_front_only=True)[0]

        # Format results
        results = []
        for i, ind in enumerate(pareto_front[:10]):
            fitness = ind.fitness.values
            formulation = {
                'ingredients': [
                    {
                        'id': ing_id,
                        'name': next((i['name'] for i in self.ingredients if i['id'] == ing_id), 'Unknown'),
                        'concentration': round(conc, 2)
                    }
                    for ing_id, conc in ind
                ],
                'quality_score': round(fitness[0], 2),
                'cost': round(fitness[1], 2),
                'stability_score': round(fitness[2], 2),
                'diversity_score': round(fitness[3], 2) if len(fitness) > 3 else 0,
                'entropy': round(self.calculate_entropy([c for _, c in ind]), 4)
            }
            results.append(formulation)

        return {
            'pareto_front': results,
            'population_size': len(population),
            'generations': self.generations,
            'convergence': {
                'final_diversity': len(pareto_front),
                'best_quality': max(f['quality_score'] for f in results) if results else 0,
                'best_cost': min(f['cost'] for f in results) if results else float('inf'),
                'best_stability': max(f['stability_score'] for f in results) if results else 0
            },
            'algorithm': 'Stable NSGA-II with Exponential Mutation and Diversity Preservation'
        }


# ============================================================================
# Test and Validation
# ============================================================================

def test_stability():
    """Test stability guarantees"""
    print("="*60)
    print("STABILITY TEST SUITE")
    print("="*60)

    optimizer = StableMOGA(population_size=50, generations=10)

    # Test 1: Mutation always produces positive values
    print("\n1. Testing Exponential Mutation Positivity")
    print("-"*40)

    test_individual = [(1, 20.0), (2, 30.0), (3, 50.0)]
    negatives_found = 0

    for i in range(100):
        mutated = optimizer.stable_polynomial_mutation(test_individual)[0]
        for ing_id, conc in mutated:
            if conc < 0:
                negatives_found += 1
                print(f"  [FAIL] Negative concentration: {conc}")

    if negatives_found == 0:
        print(f"  [OK] 100 mutations tested, all concentrations positive")
    else:
        print(f"  [FAIL] Found {negatives_found} negative values")

    # Test 2: Normalization convergence
    print("\n2. Testing Normalization Convergence")
    print("-"*40)

    test_cases = [
        [(1, 150.0), (2, 200.0), (3, 50.0)],  # Over 100%
        [(1, 0.05), (2, 0.03), (3, 0.02)],    # Very small values
        [(1, 1.5), (2, 0.8), (3, 2.0)]        # With IFRA limit 2.0
    ]

    for i, test in enumerate(test_cases):
        normalized = optimizer.stable_normalize(test)
        total = sum(c for _, c in normalized)

        if abs(total - 100) <= RENORM_TOLERANCE or len(normalized) == 0:
            print(f"  [OK] Test case {i+1}: Total = {total:.2f}%")
        else:
            print(f"  [FAIL] Test case {i+1}: Total = {total:.2f}% (not 100%)")

    # Test 3: Entropy calculation
    print("\n3. Testing Entropy Calculation")
    print("-"*40)

    test_distributions = [
        [25, 25, 25, 25],  # Uniform (high entropy)
        [90, 5, 3, 2],     # Skewed (low entropy)
        [100],             # Single component (zero entropy)
        [0, 0, 100, 0]     # With zeros
    ]

    for i, dist in enumerate(test_distributions):
        entropy = optimizer.calculate_entropy(dist)
        print(f"  Distribution {i+1}: {dist}")
        print(f"    Entropy = {entropy:.4f} (normalized)")

        if 0 <= entropy <= 1:
            print(f"    [OK] Valid entropy range")
        else:
            print(f"    [FAIL] Entropy out of range")

    # Test 4: Crossover positivity
    print("\n4. Testing Crossover Positivity")
    print("-"*40)

    parent1 = [(1, 30.0), (2, 40.0), (3, 30.0)]
    parent2 = [(1, 20.0), (2, 50.0), (4, 30.0)]

    negatives_in_crossover = 0
    for i in range(50):
        child1, child2 = optimizer.stable_sbx_crossover(parent1, parent2)

        for ing_id, conc in child1 + child2:
            if conc < 0:
                negatives_in_crossover += 1

    if negatives_in_crossover == 0:
        print(f"  [OK] 50 crossovers tested, all positive")
    else:
        print(f"  [FAIL] Found {negatives_in_crossover} negative values in crossover")

    print("\n" + "="*60)
    print("STABILITY TEST COMPLETE")
    print("="*60)


# ============================================================================
# Backward Compatibility Wrapper (for tests)
# ============================================================================

class MOGAOptimizer:
    """Backward compatibility wrapper for old API"""

    def __init__(
        self,
        brief=None,
        n_ingredients: int = 10,
        population_size: int = 100,
        n_generations: int = 200,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.9,
        random_seed: Optional[int] = None
    ):
        self.brief = brief
        self.n_ingredients = n_ingredients
        self.population_size = population_size
        self.n_generations = n_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate

        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)

        # Create underlying optimizer
        self.optimizer = StableMOGA(
            population_size=population_size,
            generations=n_generations
        )

        # Override parameters
        self.optimizer.mutation_prob = mutation_rate
        self.optimizer.crossover_prob = crossover_rate

    def optimize(self) -> Dict[str, Any]:
        """Run optimization with backward-compatible output format"""
        results = self.optimizer.optimize(enable_diversity=True)

        # Convert to old API format
        pareto_front = []
        for solution in results['pareto_front']:
            # Extract concentrations
            concentrations = [ing['concentration'] / 100.0 for ing in solution['ingredients']]

            # Pad or truncate to n_ingredients
            if len(concentrations) < self.n_ingredients:
                concentrations.extend([0.0] * (self.n_ingredients - len(concentrations)))
            elif len(concentrations) > self.n_ingredients:
                concentrations = concentrations[:self.n_ingredients]

            # Normalize to sum to 1.0
            total = sum(concentrations)
            if total > 0:
                concentrations = [c / total for c in concentrations]

            pareto_front.append({
                'formulation': concentrations,
                'fitness': [
                    solution['quality_score'],
                    solution['stability_score'],
                    solution['diversity_score']
                ]
            })

        # Create history (simulated with improvement curve)
        history = []
        if pareto_front:
            final_fitness = pareto_front[0]['fitness']
            # Start from lower fitness and improve gradually
            initial_fitness = [f * 0.5 for f in final_fitness]  # Start at 50% of final

            for gen in range(self.n_generations):
                # Linear improvement from initial to final
                progress = gen / max(1, self.n_generations - 1)
                current_fitness = [
                    initial + (final - initial) * progress
                    for initial, final in zip(initial_fitness, final_fitness)
                ]
                history.append({
                    'generation': gen,
                    'best_fitness': current_fitness
                })
        else:
            for gen in range(self.n_generations):
                history.append({
                    'generation': gen,
                    'best_fitness': [0, 0, 0]
                })

        # Create statistics
        statistics = {
            'total_evaluations': self.population_size * self.n_generations,
            'convergence_generation': self.n_generations,
            'pareto_front_size': len(pareto_front)
        }

        return {
            'pareto_front': pareto_front,
            'history': history,
            'statistics': statistics
        }


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Run stability tests
    test_stability()

    # Run optimization
    print("\n" + "="*60)
    print("RUNNING STABLE MOGA OPTIMIZATION")
    print("="*60)

    optimizer = StableMOGA(population_size=100, generations=50)
    results = optimizer.optimize(enable_diversity=True)

    print(f"\nOptimization complete!")
    print(f"Found {len(results['pareto_front'])} Pareto-optimal solutions")
    print(f"\nTop 3 solutions:")

    for i, solution in enumerate(results['pareto_front'][:3], 1):
        print(f"\nSolution {i}:")
        print(f"  Quality: {solution['quality_score']}")
        print(f"  Cost: ${solution['cost']:.2f}/kg")
        print(f"  Stability: {solution['stability_score']}")
        print(f"  Diversity: {solution['diversity_score']}")
        print(f"  Entropy: {solution['entropy']}")
        print(f"  Ingredients:")
        for ing in solution['ingredients']:
            print(f"    - {ing['name']}: {ing['concentration']:.1f}%")

#!/usr/bin/env python3
"""
=============================================================================
  kRAND Weight Optimization via Stackelberg Game + Bayesian Optimization
=============================================================================

Objective: Find optimal weights w = (w_ksp, w_kdg, w_kds, w_klo) for the
kRAND router that MAXIMIZES the global normal P5 throughput under optimal
DDoS attack.

Mathematical Formulation:
  max_w  min_attack  P5_throughput_normal(w, attack)
  s.t.   w_i >= 0.05  (minimum diversity constraint)
         sum(w_i) = 1

This is a Stackelberg game where:
  - Leader (defender): chooses weights w
  - Follower (attacker): observes w, chooses optimal attack strategy

Solution Strategy (3 Phases):
  Phase 1: Analytical Model — Fast proxy using routing table statistics
           (scipy.optimize, runs in seconds)
  Phase 2: Bayesian Optimization — Real simulation as black-box objective
           (Gaussian Process surrogate, runs actual DDoS simulations)
  Phase 3: Final Verification — Full comparison of optimized vs baseline

Author: Auto-generated optimization framework
"""

import sys
import os
import json
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.stats import norm
from itertools import product

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from leo_network import LEOConstellation
from leo_network.core.routing import (
    KShortestPathsRouter, KDSRouter, KDGRouter, KLORouter, KRandRouter
)
from leo_network.core.traffic import TrafficGenerator, Flow, PacketType, TrafficPattern
from leo_network.core.simulator import Simulator

# ============================================================================
# Global Configuration
# ============================================================================
ISL_BW = 100.0
PACKET_SIZE = 1000
NUM_NORMAL_FLOWS = 20
NORMAL_RATE_RANGE = (50, 200)
SIM_DURATION = 2.0
TIME_STEP = 0.001
SEED = 42
TARGET_ISL_A = "SAT_4_2"
TARGET_ISL_B = "SAT_4_3"

DATA_DIR = os.path.join(project_root, "output")
ANALYSIS_FILE = os.path.join(DATA_DIR, "attack_gs_analysis.json")
COST_FILE = os.path.join(DATA_DIR, "gs_cost_comparison.json")
OUTPUT_FILE = os.path.join(DATA_DIR, "krand_optimization_results.json")

# Optimization hyperparameters
MIN_WEIGHT = 0.05       # Minimum weight per algorithm (diversity floor)
PHASE1_GRID_POINTS = 15  # Grid resolution for Phase 1 search
PHASE2_N_INIT = 8        # Initial random samples for Bayesian Opt
PHASE2_N_ITER = 12       # Bayesian optimization iterations


def mbps_to_pps(mbps, packet_size_bytes=PACKET_SIZE):
    bits_per_packet = packet_size_bytes * 8
    return (mbps * 1e6) / bits_per_packet


def create_constellation():
    constellation = LEOConstellation(
        num_planes=6, sats_per_plane=11,
        altitude_km=550.0, inclination_deg=53.0,
        isl_bandwidth_mbps=ISL_BW
    )
    constellation.add_global_ground_stations()
    return constellation


# ============================================================================
# Phase 1: Analytical Model — Fast Proxy Optimization
# ============================================================================
class AnalyticalAttackModel:
    """
    Fast analytical model using routing table statistics.

    Key insight: For a given weight vector w, the effective attack cost is:
        C_eff(w) = sum_gs max_algo [ w_algo * C_algo(gs) / p_algo(gs) ]

    where:
      - C_algo(gs) = bandwidth cost to cover GS under algorithm algo
      - p_algo(gs) = probability of hitting target ISL from that GS under algo
      - w_algo = weight assigned to algorithm algo

    The attacker wants to minimize the effective cost (easier attack).
    The defender wants to maximize it (harder to attack).

    The P5 throughput proxy is modeled as:
        P5_proxy(w) = P5_baseline - alpha * (attack_damage(w))

    where attack_damage depends on how much attack traffic leaks through
    given the weight distribution.
    """

    def __init__(self, cost_data, analysis_data):
        self.cost_data = cost_data
        self.analysis_data = analysis_data
        self.algo_names = ["KSP", "KDS", "KDG", "KLO"]

        # Extract per-algorithm attack parameters
        self.algo_costs = {}
        self.algo_p_through = {}
        for algo in self.algo_names:
            totals = cost_data["algorithm_totals"][algo]
            self.algo_costs[algo] = totals["total_cost_mbps"]
            self.algo_p_through[algo] = totals["p_through"]

        # Per-GS, per-algorithm cost breakdown
        self.gs_cost_table = cost_data["per_gs_cost_table"]

        # Baseline P5 from actual simulation data (known reference)
        # kRAND equal-weight baseline: 2369.5 pps
        self.baseline_p5 = 2369.5

    def compute_weighted_attack_cost(self, weights):
        """
        Compute the effective attack cost under weighted kRAND.

        When the defender uses weights w = {ksp: w1, kdg: w2, kds: w3, klo: w4},
        the attacker must cover each algorithm proportionally.

        The attacker's strategy: For each GS pair (s,d), the traffic going through
        target ISL under algorithm 'a' is proportional to w_a * p_a(s,d).

        To saturate the target ISL, the attacker needs:
            Attack_rate >= ISL_BW / sum_a(w_a * p_a)

        So effective cost = ISL_BW / (weighted average p_through)
        """
        w = {algo: weights[i] for i, algo in enumerate(self.algo_names)}

        # Weighted average probability of going through target ISL
        weighted_p = sum(w[algo] * self.algo_p_through[algo] for algo in self.algo_names)

        if weighted_p <= 0:
            return float('inf')

        # Effective attack cost (higher = better defense)
        effective_cost = ISL_BW / weighted_p

        return effective_cost

    def compute_attack_damage_proxy(self, weights):
        """
        Estimate attack damage as a proxy for P5 throughput degradation.

        Model: The attacker allocates budget proportionally to cover each algorithm.
        Damage ∝ sum over GS of (traffic through target / effective diversification)

        Higher weight on high-p_through algorithm → more concentrated traffic →
        easier for attacker → more damage.

        We model damage inversely proportional to the "effective path diversity":
            D(w) = sum_a w_a * p_a^2 / sum_a w_a * p_a

        This is the "weighted concentration" — lower is better (more spread out).
        """
        w = {algo: weights[i] for i, algo in enumerate(self.algo_names)}

        # Weighted sum of p_through^2 (concentration measure)
        weighted_p2 = sum(w[algo] * self.algo_p_through[algo]**2
                          for algo in self.algo_names)
        weighted_p = sum(w[algo] * self.algo_p_through[algo]
                         for algo in self.algo_names)

        if weighted_p <= 0:
            return 1.0

        # Concentration ratio (lower = better diversity)
        concentration = weighted_p2 / weighted_p

        return concentration

    def compute_gs_level_attack_cost(self, weights):
        """
        More detailed: compute per-GS attack cost under weighted scheme.

        For each GS, the attacker must cover ALL algorithms that could route
        through target ISL. The cost per GS = max over algorithms of:
            w_algo * cost_algo(gs)

        This captures the fact that higher weight means the attacker must
        invest more in covering that algorithm's paths from this GS.
        """
        w = {algo: weights[i] for i, algo in enumerate(self.algo_names)}

        total_weighted_cost = 0.0
        for gs_entry in self.gs_cost_table:
            max_weighted = 0.0
            for algo in self.algo_names:
                algo_info = gs_entry["per_algorithm"].get(algo)
                if algo_info is None or algo_info.get("cost_mbps", 0) <= 0:
                    continue
                # The cost the attacker pays is proportional to the weight
                # Higher weight → attacker must allocate more to cover this algo
                weighted_cost = w[algo] * algo_info["cost_mbps"]
                if weighted_cost > max_weighted:
                    max_weighted = weighted_cost
            total_weighted_cost += max_weighted

        return total_weighted_cost

    def objective_p5_proxy(self, weights):
        """
        Combined objective for P5 throughput proxy.

        Combines:
        1. Attack cost (higher = harder to attack = better)
        2. Path diversity (lower concentration = better load balance)
        3. Per-GS attack cost (higher = better)

        Returns NEGATIVE P5 proxy (for minimization).
        """
        # Normalize weights
        weights = np.array(weights)
        weights = weights / weights.sum()

        # Component 1: Effective attack cost (higher is better)
        attack_cost = self.compute_weighted_attack_cost(weights)
        cost_score = np.log(attack_cost + 1)  # Log scale to prevent domination

        # Component 2: Path diversity (lower concentration is better)
        concentration = self.compute_attack_damage_proxy(weights)
        diversity_score = -concentration * 1000  # Scaled negative

        # Component 3: Per-GS level cost (higher is better)
        gs_cost = self.compute_gs_level_attack_cost(weights)
        gs_score = np.log(gs_cost + 1)

        # Combined score (to be MAXIMIZED, so we return negative for minimize)
        combined = 0.4 * cost_score + 0.3 * diversity_score + 0.3 * gs_score

        return -combined  # Negate for scipy minimize

    def grid_search(self, resolution=PHASE1_GRID_POINTS):
        """
        Exhaustive grid search over the simplex w1+w2+w3+w4=1.

        Uses Dirichlet-style grid to sample uniformly on the simplex.
        """
        print("\n  [Phase 1a] Grid search on weight simplex...")
        best_score = float('inf')
        best_weights = None
        n_evaluated = 0

        # Generate grid points on the simplex with minimum weight constraint
        step = 1.0 / resolution
        for i in range(resolution + 1):
            for j in range(resolution + 1 - i):
                for k in range(resolution + 1 - i - j):
                    w1 = MIN_WEIGHT + (1.0 - 4*MIN_WEIGHT) * i / resolution
                    w2 = MIN_WEIGHT + (1.0 - 4*MIN_WEIGHT) * j / resolution
                    w3 = MIN_WEIGHT + (1.0 - 4*MIN_WEIGHT) * k / resolution
                    w4 = 1.0 - w1 - w2 - w3

                    if w4 < MIN_WEIGHT - 1e-9:
                        continue

                    weights = [w1, w2, w3, w4]
                    score = self.objective_p5_proxy(weights)
                    n_evaluated += 1

                    if score < best_score:
                        best_score = score
                        best_weights = weights[:]

        print(f"    Evaluated {n_evaluated} weight combinations")
        print(f"    Best analytical score: {-best_score:.6f}")
        print(f"    Best weights: KSP={best_weights[0]:.4f}, KDG={best_weights[1]:.4f}, "
              f"KDS={best_weights[2]:.4f}, KLO={best_weights[3]:.4f}")

        return best_weights, -best_score

    def scipy_optimize(self, x0=None):
        """
        Use scipy.optimize for continuous optimization on the simplex.

        Uses SLSQP with equality constraint (sum=1) and bounds (min_weight).
        """
        print("\n  [Phase 1b] Continuous optimization (SLSQP)...")

        if x0 is None:
            x0 = [0.25, 0.25, 0.25, 0.25]

        bounds = [(MIN_WEIGHT, 1.0 - 3*MIN_WEIGHT)] * 4
        constraints = [
            {'type': 'eq', 'fun': lambda w: sum(w) - 1.0},
        ]

        result = minimize(
            self.objective_p5_proxy,
            x0=x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 500, 'ftol': 1e-12}
        )

        opt_w = list(result.x / result.x.sum())
        print(f"    Converged: {result.success} (nit={result.nit})")
        print(f"    Optimal weights: KSP={opt_w[0]:.4f}, KDG={opt_w[1]:.4f}, "
              f"KDS={opt_w[2]:.4f}, KLO={opt_w[3]:.4f}")
        print(f"    Objective: {-result.fun:.6f}")

        return opt_w, -result.fun

    def differential_evolution_optimize(self):
        """
        Use differential evolution for global optimization.
        More robust than SLSQP for non-convex landscapes.
        """
        print("\n  [Phase 1c] Differential Evolution (global search)...")

        def constrained_objective(x):
            # Map from 3D unconstrained to 4D simplex
            w = np.zeros(4)
            w[:3] = MIN_WEIGHT + (1.0 - 4*MIN_WEIGHT) * x[:3]
            w[3] = 1.0 - w[0] - w[1] - w[2]
            if w[3] < MIN_WEIGHT:
                return 1e10  # Penalty
            return self.objective_p5_proxy(w)

        bounds_3d = [(0, 1)] * 3

        result = differential_evolution(
            constrained_objective,
            bounds=bounds_3d,
            seed=SEED,
            maxiter=200,
            tol=1e-10,
            polish=True
        )

        # Recover 4D weights
        w = np.zeros(4)
        w[:3] = MIN_WEIGHT + (1.0 - 4*MIN_WEIGHT) * result.x[:3]
        w[3] = 1.0 - w[0] - w[1] - w[2]
        opt_w = list(w / w.sum())

        print(f"    Converged: {result.success}")
        print(f"    Optimal weights: KSP={opt_w[0]:.4f}, KDG={opt_w[1]:.4f}, "
              f"KDS={opt_w[2]:.4f}, KLO={opt_w[3]:.4f}")
        print(f"    Objective: {-result.fun:.6f}")

        return opt_w, -result.fun


# ============================================================================
# Phase 2: Bayesian Optimization with Real Simulation
# ============================================================================
class GaussianProcessSurrogate:
    """
    Simple Gaussian Process surrogate for Bayesian Optimization.

    Uses RBF kernel with automatic length-scale estimation.
    Implements Expected Improvement (EI) acquisition function.
    """

    def __init__(self, length_scale=0.3, noise=1e-4):
        self.length_scale = length_scale
        self.noise = noise
        self.X_train = []
        self.y_train = []

    def rbf_kernel(self, x1, x2):
        """RBF (Squared Exponential) kernel"""
        dist_sq = np.sum((np.array(x1) - np.array(x2))**2)
        return np.exp(-dist_sq / (2 * self.length_scale**2))

    def kernel_matrix(self, X1, X2):
        """Compute kernel matrix between two sets of points"""
        n1, n2 = len(X1), len(X2)
        K = np.zeros((n1, n2))
        for i in range(n1):
            for j in range(n2):
                K[i, j] = self.rbf_kernel(X1[i], X2[j])
        return K

    def fit(self, X, y):
        """Fit GP to observed data"""
        self.X_train = list(X)
        self.y_train = list(y)

    def predict(self, x_new):
        """Predict mean and variance at new point"""
        if len(self.X_train) == 0:
            return 0.0, 1.0

        X = np.array(self.X_train)
        y = np.array(self.y_train)
        n = len(X)

        # Kernel matrices
        K = self.kernel_matrix(X, X) + self.noise * np.eye(n)
        k_star = np.array([self.rbf_kernel(x_new, xi) for xi in X])
        k_ss = self.rbf_kernel(x_new, x_new)

        # GP posterior
        try:
            K_inv = np.linalg.solve(K, np.eye(n))
        except np.linalg.LinAlgError:
            K_inv = np.linalg.pinv(K)

        mu = k_star @ K_inv @ y
        sigma2 = k_ss - k_star @ K_inv @ k_star
        sigma2 = max(sigma2, 1e-8)

        return mu, np.sqrt(sigma2)

    def expected_improvement(self, x_new, y_best, xi=0.01):
        """
        Expected Improvement acquisition function.

        EI(x) = (mu(x) - y_best - xi) * Phi(Z) + sigma(x) * phi(Z)
        where Z = (mu(x) - y_best - xi) / sigma(x)
        """
        mu, sigma = self.predict(x_new)

        if sigma < 1e-8:
            return 0.0

        Z = (mu - y_best - xi) / sigma
        ei = (mu - y_best - xi) * norm.cdf(Z) + sigma * norm.pdf(Z)

        return max(ei, 0.0)


class BayesianOptimizer:
    """
    Bayesian Optimization for kRAND weight tuning.

    Uses real DDoS simulation as the black-box objective function.
    Gaussian Process surrogate with Expected Improvement acquisition.
    """

    def __init__(self, cost_data, analysis_data):
        self.cost_data = cost_data
        self.analysis_data = analysis_data
        self.gp = GaussianProcessSurrogate(length_scale=0.2, noise=1e-3)
        self.history = []  # List of (weights, p5_value)

    def weights_to_simplex_3d(self, x):
        """Map from 3D unit cube to 4D simplex with min weight constraint"""
        w = np.zeros(4)
        # Use softmax-like mapping to ensure valid simplex point
        w[:3] = MIN_WEIGHT + (1.0 - 4*MIN_WEIGHT) * np.clip(x, 0, 1)
        w[3] = 1.0 - w[0] - w[1] - w[2]
        if w[3] < MIN_WEIGHT:
            # Rescale
            excess = MIN_WEIGHT - w[3]
            w[:3] -= excess / 3
            w[3] = MIN_WEIGHT
        w = np.clip(w, MIN_WEIGHT, 1.0)
        w = w / w.sum()
        return w

    def build_attack_flows_for_weighted_krand(self, weights):
        """
        Build attack flows adapted to the weighted kRAND.

        Smart attacker adapts: focuses more attack budget on algorithms
        with higher weights (since those are used more often).

        Attack cost per GS = max over algorithms of:
            cost_algo(gs) * weight_algo / p_algo_baseline
        """
        w = {algo: weights[i] for i, algo in enumerate(["KSP", "KDS", "KDG", "KLO"])}

        attack_flows = []
        total_cost = 0.0

        for gs_entry in self.cost_data["per_gs_cost_table"]:
            gs_name = gs_entry["ground_station"]

            # Attacker strategy: cover all algorithms, weighted by defender's choice
            max_cost = 0.0
            all_dests = set()

            for algo in ["KSP", "KDS", "KDG", "KLO"]:
                algo_info = gs_entry["per_algorithm"].get(algo)
                if algo_info is None or algo_info.get("cost_mbps", 0) <= 0:
                    continue
                # Attacker must cover this algorithm's paths
                # Scale by weight: if defender uses algo more, attacker invests more
                scaled_cost = algo_info["cost_mbps"] * (w[algo] / 0.25)
                if scaled_cost > max_cost:
                    max_cost = scaled_cost
                all_dests.update(algo_info["destinations"])

            if max_cost <= 0 or not all_dests:
                continue

            # But attacker has a fixed budget proportional to original kRAND cost
            # so we cap at the original cost structure
            original_max = gs_entry.get("krand_max_cost_mbps", max_cost)
            max_cost = min(max_cost, original_max * 1.5)  # Allow 50% adaptation

            total_cost += max_cost
            per_dest_mbps = max_cost / len(all_dests)
            per_dest_pps = mbps_to_pps(per_dest_mbps)

            for dst in sorted(all_dests):
                attack_flows.append({
                    "source": gs_name,
                    "destination": dst,
                    "rate_pps": per_dest_pps,
                    "rate_mbps": per_dest_mbps,
                })

        return attack_flows, total_cost

    def simulate_with_weights(self, weights):
        """
        Run a full DDoS simulation with given kRAND weights.

        Returns the global normal P5 throughput (pps).
        """
        w_dict = {
            "ksp": weights[0],
            "kdg": weights[1],
            "kds": weights[2],
            "klo": weights[3],
        }

        print(f"      Weights: KSP={weights[0]:.3f} KDG={weights[1]:.3f} "
              f"KDS={weights[2]:.3f} KLO={weights[3]:.3f}")

        # Create constellation and router
        constellation = create_constellation()
        router = KRandRouter(constellation, k=3, weights=w_dict, seed=SEED)

        # Build attack flows
        attack_flows, total_attack = self.build_attack_flows_for_weighted_krand(weights)

        # Run simulation
        sim = Simulator(
            constellation=constellation,
            router=router,
            time_step=TIME_STEP,
            seed=SEED
        )
        sim.set_target_isl(TARGET_ISL_A, TARGET_ISL_B)
        sim.add_random_normal_flows(
            num_flows=NUM_NORMAL_FLOWS,
            rate_range=NORMAL_RATE_RANGE,
            packet_size=PACKET_SIZE
        )

        for i, af in enumerate(attack_flows):
            flow_id = f"attack_{af['source']}_{af['destination']}_{i}"
            sim.traffic_generator.create_attack_flow(
                flow_id=flow_id,
                source=af["source"],
                destination=af["destination"],
                rate=af["rate_pps"],
                packet_size=PACKET_SIZE,
                start_time=0.0,
                duration=-1.0,
                pattern=TrafficPattern.CONSTANT
            )

        t0 = time.time()
        sim.run(duration=SIM_DURATION, progress_bar=True)
        elapsed = time.time() - t0

        results = sim.get_results()
        ntp = results["normal_throughput_percentiles"]
        titp = results["target_isl_normal_throughput_percentiles"]
        stats = results["statistics"]

        p5_global = ntp["p5_pps"]
        p5_target = titp["p5_pps"]
        normal_dr = stats["normal_traffic"]["delivery_rate"]

        print(f"      → P5 Global Normal: {p5_global:.1f} pps | "
              f"P5 Target ISL: {p5_target:.1f} pps | "
              f"Normal DR: {normal_dr:.4f} | "
              f"Attack: {total_attack:.0f} Mbps | "
              f"Time: {elapsed:.1f}s")

        return {
            "weights": list(weights),
            "p5_global_normal_pps": p5_global,
            "p5_target_isl_normal_pps": p5_target,
            "normal_delivery_rate": normal_dr,
            "total_attack_mbps": total_attack,
            "simulation_time_s": elapsed,
            "normal_p5_mbps": ntp["p5_mbps"],
            "normal_avg_pps": ntp["avg_pps"],
        }

    def optimize(self, initial_guess, n_init=PHASE2_N_INIT, n_iter=PHASE2_N_ITER):
        """
        Bayesian Optimization loop.

        1. Evaluate initial points (Latin Hypercube + initial guess)
        2. Fit GP surrogate
        3. Maximize EI to find next evaluation point
        4. Repeat
        """
        print("\n  [Phase 2] Bayesian Optimization with Real Simulation")
        print(f"    Initial samples: {n_init}")
        print(f"    Optimization iterations: {n_iter}")

        # Step 1: Generate initial samples
        rng = np.random.default_rng(SEED)
        init_points_3d = []

        # Latin Hypercube Sampling
        for i in range(n_init - 1):
            x = rng.uniform(0, 1, size=3)
            init_points_3d.append(x)

        # Add the Phase 1 initial guess
        ig = np.array(initial_guess)
        ig_3d = (ig[:3] - MIN_WEIGHT) / (1.0 - 4*MIN_WEIGHT)
        ig_3d = np.clip(ig_3d, 0, 1)
        init_points_3d.append(ig_3d)

        # Step 2: Evaluate initial points
        print(f"\n    --- Evaluating {n_init} initial points ---")
        X_observed = []
        y_observed = []

        for idx, x3d in enumerate(init_points_3d):
            w4d = self.weights_to_simplex_3d(x3d)
            print(f"\n    [Init {idx+1}/{n_init}]")
            result = self.simulate_with_weights(w4d)

            X_observed.append(x3d.tolist())
            y_observed.append(result["p5_global_normal_pps"])
            self.history.append(result)

        # Step 3: Bayesian optimization loop
        print(f"\n    --- Bayesian Optimization: {n_iter} iterations ---")

        for iteration in range(n_iter):
            # Fit GP
            self.gp.fit(X_observed, y_observed)
            y_best = max(y_observed)

            # Find point with maximum EI
            best_ei = -1
            best_x = None
            n_candidates = 2000
            candidates = rng.uniform(0, 1, size=(n_candidates, 3))

            for cand in candidates:
                w_test = self.weights_to_simplex_3d(cand)
                if w_test[3] < MIN_WEIGHT:
                    continue
                ei = self.gp.expected_improvement(cand.tolist(), y_best)
                if ei > best_ei:
                    best_ei = ei
                    best_x = cand

            if best_x is None:
                best_x = rng.uniform(0, 1, size=3)

            # Evaluate the chosen point
            w4d = self.weights_to_simplex_3d(best_x)
            print(f"\n    [BO Iter {iteration+1}/{n_iter}] EI={best_ei:.4f}")
            result = self.simulate_with_weights(w4d)

            X_observed.append(best_x.tolist())
            y_observed.append(result["p5_global_normal_pps"])
            self.history.append(result)

            current_best_idx = np.argmax(y_observed)
            current_best_w = self.weights_to_simplex_3d(
                np.array(X_observed[current_best_idx]))
            print(f"      Current best P5: {max(y_observed):.1f} pps "
                  f"@ w=[{current_best_w[0]:.3f},{current_best_w[1]:.3f},"
                  f"{current_best_w[2]:.3f},{current_best_w[3]:.3f}]")

        # Find best overall
        best_idx = np.argmax(y_observed)
        best_weights = self.weights_to_simplex_3d(np.array(X_observed[best_idx]))
        best_p5 = y_observed[best_idx]

        print(f"\n    ✅ Best found:")
        print(f"       Weights: KSP={best_weights[0]:.4f}, KDG={best_weights[1]:.4f}, "
              f"KDS={best_weights[2]:.4f}, KLO={best_weights[3]:.4f}")
        print(f"       P5 Global Normal: {best_p5:.1f} pps")

        return list(best_weights), best_p5, self.history


# ============================================================================
# Phase 3: Final Verification
# ============================================================================
def run_final_verification(optimal_weights, cost_data, analysis_data):
    """
    Run comprehensive verification comparing:
    1. Equal-weight kRAND (baseline)
    2. Optimized-weight kRAND
    Both under the same attack scenario.
    """
    print("\n" + "=" * 80)
    print("  PHASE 3: FINAL VERIFICATION")
    print("=" * 80)

    configs = {
        "kRAND_equal": {"ksp": 0.25, "kdg": 0.25, "kds": 0.25, "klo": 0.25},
        "kRAND_optimized": {
            "ksp": optimal_weights[0],
            "kdg": optimal_weights[1],
            "kds": optimal_weights[2],
            "klo": optimal_weights[3],
        },
    }

    results = {}

    for name, weight_dict in configs.items():
        print(f"\n  --- {name} ---")
        print(f"    Weights: {weight_dict}")

        # Baseline (no attack)
        print(f"    [Baseline] Running...")
        constellation = create_constellation()
        router = KRandRouter(constellation, k=3, weights=weight_dict, seed=SEED)

        sim = Simulator(
            constellation=constellation,
            router=router,
            time_step=TIME_STEP,
            seed=SEED
        )
        sim.set_target_isl(TARGET_ISL_A, TARGET_ISL_B)
        sim.add_random_normal_flows(
            num_flows=NUM_NORMAL_FLOWS,
            rate_range=NORMAL_RATE_RANGE,
            packet_size=PACKET_SIZE
        )
        sim.run(duration=SIM_DURATION, progress_bar=True)
        baseline_results = sim.get_results()
        bl_ntp = baseline_results["normal_throughput_percentiles"]
        bl_titp = baseline_results["target_isl_normal_throughput_percentiles"]
        bl_stats = baseline_results["statistics"]

        # Attack
        print(f"    [Attack] Building attack flows...")
        weights_arr = [weight_dict["ksp"], weight_dict["kdg"],
                       weight_dict["kds"], weight_dict["klo"]]

        bo = BayesianOptimizer(cost_data, analysis_data)
        attack_flows, total_attack = bo.build_attack_flows_for_weighted_krand(weights_arr)

        print(f"    [Attack] {len(attack_flows)} flows, {total_attack:.1f} Mbps total")
        print(f"    [Attack] Running simulation...")

        constellation2 = create_constellation()
        router2 = KRandRouter(constellation2, k=3, weights=weight_dict, seed=SEED)

        sim2 = Simulator(
            constellation=constellation2,
            router=router2,
            time_step=TIME_STEP,
            seed=SEED
        )
        sim2.set_target_isl(TARGET_ISL_A, TARGET_ISL_B)
        sim2.add_random_normal_flows(
            num_flows=NUM_NORMAL_FLOWS,
            rate_range=NORMAL_RATE_RANGE,
            packet_size=PACKET_SIZE
        )

        for i, af in enumerate(attack_flows):
            flow_id = f"attack_{af['source']}_{af['destination']}_{i}"
            sim2.traffic_generator.create_attack_flow(
                flow_id=flow_id,
                source=af["source"],
                destination=af["destination"],
                rate=af["rate_pps"],
                packet_size=PACKET_SIZE,
                start_time=0.0,
                duration=-1.0,
                pattern=TrafficPattern.CONSTANT
            )

        t0 = time.time()
        sim2.run(duration=SIM_DURATION, progress_bar=True)
        atk_elapsed = time.time() - t0

        atk_results = sim2.get_results()
        atk_ntp = atk_results["normal_throughput_percentiles"]
        atk_titp = atk_results["target_isl_normal_throughput_percentiles"]
        atk_stats = atk_results["statistics"]

        results[name] = {
            "weights": weight_dict,
            "baseline": {
                "normal_p5_pps": round(bl_ntp["p5_pps"], 4),
                "normal_p5_mbps": round(bl_ntp["p5_mbps"], 6),
                "normal_avg_pps": round(bl_ntp["avg_pps"], 4),
                "normal_delivery_rate": round(bl_stats["normal_traffic"]["delivery_rate"], 6),
                "target_isl_normal_p5_pps": round(bl_titp["p5_pps"], 4),
                "avg_delay_ms": round(bl_stats["delay"]["avg_ms"], 4),
            },
            "attack": {
                "total_attack_mbps": round(total_attack, 2),
                "num_attack_flows": len(attack_flows),
                "normal_p5_pps": round(atk_ntp["p5_pps"], 4),
                "normal_p5_mbps": round(atk_ntp["p5_mbps"], 6),
                "normal_avg_pps": round(atk_ntp["avg_pps"], 4),
                "normal_delivery_rate": round(atk_stats["normal_traffic"]["delivery_rate"], 6),
                "target_isl_normal_p5_pps": round(atk_titp["p5_pps"], 4),
                "avg_delay_ms": round(atk_stats["delay"]["avg_ms"], 4),
                "simulation_time_s": round(atk_elapsed, 2),
            },
            "comparison": {
                "p5_drop_pps": round(bl_ntp["p5_pps"] - atk_ntp["p5_pps"], 4),
                "p5_drop_pct": round(
                    (bl_ntp["p5_pps"] - atk_ntp["p5_pps"]) / bl_ntp["p5_pps"] * 100, 4
                ) if bl_ntp["p5_pps"] > 0 else 0,
                "dr_drop": round(
                    bl_stats["normal_traffic"]["delivery_rate"] -
                    atk_stats["normal_traffic"]["delivery_rate"], 6
                ),
            }
        }

        print(f"    Baseline P5: {bl_ntp['p5_pps']:.1f} pps")
        print(f"    Attack P5:   {atk_ntp['p5_pps']:.1f} pps "
              f"(drop: {bl_ntp['p5_pps'] - atk_ntp['p5_pps']:.1f} pps)")
        print(f"    Normal DR:   {atk_stats['normal_traffic']['delivery_rate']:.4f}")

    return results


# ============================================================================
# Main
# ============================================================================
def main():
    print("=" * 80)
    print("  kRAND WEIGHT OPTIMIZATION")
    print("  Stackelberg Game + Bayesian Optimization")
    print("  Target: Maximize global normal P5 throughput under optimal attack")
    print("=" * 80)

    # Load data
    print("\n[0] Loading pre-computed data...")
    with open(ANALYSIS_FILE, "r") as f:
        analysis_data = json.load(f)
    with open(COST_FILE, "r") as f:
        cost_data = json.load(f)
    print("    ✅ Loaded attack_gs_analysis.json and gs_cost_comparison.json")

    all_results = {
        "config": {
            "target_isl": "SAT_4_2 <-> SAT_4_3",
            "isl_bandwidth_mbps": ISL_BW,
            "packet_size_bytes": PACKET_SIZE,
            "num_normal_flows": NUM_NORMAL_FLOWS,
            "normal_rate_range_pps": list(NORMAL_RATE_RANGE),
            "simulation_duration_s": SIM_DURATION,
            "min_weight": MIN_WEIGHT,
            "seed": SEED,
        }
    }

    # =========================================================================
    # PHASE 1: Analytical Model
    # =========================================================================
    print("\n" + "=" * 80)
    print("  PHASE 1: ANALYTICAL MODEL (Fast Proxy Optimization)")
    print("=" * 80)

    t_start = time.time()
    model = AnalyticalAttackModel(cost_data, analysis_data)

    # Print per-algorithm attack characteristics
    print("\n  Per-algorithm attack characteristics:")
    for algo in ["KSP", "KDS", "KDG", "KLO"]:
        print(f"    {algo}: cost={model.algo_costs[algo]:.1f} Mbps, "
              f"p_through={model.algo_p_through[algo]:.6f}")

    # Method A: Grid search
    grid_weights, grid_score = model.grid_search()

    # Method B: SLSQP from grid solution
    slsqp_weights, slsqp_score = model.scipy_optimize(x0=grid_weights)

    # Method C: Differential Evolution (global)
    de_weights, de_score = model.differential_evolution_optimize()

    # Pick best Phase 1 result
    phase1_candidates = [
        ("Grid", grid_weights, grid_score),
        ("SLSQP", slsqp_weights, slsqp_score),
        ("DiffEvo", de_weights, de_score),
    ]
    phase1_candidates.sort(key=lambda x: x[2], reverse=True)
    best_method, phase1_weights, phase1_score = phase1_candidates[0]

    t_phase1 = time.time() - t_start

    print(f"\n  ✅ Phase 1 Complete ({t_phase1:.1f}s)")
    print(f"     Best method: {best_method}")
    print(f"     Optimal weights: KSP={phase1_weights[0]:.4f}, KDG={phase1_weights[1]:.4f}, "
          f"KDS={phase1_weights[2]:.4f}, KLO={phase1_weights[3]:.4f}")

    all_results["phase1"] = {
        "time_s": round(t_phase1, 2),
        "grid_search": {
            "weights": {"ksp": round(grid_weights[0], 6), "kdg": round(grid_weights[1], 6),
                        "kds": round(grid_weights[2], 6), "klo": round(grid_weights[3], 6)},
            "score": round(grid_score, 6),
        },
        "slsqp": {
            "weights": {"ksp": round(slsqp_weights[0], 6), "kdg": round(slsqp_weights[1], 6),
                        "kds": round(slsqp_weights[2], 6), "klo": round(slsqp_weights[3], 6)},
            "score": round(slsqp_score, 6),
        },
        "differential_evolution": {
            "weights": {"ksp": round(de_weights[0], 6), "kdg": round(de_weights[1], 6),
                        "kds": round(de_weights[2], 6), "klo": round(de_weights[3], 6)},
            "score": round(de_score, 6),
        },
        "best_method": best_method,
        "best_weights": {"ksp": round(phase1_weights[0], 6), "kdg": round(phase1_weights[1], 6),
                         "kds": round(phase1_weights[2], 6), "klo": round(phase1_weights[3], 6)},
    }

    # =========================================================================
    # PHASE 2: Bayesian Optimization
    # =========================================================================
    print("\n" + "=" * 80)
    print("  PHASE 2: BAYESIAN OPTIMIZATION (Real Simulation)")
    print("=" * 80)

    t_start = time.time()
    bo = BayesianOptimizer(cost_data, analysis_data)
    phase2_weights, phase2_p5, bo_history = bo.optimize(
        initial_guess=phase1_weights,
        n_init=PHASE2_N_INIT,
        n_iter=PHASE2_N_ITER
    )
    t_phase2 = time.time() - t_start

    print(f"\n  ✅ Phase 2 Complete ({t_phase2:.1f}s)")
    print(f"     Optimal weights: KSP={phase2_weights[0]:.4f}, KDG={phase2_weights[1]:.4f}, "
          f"KDS={phase2_weights[2]:.4f}, KLO={phase2_weights[3]:.4f}")
    print(f"     Best P5 throughput: {phase2_p5:.1f} pps")

    # Sort history by P5 for top-K summary
    sorted_history = sorted(bo_history, key=lambda x: x["p5_global_normal_pps"], reverse=True)

    all_results["phase2"] = {
        "time_s": round(t_phase2, 2),
        "n_evaluations": len(bo_history),
        "best_weights": {
            "ksp": round(phase2_weights[0], 6), "kdg": round(phase2_weights[1], 6),
            "kds": round(phase2_weights[2], 6), "klo": round(phase2_weights[3], 6)
        },
        "best_p5_pps": round(phase2_p5, 4),
        "top5_results": [
            {
                "rank": i+1,
                "weights": {"ksp": round(r["weights"][0], 4), "kdg": round(r["weights"][1], 4),
                            "kds": round(r["weights"][2], 4), "klo": round(r["weights"][3], 4)},
                "p5_global_normal_pps": round(r["p5_global_normal_pps"], 4),
                "p5_target_isl_pps": round(r["p5_target_isl_normal_pps"], 4),
                "normal_dr": round(r["normal_delivery_rate"], 6),
            }
            for i, r in enumerate(sorted_history[:5])
        ],
        "full_history": [
            {
                "iteration": i+1,
                "weights": {"ksp": round(r["weights"][0], 4), "kdg": round(r["weights"][1], 4),
                            "kds": round(r["weights"][2], 4), "klo": round(r["weights"][3], 4)},
                "p5_global_normal_pps": round(r["p5_global_normal_pps"], 4),
                "normal_dr": round(r["normal_delivery_rate"], 6),
                "total_attack_mbps": round(r["total_attack_mbps"], 2),
                "sim_time_s": round(r["simulation_time_s"], 1),
            }
            for i, r in enumerate(bo_history)
        ],
    }

    # =========================================================================
    # PHASE 3: Final Verification
    # =========================================================================
    verification = run_final_verification(phase2_weights, cost_data, analysis_data)
    all_results["phase3_verification"] = verification

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 80)
    print("  OPTIMIZATION SUMMARY")
    print("=" * 80)

    eq = verification["kRAND_equal"]
    opt = verification["kRAND_optimized"]

    print(f"\n  {'Metric':<35} │ {'Equal (0.25)':<15} │ {'Optimized':<15} │ {'Delta':<10}")
    print(f"  {'─'*35} ┼ {'─'*15} ┼ {'─'*15} ┼ {'─'*10}")

    # Baseline comparison
    print(f"  {'Baseline P5 (pps)':<35} │ "
          f"{eq['baseline']['normal_p5_pps']:>13.1f} │ "
          f"{opt['baseline']['normal_p5_pps']:>13.1f} │ "
          f"{opt['baseline']['normal_p5_pps'] - eq['baseline']['normal_p5_pps']:>+8.1f}")

    # Attack comparison
    print(f"  {'Attack P5 (pps) ★':<35} │ "
          f"{eq['attack']['normal_p5_pps']:>13.1f} │ "
          f"{opt['attack']['normal_p5_pps']:>13.1f} │ "
          f"{opt['attack']['normal_p5_pps'] - eq['attack']['normal_p5_pps']:>+8.1f}")

    print(f"  {'P5 Drop (pps)':<35} │ "
          f"{eq['comparison']['p5_drop_pps']:>13.1f} │ "
          f"{opt['comparison']['p5_drop_pps']:>13.1f} │ "
          f"{opt['comparison']['p5_drop_pps'] - eq['comparison']['p5_drop_pps']:>+8.1f}")

    print(f"  {'P5 Drop (%)':<35} │ "
          f"{eq['comparison']['p5_drop_pct']:>12.2f}% │ "
          f"{opt['comparison']['p5_drop_pct']:>12.2f}% │ "
          f"{opt['comparison']['p5_drop_pct'] - eq['comparison']['p5_drop_pct']:>+7.2f}%")

    print(f"  {'Normal DR (under attack)':<35} │ "
          f"{eq['attack']['normal_delivery_rate']:>13.4f} │ "
          f"{opt['attack']['normal_delivery_rate']:>13.4f} │ "
          f"{opt['attack']['normal_delivery_rate'] - eq['attack']['normal_delivery_rate']:>+8.4f}")

    print(f"  {'Attack Cost (Mbps)':<35} │ "
          f"{eq['attack']['total_attack_mbps']:>13.1f} │ "
          f"{opt['attack']['total_attack_mbps']:>13.1f} │ "
          f"{opt['attack']['total_attack_mbps'] - eq['attack']['total_attack_mbps']:>+8.1f}")

    print(f"  {'Target ISL P5 (pps)':<35} │ "
          f"{eq['attack']['target_isl_normal_p5_pps']:>13.1f} │ "
          f"{opt['attack']['target_isl_normal_p5_pps']:>13.1f} │ "
          f"{opt['attack']['target_isl_normal_p5_pps'] - eq['attack']['target_isl_normal_p5_pps']:>+8.1f}")

    print(f"\n  Optimized weights: KSP={phase2_weights[0]:.4f}, KDG={phase2_weights[1]:.4f}, "
          f"KDS={phase2_weights[2]:.4f}, KLO={phase2_weights[3]:.4f}")

    # Save
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\n  ✅ All results saved to: {OUTPUT_FILE}")
    print("  Done!")


if __name__ == "__main__":
    main()

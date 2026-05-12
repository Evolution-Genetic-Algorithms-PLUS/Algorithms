import numpy as np
import matplotlib.pyplot as plt
import time

class DeceptiveTrapCGA:
    def __init__(self, length: int = 100, k: int = 5, population_size: int = 1000):
        self.length = length
        self.k = k
        self.m = length // k
        self.n = population_size
        
        # Ensure length is perfectly divisible by k
        if self.length % self.k != 0:
            raise ValueError("Length L must be perfectly divisible by block size k.")
            
        # Initialize probability vector to 0.5 for all bits
        self.p = np.ones(self.length, dtype=np.float64) * 0.5

    def calculate_fitness(self, bitstring: np.ndarray) -> int:
        # Reshape the 1D bitstring into a 2D array of m blocks, each of k bits
        blocks = bitstring.reshape(self.m, self.k)
        
        # Calculate u (number of ones) for each block
        u = np.sum(blocks, axis=1)
        
        # Apply the deceptive trap logic: 
        # If u == k, fitness is k. Else, fitness is k - u - 1
        block_fitness = np.where(u == self.k, self.k, self.k - u - 1)
        
        # Total fitness is the sum of all block fitnesses
        return np.sum(block_fitness)

    def generate(self) -> np.ndarray:
        return (np.random.random(self.length) < self.p).astype(int)

    def compete(self, a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        score_a = self.calculate_fitness(a)
        score_b = self.calculate_fitness(b)
        # Maximization problem
        return (a, b) if score_a >= score_b else (b, a)

    def update(self, winner: np.ndarray, loser: np.ndarray) -> None:
        for i in range(self.length):
            if winner[i] != loser[i]:
                if winner[i] == 1:
                    self.p[i] += 1 / self.n
                else:
                    self.p[i] -= 1 / self.n
                self.p[i] = np.clip(self.p[i], 0.0, 1.0)

class BenchmarkableDeceptiveTrapCGA(DeceptiveTrapCGA):
    def execute_with_metrics(self, max_iter=20000):
        start_time = time.time()
        converged_iteration = max_iter
        fitness_history = []

        for i in range(max_iter):
            a, b = self.generate(), self.generate()
            winner, loser = self.compete(a, b)
            self.update(winner, loser)

            # Record fitness every 50 iterations to save computation time over a longer run
            if i % 50 == 0:
                best_bits = (self.p > 0.5).astype(int)
                fitness_history.append(self.calculate_fitness(best_bits))

            # Check for full convergence
            if np.all((self.p <= 0.0) | (self.p >= 1.0)):
                converged_iteration = i
                best_bits = (self.p > 0.5).astype(int)
                final_fit = self.calculate_fitness(best_bits)
                while len(fitness_history) < max_iter // 50:
                    fitness_history.append(final_fit)
                break

        execution_time = time.time() - start_time
        best_bits = (self.p > 0.5).astype(int)
        final_fitness = self.calculate_fitness(best_bits)

        while len(fitness_history) < max_iter // 50:
            fitness_history.append(final_fitness)

        return {
            "iterations": converged_iteration,
            "final_fitness": final_fitness,
            "time": execution_time,
            "fitness_history": np.array(fitness_history)
        }

def run_visualized_benchmark(length=100, k=5, pop_size=1000, max_iter=20000, runs=30):
    print(f"--- Running Deceptive Trap Benchmark (L={length}, k={k}, Pop={pop_size}) over {runs} Runs ---")
    
    # The global maximum is all 1s (fitness = m * k = 100)
    # The deceptive local maximum is all 0s (fitness = m * (k-1) = 20 * 4 = 80)
    
    fitnesses = []
    iterations = []
    histories = []

    for seed in range(runs):
        if (seed + 1) % 5 == 0 or seed == 0:
            print(f"Run {seed+1}/{runs}...")
            
        np.random.seed(seed)
        cga = BenchmarkableDeceptiveTrapCGA(length=length, k=k, population_size=pop_size)
        metrics = cga.execute_with_metrics(max_iter=max_iter)

        fitnesses.append(metrics["final_fitness"])
        iterations.append(metrics["iterations"])
        histories.append(metrics["fitness_history"])

    fitnesses = np.array(fitnesses)
    iterations = np.array(iterations)
    histories = np.array(histories)

    # --- PLOT 1: Fitness vs Iterations ---
    plt.figure(figsize=(8, 5))
    mean_history = np.mean(histories, axis=0)
    std_history = np.std(histories, axis=0)
    x_axis = np.arange(0, max_iter, 50)

    plt.plot(x_axis, mean_history, label="Mean Fitness", color='#d62728')
    plt.fill_between(x_axis, mean_history - std_history, mean_history + std_history, alpha=0.3, color='#d62728')
    
    plt.axhline(y=100, color='g', linestyle='-', linewidth=2, label="Global Optimum (100)")
    plt.axhline(y=80, color='orange', linestyle='--', linewidth=2, label="Deceptive Local Optimum (80)")
    
    plt.title(f"Deceptive Trap: Fitness vs Iterations (Pop={pop_size})", fontsize=14)
    plt.xlabel("Iterations", fontsize=12)
    plt.ylabel("Fitness Score", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("trap_fitness_vs_iterations.png")

    # --- PLOT 2: Final Fitness Distribution ---
    plt.figure(figsize=(6, 5))
    plt.boxplot([fitnesses], labels=["Deceptive Trap"])
    plt.axhline(y=100, color='g', linestyle='-', alpha=0.5)
    plt.axhline(y=80, color='orange', linestyle='--', alpha=0.5)
    plt.title(f"Final Fitness Distribution (Pop={pop_size})", fontsize=14)
    plt.ylabel("Final Fitness Score", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("trap_final_fitness_distribution.png")

    print("\nVisualizations saved successfully!")

if __name__ == "__main__":
    # NOTE: Set pop_size higher (e.g., 1000 or 2000) to see it solve the trap.
    # Set pop_size lower (e.g., 50 or 100) to watch it fall into the deceptive local optimum of 80.
    run_visualized_benchmark(length=100, k=5, pop_size=1000, max_iter=20000, runs=30)

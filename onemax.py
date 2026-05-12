import numpy as np
import matplotlib.pyplot as plt
import time

class OneMaxCGA:
    def __init__(self, length: int = 100, population_size: int = 100):
        self.length = length
        self.n = population_size
        # Initialize probability vector to 0.5 for all bits
        self.p = np.ones(self.length, dtype=np.float64) * 0.5

    def calculate_fitness(self, bitstring: np.ndarray) -> int:
        # Fitness is simply the sum of 1s in the bitstring
        return np.sum(bitstring)

    def generate(self) -> np.ndarray:
        # Generate bitstring based on probability vector p
        return (np.random.random(self.length) < self.p).astype(int)

    def compete(self, a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        score_a = self.calculate_fitness(a)
        score_b = self.calculate_fitness(b)
        # Maximization problem: higher score wins
        return (a, b) if score_a >= score_b else (b, a)

    def update(self, winner: np.ndarray, loser: np.ndarray) -> None:
        for i in range(self.length):
            if winner[i] != loser[i]:
                if winner[i] == 1:
                    self.p[i] += 1 / self.n
                else:
                    self.p[i] -= 1 / self.n
                # Ensure probabilities stay strictly between 0 and 1
                self.p[i] = np.clip(self.p[i], 0.0, 1.0)

class BenchmarkableOneMaxCGA(OneMaxCGA):
    def execute_with_metrics(self, max_iter=5000):
        start_time = time.time()
        converged_iteration = max_iter

        # Track fitness history for the line plot
        fitness_history = []

        for i in range(max_iter):
            a, b = self.generate(), self.generate()
            winner, loser = self.compete(a, b)
            self.update(winner, loser)

            # Record fitness every 10 iterations to save computation time
            if i % 10 == 0:
                best_bits = (self.p > 0.5).astype(int)
                fitness_history.append(self.calculate_fitness(best_bits))

            # Check for convergence (all probabilities are 0 or 1)
            if np.all((self.p <= 0.0) | (self.p >= 1.0)):
                converged_iteration = i

                # Pad the rest of the history so arrays are equal length for plotting
                best_bits = (self.p > 0.5).astype(int)
                final_fit = self.calculate_fitness(best_bits)
                while len(fitness_history) < max_iter // 10:
                    fitness_history.append(final_fit)
                break

        execution_time = time.time() - start_time
        best_bits = (self.p > 0.5).astype(int)
        final_fitness = self.calculate_fitness(best_bits)

        # Ensure array is full if it exits naturally without full convergence
        while len(fitness_history) < max_iter // 10:
            fitness_history.append(final_fitness)

        return {
            "iterations": converged_iteration,
            "final_fitness": final_fitness,
            "time": execution_time,
            "fitness_history": np.array(fitness_history)
        }

def run_visualized_benchmark(length=100, pop_size=100, max_iter=5000, runs=30):
    print(f"--- Running Visualized Benchmark (OneMax L={length}) over {runs} Runs ---")

    fitnesses = []
    iterations = []
    histories = []

    # 30 Independent Runs
    for seed in range(runs):
        print(f"Run {seed+1}/{runs}...")
        np.random.seed(seed)
        cga = BenchmarkableOneMaxCGA(length=length, population_size=pop_size)
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
    x_axis = np.arange(0, max_iter, 10)

    plt.plot(x_axis, mean_history, label="Mean Fitness (Higher is better)", color='#2ca02c')
    plt.fill_between(x_axis, mean_history - std_history, mean_history + std_history, alpha=0.3, color='#2ca02c')
    plt.axhline(y=length, color='r', linestyle='--', label="Optimal Target (100)")
    plt.title("OneMax: Fitness vs Iterations", fontsize=14)
    plt.xlabel("Iterations", fontsize=12)
    plt.ylabel("Fitness Score", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig("onemax_fitness_vs_iterations.png")

    # --- PLOT 2: Final Fitness Distribution ---
    plt.figure(figsize=(6, 5))
    plt.boxplot([fitnesses], labels=["OneMax"])
    plt.title("Final Fitness Distribution", fontsize=14)
    plt.ylabel("Final Fitness Score (Max 100)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("onemax_final_fitness_distribution.png")

    # --- PLOT 3: Convergence Speed Distribution ---
    plt.figure(figsize=(6, 5))
    plt.boxplot([iterations], labels=["OneMax"])
    plt.title("Convergence Speed Distribution", fontsize=14)
    plt.ylabel("Iterations to Convergence", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("onemax_convergence_speed_distribution.png")

    print("\nVisualizations saved successfully!")

if __name__ == "__main__":
    # Executing the sanity check benchmark
    run_visualized_benchmark(length=100, pop_size=100, max_iter=5000, runs=30)

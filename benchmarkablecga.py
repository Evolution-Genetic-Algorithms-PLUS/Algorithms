import time
import numpy as np
import matplotlib.pyplot as plt
from circlepacking import CirclePackingCGA

class BenchmarkableCGA(CirclePackingCGA):
    def execute_with_metrics(self, max_iter=2000):
        start_time = time.time()
        converged_iteration = max_iter

        # Track fitness history and vector history for the line plot
        fitness_history = []
        vector_history = []

        for i in range(max_iter):
            a, b = self.generate(), self.generate()
            winner, loser = self.compete(a, b)
            self.update(winner, loser)

            # Record fitness and vector every 10 iterations to save heavy computation time
            if i % 10 == 0:
                best_bits = (self.p > 0.5).astype(int)
                fitness_history.append(self.calculate_fitness(best_bits))
                vector_history.append(best_bits.copy())

            if np.all((self.p == 0) | (self.p == 1)):
                converged_iteration = i

                # Pad the rest of the history so arrays are equal length for plotting
                best_bits = (self.p > 0.5).astype(int)
                final_fit = self.calculate_fitness(best_bits)
                while len(fitness_history) < max_iter // 10:
                    fitness_history.append(final_fit)
                    vector_history.append(best_bits.copy())
                break

        execution_time = time.time() - start_time
        best_bits = (self.p > 0.5).astype(int)
        final_penalty = self.calculate_fitness(best_bits)

        # Ensure array is full if it exits naturally
        while len(fitness_history) < max_iter // 10:
            fitness_history.append(final_penalty)
            vector_history.append(best_bits.copy())

        return {
            "iterations": converged_iteration,
            "final_penalty": final_penalty,
            "final_vector": best_bits,
            "time": execution_time,
            "fitness_history": np.array(fitness_history),
            "vector_history": np.array(vector_history)
        }

def run_visualized_benchmark(num_circles=10, radius=0.1, pop_size=100, max_iter=10000, runs=30):
    print(f"--- Running Visualized Benchmark (N={num_circles}, R={radius}) over {runs} Runs ---")

    penalties = []
    iterations = []
    histories = []

    # 30 Independent Runs
    for seed in range(runs):
        print(f"Run {seed+1}/{runs}...")
        np.random.seed(seed)
        cga = BenchmarkableCGA(num_circles=num_circles, radius=radius, population_size=pop_size)
        metrics = cga.execute_with_metrics(max_iter=max_iter)

        penalties.append(metrics["final_penalty"])
        iterations.append(metrics["iterations"])
        histories.append(metrics["fitness_history"])

    penalties = np.array(penalties)
    iterations = np.array(iterations)
    histories = np.array(histories)

    # --- PLOT 1: Fitness vs Iterations ---
    plt.figure(figsize=(8, 5))
    mean_history = np.mean(histories, axis=0)
    std_history = np.std(histories, axis=0)
    x_axis = np.arange(0, max_iter, 10)

    plt.plot(x_axis, mean_history, label="Mean Penalty (Lower is better)", color='#1f77b4')
    plt.fill_between(x_axis, mean_history - std_history, mean_history + std_history, alpha=0.3, color='#1f77b4')
    plt.title("Circle Packing: Penalty vs Iterations", fontsize=14)
    plt.xlabel("Iterations", fontsize=12)
    plt.ylabel("Penalty Score", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig("fitness_vs_iterations.png")

    # --- PLOT 2: Final Fitness Distribution ---
    plt.figure(figsize=(6, 5))
    plt.boxplot([penalties], labels=["Circle Packing"])
    plt.title("Final Penalty Distribution", fontsize=14)
    plt.ylabel("Final Penalty Score", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("final_fitness_distribution.png")

    # --- PLOT 3: Convergence Speed Distribution ---
    plt.figure(figsize=(6, 5))
    plt.boxplot([iterations], labels=["Circle Packing"])
    plt.title("Convergence Speed Distribution", fontsize=14)
    plt.ylabel("Iterations to Convergence", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("convergence_speed_distribution.png")

    print("\nVisualizations saved successfully!")

if __name__ == "__main__":
    # Feel free to reduce these parameters if execution is taking too long
    run_visualized_benchmark(num_circles=30, radius=0.1, pop_size=60000, max_iter=2000, runs=100)

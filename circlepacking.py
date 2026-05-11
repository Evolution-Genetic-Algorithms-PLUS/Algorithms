import numpy as np
import matplotlib.pyplot as plt

class CirclePackingCGA:
    def __init__(self, num_circles: int, radius: float, bits_per_val: int = 10, population_size: int = 100):
        self.num_circles = num_circles
        self.radius = radius
        self.bits_per_val = bits_per_val
        self.vec_length = num_circles * 2 * bits_per_val  # x and y for each circle
        self.n = population_size
        self.p = np.ones(self.vec_length, dtype=np.float64) * 0.5

    def decode(self, bitstring: np.ndarray) -> np.ndarray:
        """Converts bitstring to [N, 2] array of coordinates."""
        coords = []
        for i in range(0, len(bitstring), self.bits_per_val):
            # Convert bits to integer, then normalize to [0, 1]
            bits = bitstring[i : i + self.bits_per_val]
            val = np.dot(bits, 2 ** np.arange(len(bits))[::-1])
            coords.append(val / (2**self.bits_per_val - 1))
        return np.array(coords).reshape(self.num_circles, 2)

    def calculate_fitness(self, bitstring: np.ndarray) -> float:
        """Lower score is better (fewer overlaps/violations)."""
        centers = self.decode(bitstring)
        penalty = 0.0
        r = self.radius

        for i, (x, y) in enumerate(centers):
            # 1. Boundary violations
            if x - r < 0: penalty += abs(x - r)
            if x + r > 1: penalty += abs(x + r - 1)
            if y - r < 0: penalty += abs(y - r)
            if y + r > 1: penalty += abs(y + r - 1)

            # 2. Overlaps with other circles
            for j in range(i + 1, self.num_circles):
                dist = np.linalg.norm(centers[i] - centers[j])
                if dist < 2 * r:
                    penalty += (2 * r - dist) ** 2  # Squared to prioritize large overlaps
        
        return penalty

    def generate(self) -> np.ndarray:
        return (np.random.random(self.vec_length) < self.p).astype(int)

    def compete(self, a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        score_a = self.calculate_fitness(a)
        score_b = self.calculate_fitness(b)
        return (a, b) if score_a <= score_b else (b, a)

    def update(self, winner: np.ndarray, loser: np.ndarray) -> None:
        for i in range(self.vec_length):
            if winner[i] != loser[i]:
                if winner[i] == 1:
                    self.p[i] += 1 / self.n
                else:
                    self.p[i] -= 1 / self.n
                self.p[i] = np.clip(self.p[i], 0, 1)

    def execute(self, max_iter=2000):
        for i in range(max_iter):
            a, b = self.generate(), self.generate()
            winner, loser = self.compete(a, b)
            self.update(winner, loser)
            
            # Check for convergence
            if np.all((self.p == 0) | (self.p == 1)):
                print(f"Converged at iteration {i}")
                break

    def visualize(self):
        # Best guess is the current probability vector rounded
        best_bits = (self.p > 0.5).astype(int)
        centers = self.decode(best_bits)
        
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.set_title(f"CGA Circle Packing (N={self.num_circles}, R={self.radius})")

        for x, y in centers:
            circle = plt.Circle((x, y), self.radius, color='teal', alpha=0.6, ec='black')
            ax.add_patch(circle)
        
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.show()

# Execution
if __name__ == "__main__":
    # N=10 circles with radius 0.1 in a 1.0x1.0 square
    cga = CirclePackingCGA(num_circles=30, radius=0.1, population_size=200)
    cga.execute(max_iter=60000)
    cga.visualize()

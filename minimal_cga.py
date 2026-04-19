import numpy as np

class CGA:
    def __init__(self, vec_length: int, population_size: int):
        self.p = np.ones(vec_length, dtype=np.float64) * 0.5  # 1), float64 necessary to reduce float point errors
        self.vec_length = vec_length
        self.n = population_size

    def execute(self, keep_history=False) -> list:
        i: int = 0
        history: list[tuple] = []

        while not self.has_converged():
            a: np.ndarray = self.generate()
            b: np.ndarray = self.generate()

            winner, looser = self.compete(a, b)

            self.update(winner, looser)

            if keep_history:
                history.append((i, self.p.copy(), a, b))

        return history

    def generate(self) -> np.ndarray:
        # 2) for the easy problem generate a binary vector
        return np.array([np.random.choice([0, 1], p=[1-p, p]) for p in self.p])

    def compete(self, a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # 3) binary vector to integer
        score_a = np.dot(a, 2 ** np.arange(a.size)[::-1])
        score_b = np.dot(b, 2 ** np.arange(b.size)[::-1])
        return (a, b) if score_a >= score_b else (b, a)

    def update(self, winner: np.ndarray, loser: np.ndarray) -> None:
        # 4)
        for i in range(self.vec_length):
            if winner[i] != loser[i]:
                if winner[i] == 1:
                    self.p[i] += 1 / self.n
                else:
                    self.p[i] -= 1 / self.n
                self.p[i] = max(0, min(1, self.p[i]))

    def has_converged(self) -> bool:
        # 5)
        return all([p == 0 or p == 1 for p in self.p])


if __name__ == "__main__":
    cga = CGA(vec_length=8, population_size=16)

    history = cga.execute(keep_history=True)
    for elem in history:
        print(*elem, sep=", ")

    print("Final Probability Vector: ", cga.p)
# Algorithms

This folder contains a minimal Compact Genetic Algorithm (cGA) implementation for binary optimization.

## File Overview

- `minimal_cga.py`: Minimal cGA class with sampling, competition, probability-vector update, and convergence check.

## Implemented cGA Logic

The implementation maintains a probability vector `p` of length `vec_length`.

1. Initialize all probabilities with `0.5`.
2. In each iteration, sample two binary candidate solutions from `p`.
3. Evaluate both candidates and select winner/loser.
4. Update each differing probability by `±1/n` (`n` = virtual population size).
5. Clamp probabilities to `[0, 1]`.
6. Stop when all entries are exactly `0` or `1`.

## Current Fitness in `minimal_cga.py`

The current `compete` method interprets each bit vector as a binary number and maximizes that integer value.

- This is a simple baseline behavior.
- For OneMax or other benchmark tasks, replace `compete` scoring with the task-specific fitness function.

## Usage

Run the script directly:

```bash
python minimal_cga.py
```

The current `__main__` block uses:

- `vec_length = 8`
- `population_size = 16`
- `keep_history = True`

and prints iteration history plus the final probability vector.

## Notes

- Probability updates use `float64` to reduce floating-point drift.
- Convergence currently checks exact equality to `0` or `1`.
- If needed, this can be changed to tolerance-based convergence for larger experiments.
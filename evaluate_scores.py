import json
import os
import numpy as np
import matplotlib.pyplot as plt
from onemax import BenchmarkableOneMaxCGA
from deceptivetrapcga import BenchmarkableDeceptiveTrapCGA
from benchmarkablecga import BenchmarkableCGA


def compute_scores(final_vals, histories, iterations, problem_type, params, max_iter):
    # final_vals: array of final fitness/penalty per run
    # histories: list/array of per-run arrays (recorded fitness history)
    # iterations: array of iterations-to-convergence per run
    # problem_type: 'max' or 'min'
    # params: dict with keys for global/local/worst definitions where applicable

    runs = len(final_vals)

    # Determine global and worst values
    V_global = params.get('V_global')
    V_worst = params.get('V_worst')
    V_local = params.get('V_local')
    V_worst_local = params.get('V_worst_local')

    # If worst values are not provided, use observed extremes
    if V_worst is None:
        V_worst = np.max(final_vals) if problem_type == 'min' else np.min(final_vals)
    if V_worst_local is None and V_local is not None:
        # use extreme from histories
        all_hist = np.concatenate(histories) if len(histories) > 0 else final_vals
        V_worst_local = np.max(all_hist) if problem_type == 'min' else np.min(all_hist)

    # S1 per-run
    S1 = []
    for v in final_vals:
        denom = abs(V_worst - V_global) if V_worst is not None and V_global is not None else 1.0
        s = 1 - abs(v - V_global) / denom if denom != 0 else 1.0
        S1.append(s)
    S1 = np.clip(S1, 0.0, 1.0)

    # S2: local score using recorded history values (if local defined)
    if V_local is None:
        # fallback: set local score equal to global score mean
        S2 = np.array(S1)
    else:
        ratios = []
        for hist in histories:
            denom = abs(V_worst_local - V_local) if V_worst_local is not None else 1.0
            if denom == 0:
                ratios.append(0.0)
            else:
                ratios.append(np.mean(np.abs(hist - V_local) / denom))
        S2 = 1 - np.array(ratios)
        S2 = np.clip(S2, 0.0, 1.0)

    # S3: convergence score per run
    S3 = 1 - (np.array(iterations) / float(max_iter))
    S3 = np.clip(S3, 0.0, 1.0)

    return {
        'S1_per_run': np.array(S1),
        'S2_per_run': np.array(S2),
        'S3_per_run': np.array(S3),
        'S1_mean': float(np.mean(S1)),
        'S2_mean': float(np.mean(S2)),
        'S3_mean': float(np.mean(S3)),
    }


def summarize_and_save(name: str, scores: dict, max_iter: int, out_dir: str = '.') -> None:
    """Pretty-print a human-readable summary and save JSON results to out_dir."""
    os.makedirs(out_dir, exist_ok=True)

    s1 = np.array(scores['S1_per_run'])
    s2 = np.array(scores['S2_per_run'])
    s3 = np.array(scores['S3_per_run'])

    s1_mean = float(scores['S1_mean'])
    s2_mean = float(scores['S2_mean'])
    s3_mean = float(scores['S3_mean'])

    s1_std = float(np.std(s1))
    s2_std = float(np.std(s2))
    s3_std = float(np.std(s3))

    # Convergence: count runs with S3 > 0 (i.e., converged before max_iter)
    converged = int(np.sum(s3 > 0))
    runs = len(s1)

    # Human-readable per-run lists (rounded)
    s1_list = [round(float(x), 3) for x in s1]
    s2_list = [round(float(x), 3) for x in s2]
    s3_list = [round(float(x), 3) for x in s3]

    print('\n' + '=' * 60)
    print(f'Benchmark: {name}')
    print('-' * 60)
    print(f'Runs: {runs} | Converged runs: {converged}/{runs} | max_iter: {max_iter}')

    print('\nS1 (Global)')
    print(f'  per-run : {s1_list}')
    print(f'  mean ± std : {s1_mean:.4f} ± {s1_std:.4f}')

    print('\nS2 (Local)')
    print(f'  per-run : {s2_list}')
    print(f'  mean ± std : {s2_mean:.4f} ± {s2_std:.4f}')

    print('\nS3 (Convergence)')
    print(f'  per-run : {s3_list}')
    print(f'  mean ± std : {s3_mean:.4f} ± {s3_std:.4f}')

    # Save JSON summary
    out = {
        'name': name,
        'runs': runs,
        'converged_runs': converged,
        'max_iter': max_iter,
        'S1_per_run': s1_list,
        'S1_mean': s1_mean,
        'S1_std': s1_std,
        'S2_per_run': s2_list,
        'S2_mean': s2_mean,
        'S2_std': s2_std,
        'S3_per_run': s3_list,
        'S3_mean': s3_mean,
        'S3_std': s3_std,
    }

    out_path = os.path.join(out_dir, f'results_{name.replace(" ","_")}.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)

    print(f'JSON results saved to: {out_path}')


def load_result_files(results_dir: str = 'results') -> list[dict]:
    """Load all benchmark result JSON files from results_dir."""
    files = [
        os.path.join(results_dir, filename)
        for filename in os.listdir(results_dir)
        if filename.startswith('results_') and filename.endswith('.json')
    ]
    loaded = []
    for path in sorted(files):
        with open(path, 'r') as handle:
            loaded.append(json.load(handle))
    return loaded


def save_raw_benchmark_data(raw_data: dict, out_dir: str = 'results') -> str:
    """Save raw benchmark histories, final values, and iterations for plotting."""
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'benchmark_raw_data.json')
    with open(out_path, 'w') as handle:
        json.dump(raw_data, handle, indent=2)
    return out_path


def load_raw_benchmark_data(raw_path: str = 'results/benchmark_raw_data.json') -> dict:
    with open(raw_path, 'r') as handle:
        return json.load(handle)


def collect_benchmark_runs(runner, runs: int, max_iter: int, seed_offset: int = 0) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
    finals = []
    histories = []
    iters = []
    for seed in range(runs):
        np.random.seed(seed + seed_offset)
        metrics = runner(max_iter=max_iter)
        finals.append(metrics['final_fitness'] if 'final_fitness' in metrics else metrics['final_penalty'])
        histories.append(np.array(metrics['fitness_history'], dtype=float))
        iters.append(metrics['iterations'])
    return np.array(finals, dtype=float), histories, np.array(iters, dtype=int)


def plot_grouped_benchmark_plots(raw_data: dict, out_dir: str = 'results') -> None:
    """Create the three benchmark-style plots for all benchmarks using raw histories."""
    os.makedirs(out_dir, exist_ok=True)

    benchmark_order = ['OneMax', 'Deceptive_Trap', 'Circle_Packing']
    display_names = {
        'OneMax': 'OneMax',
        'Deceptive_Trap': 'Deceptive Trap',
        'Circle_Packing': 'Circle Packing',
    }
    color_map = {
        'OneMax': '#2ca02c',
        'Deceptive_Trap': '#d62728',
        'Circle_Packing': '#1f77b4',
    }

    # Fitness vs iteration: one subplot per benchmark.
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)
    for axis, name in zip(axes, benchmark_order):
        item = raw_data[name]
        histories = np.array(item['histories'], dtype=float)
        mean_history = np.mean(histories, axis=0)
        std_history = np.std(histories, axis=0)
        x_axis = np.arange(len(mean_history)) * item['record_stride']

        axis.plot(x_axis, mean_history, color=color_map[name], label='Mean')
        axis.fill_between(x_axis, mean_history - std_history, mean_history + std_history, alpha=0.25, color=color_map[name])
        axis.set_title(display_names[name])
        axis.set_xlabel('Iterations')
        axis.grid(True, linestyle='--', alpha=0.4)
        axis.legend(loc='best')

        y_label = 'Penalty Score' if name == 'Circle_Packing' else 'Fitness Score'
        axis.set_ylabel(y_label)

    fig.suptitle('Fitness vs Iterations Across Benchmarks', y=1.02)
    fig.tight_layout()
    fitness_path = os.path.join(out_dir, 'benchmark_fitness_vs_iterations.png')
    fig.savefig(fitness_path, dpi=200, bbox_inches='tight')
    plt.close(fig)

    # Final fitness distribution boxplot: grouped by benchmark.
    plt.figure(figsize=(9, 5))
    final_data = [raw_data[name]['finals'] for name in benchmark_order]
    plt.boxplot(final_data, tick_labels=[display_names[name] for name in benchmark_order], showmeans=True)
    plt.ylabel('Final Score / Penalty')
    plt.title('Final Fitness Distribution by Benchmark')
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    final_path = os.path.join(out_dir, 'benchmark_final_fitness_distribution.png')
    plt.savefig(final_path, dpi=200)
    plt.close()

    # Convergence speed boxplot: grouped by benchmark.
    plt.figure(figsize=(9, 5))
    iter_data = [raw_data[name]['iterations'] for name in benchmark_order]
    plt.boxplot(iter_data, tick_labels=[display_names[name] for name in benchmark_order], showmeans=True)
    plt.ylabel('Iterations to Convergence')
    plt.title('Convergence Speed by Benchmark')
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    iter_path = os.path.join(out_dir, 'benchmark_convergence_speed_distribution.png')
    plt.savefig(iter_path, dpi=200)
    plt.close()

    print(f'Visualization saved to: {fitness_path}')
    print(f'Visualization saved to: {final_path}')
    print(f'Visualization saved to: {iter_path}')


def plot_results(results_dir: str = 'results', out_dir: str = 'results') -> None:
    """Create presentation-friendly visualizations from saved benchmark JSON files."""
    os.makedirs(out_dir, exist_ok=True)
    results = load_result_files(results_dir)
    if not results:
        print(f'No result JSON files found in {results_dir}.')
        return

    # Sort benchmarks into a stable, presentation-friendly order.
    order = {'OneMax': 0, 'Deceptive_Trap': 1, 'Circle_Packing': 2}
    results.sort(key=lambda item: order.get(item['name'], 99))

    names = [item['name'].replace('_', ' ') for item in results]
    s1_means = [item['S1_mean'] for item in results]
    s2_means = [item['S2_mean'] for item in results]
    s3_means = [item['S3_mean'] for item in results]

    x = np.arange(len(names))
    width = 0.24

    plt.figure(figsize=(10, 6))
    plt.bar(x - width, s1_means, width=width, label='S1 Global', color='#1f77b4')
    plt.bar(x, s2_means, width=width, label='S2 Local', color='#ff7f0e')
    plt.bar(x + width, s3_means, width=width, label='S3 Convergence', color='#2ca02c')
    plt.xticks(x, names)
    plt.ylim(0, 1.05)
    plt.ylabel('Mean Score')
    plt.title('Benchmark Score Summary')
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.legend()
    plt.tight_layout()
    summary_path = os.path.join(out_dir, 'results_summary_bars.png')
    plt.savefig(summary_path, dpi=200)
    plt.close()

    # Boxplots grouped by score so each metric is compared across benchmarks.
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)
    score_keys = [
        ('S1_per_run', 'S1 Global'),
        ('S2_per_run', 'S2 Local'),
        ('S3_per_run', 'S3 Convergence'),
    ]

    for axis, (score_key, title) in zip(axes, score_keys):
        box_data = [item[score_key] for item in results]
        axis.boxplot(box_data, tick_labels=names, showmeans=True)
        axis.set_title(title)
        axis.set_ylim(0, 1.05)
        axis.grid(axis='y', linestyle='--', alpha=0.4)
        axis.tick_params(axis='x', rotation=20)
        if axis is axes[0]:
            axis.set_ylabel('Score')

    fig.suptitle('Score Comparison Across Benchmarks', y=1.02)
    fig.tight_layout()
    boxplot_path = os.path.join(out_dir, 'results_score_boxplots_by_metric.png')
    fig.savefig(boxplot_path, dpi=200, bbox_inches='tight')
    plt.close(fig)

    print(f'Visualization saved to: {summary_path}')
    print(f'Visualization saved to: {boxplot_path}')


def eval_onemax(runs=30, length=100, pop_size=100, max_iter=5000):
    finals = []
    histories = []
    iters = []
    for seed in range(runs):
        np.random.seed(seed)
        cga = BenchmarkableOneMaxCGA(length=length, population_size=pop_size)
        metrics = cga.execute_with_metrics(max_iter=max_iter)
        finals.append(metrics['final_fitness'])
        histories.append(metrics['fitness_history'])
        iters.append(metrics['iterations'])

    finals = np.array(finals)
    iters = np.array(iters)

    params = {
        'V_global': length,
        'V_worst': 0,
        'V_local': None,
        'V_worst_local': None
    }

    return compute_scores(finals, histories, iters, problem_type='max', params=params, max_iter=max_iter), finals, histories, iters


def eval_deceptive_trap(runs=30, length=100, k=5, pop_size=1000, max_iter=20000):
    finals = []
    histories = []
    iters = []
    m = length // k
    for seed in range(runs):
        np.random.seed(seed)
        cga = BenchmarkableDeceptiveTrapCGA(length=length, k=k, population_size=pop_size)
        metrics = cga.execute_with_metrics(max_iter=max_iter)
        finals.append(metrics['final_fitness'])
        histories.append(metrics['fitness_history'])
        iters.append(metrics['iterations'])

    finals = np.array(finals)
    iters = np.array(iters)

    params = {
        'V_global': length,                # all ones
        'V_worst': 0,                      # theoretical worst
        'V_local': m * (k - 1),            # deceptive local optimum (all zeros)
        'V_worst_local': 0
    }

    return compute_scores(finals, histories, iters, problem_type='max', params=params, max_iter=max_iter), finals, histories, iters


def eval_circle_packing(runs=30, num_circles=30, radius=0.1, pop_size=200, max_iter=2000):
    finals = []
    histories = []
    iters = []
    for seed in range(runs):
        np.random.seed(seed)
        cga = BenchmarkableCGA(num_circles=num_circles, radius=radius, population_size=pop_size)
        metrics = cga.execute_with_metrics(max_iter=max_iter)
        finals.append(metrics['final_penalty'])
        histories.append(metrics['fitness_history'])
        iters.append(metrics['iterations'])

    finals = np.array(finals)
    iters = np.array(iters)

    # For circle packing lower is better; global optimum = 0 penalty.
    # Use observed worst as normalization denominator if not known.
    params = {
        'V_global': 0.0,
        'V_worst': float(np.max(finals)),
        'V_local': None,
        'V_worst_local': None
    }

    return compute_scores(finals, histories, iters, problem_type='min', params=params, max_iter=max_iter), finals, histories, iters


if __name__ == '__main__':
    raw_path = os.path.join('results', 'benchmark_raw_data.json')

    if os.path.exists(raw_path):
        raw_data = load_raw_benchmark_data(raw_path)
    else:
        # Canonical evaluation per Problem-Descriptions/README.md (runs=30, seeds 0-29)
        print('Evaluating OneMax (L=100, runs=30, max_iter=5000)...')
        onemax_scores, onemax_finals, onemax_histories, onemax_iters = eval_onemax(runs=30, length=100, pop_size=100, max_iter=5000)
        summarize_and_save('OneMax', onemax_scores, max_iter=5000, out_dir='results')

        print('\nEvaluating Deceptive Trap (L=100, k=5, runs=30, max_iter=20000)...')
        trap_scores, trap_finals, trap_histories, trap_iters = eval_deceptive_trap(runs=30, length=100, k=5, pop_size=1000, max_iter=20000)
        summarize_and_save('Deceptive_Trap', trap_scores, max_iter=20000, out_dir='results')

        print('\nEvaluating Circle Packing (N=30, r=0.1, runs=30, max_iter=50000)...')
        cp_scores, cp_finals, cp_histories, cp_iters = eval_circle_packing(runs=30, num_circles=30, radius=0.1, pop_size=200, max_iter=50000)
        summarize_and_save('Circle_Packing', cp_scores, max_iter=50000, out_dir='results')

        raw_data = {
            'OneMax': {
                'finals': onemax_finals.tolist(),
                'histories': [hist.tolist() for hist in onemax_histories],
                'iterations': onemax_iters.tolist(),
                'record_stride': 10,
            },
            'Deceptive_Trap': {
                'finals': trap_finals.tolist(),
                'histories': [hist.tolist() for hist in trap_histories],
                'iterations': trap_iters.tolist(),
                'record_stride': 50,
            },
            'Circle_Packing': {
                'finals': cp_finals.tolist(),
                'histories': [hist.tolist() for hist in cp_histories],
                'iterations': cp_iters.tolist(),
                'record_stride': 10,
            },
        }
        save_raw_benchmark_data(raw_data, out_dir='results')

    plot_results(results_dir='results', out_dir='results')
    plot_grouped_benchmark_plots(raw_data, out_dir='results')

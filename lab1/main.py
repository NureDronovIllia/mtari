"""ЛР №1: Multiple Knapsack Problem із Max-Min Ant System (MMAS).

Запуск: python main.py --mode all
Залежності: matplotlib, pillow (див. requirements.txt).
"""

import argparse
import csv
import random
import statistics
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


@dataclass(frozen=True)
class Item:
    name: str
    weight: int
    value: int


@dataclass(frozen=True)
class Problem:
    name: str
    capacities: tuple[int, ...]
    items: tuple[Item, ...]

    def __post_init__(self) -> None:
        if not self.capacities or any(c < 0 for c in self.capacities):
            raise ValueError("Потрібні невід'ємні місткості рюкзаків")
        if any(i.weight <= 0 or i.value <= 0 for i in self.items):
            raise ValueError("Вага й цінність предметів мають бути додатними")
        if len({i.name for i in self.items}) != len(self.items):
            raise ValueError("Назви предметів мають бути унікальними")


@dataclass(frozen=True)
class Config:
    ants: int = 16
    iterations: int = 40
    alpha: float = 1.0
    beta: float = 2.0
    rho: float = 0.2
    tau_min: float = 0.1
    tau_max: float = 5.0
    fill_penalty: float = 0.5

    def __post_init__(self) -> None:
        if self.ants < 1 or self.iterations < 1:
            raise ValueError("ants та iterations повинні бути додатними")
        if self.alpha < 0 or self.beta < 0 or self.fill_penalty < 0:
            raise ValueError("alpha, beta, fill_penalty повинні бути невід'ємними")
        if not 0 < self.rho < 1 or not 0 < self.tau_min <= self.tau_max:
            raise ValueError("Потрібно 0 < rho < 1 та 0 < tau_min <= tau_max")


@dataclass(frozen=True)
class Solution:
    # assignment[i] = індекс рюкзака (від 0), або -1, якщо предмет не вибрано
    assignment: tuple[int, ...]
    value: int


@dataclass
class RunResult:
    solution: Solution
    history: list[int]
    snapshots: list[tuple[int, list[list[float]]]]
    pheromone: list[list[float]]


def examples() -> dict[str, Problem]:
    def p(name: str, caps: tuple[int, ...], pairs: list[tuple[int, int]]) -> Problem:
        return Problem(
            name, caps, tuple(Item(f"I{i + 1}", w, v) for i, (w, v) in enumerate(pairs))
        )

    return {
        "example": Problem(
            "example", (5, 6), (Item("a", 4, 7), Item("b", 3, 5), Item("c", 2, 3))
        ),
        "balanced": p(
            "balanced",
            (12, 12),
            [
                (7, 13),
                (6, 12),
                (5, 10),
                (4, 7),
                (4, 8),
                (3, 5),
                (3, 7),
                (2, 4),
                (2, 3),
                (6, 11),
                (8, 16),
            ],
        ),
        "uneven": p(
            "uneven",
            (7, 11, 14),
            [
                (8, 17),
                (7, 13),
                (6, 12),
                (5, 10),
                (4, 8),
                (3, 7),
                (3, 5),
                (2, 4),
                (2, 3),
                (7, 12),
                (9, 18),
                (5, 9),
                (4, 7),
            ],
        ),
        "tight": p(
            "tight",
            (10, 10, 10),
            [
                (6, 11),
                (6, 12),
                (6, 13),
                (5, 10),
                (5, 9),
                (5, 11),
                (4, 9),
                (4, 8),
                (3, 6),
                (3, 7),
                (2, 5),
                (2, 4),
                (7, 14),
                (8, 17),
            ],
        ),
        "unfit": p("unfit", (3, 4), [(5, 10), (6, 12), (7, 14)]),
    }


def assess(problem: Problem, assignment: tuple[int, ...]) -> int:
    """Перевіряє допустимість і повертає сумарну цінність."""
    if len(assignment) != len(problem.items):
        raise ValueError("Невідповідна кількість призначень")
    used = [0] * len(problem.capacities)
    total = 0
    for item, k in zip(problem.items, assignment):
        if k == -1:
            continue
        if k < 0 or k >= len(used):
            raise ValueError("Некоректний рюкзак")
        used[k] += item.weight
        total += item.value
    if any(w > c for w, c in zip(used, problem.capacities)):
        raise ValueError("Перевищення місткості")
    return total


def exact_optimum(problem: Problem) -> Solution:
    """Точний розв'язок DP із мемоізацією: лише для невеликих тестів."""
    n = len(problem.items)

    @lru_cache(maxsize=None)
    def dp(i: int, remaining: tuple[int, ...]) -> tuple[int, tuple[int, ...]]:
        if i == n:
            return 0, ()
        item = problem.items[i]
        best_val, tail = dp(i + 1, remaining)
        best_assign = (-1,) + tail
        for k, free in enumerate(remaining):
            if item.weight <= free:
                updated = list(remaining)
                updated[k] -= item.weight
                other_val, other_assign = dp(i + 1, tuple(updated))
                candidate = item.value + other_val
                if candidate > best_val:
                    best_val = candidate
                    best_assign = (k,) + other_assign
        return best_val, best_assign

    value, assignment = dp(0, problem.capacities)
    assert assess(problem, assignment) == value
    return Solution(assignment, value)


def construct_ant(
    problem: Problem, pheromone: list[list[float]], cfg: Config, rng: random.Random
) -> Solution:
    """Кожен мураха додає лише допустимі пари (предмет, рюкзак)."""
    n, m = len(problem.items), len(problem.capacities)
    left = list(problem.capacities)
    assignment = [-1] * n
    available = set(range(n))

    while available:
        moves: list[tuple[int, int]] = []
        weights: list[float] = []
        for i in sorted(available):
            item = problem.items[i]
            for k in range(m):
                if item.weight > left[k] or problem.capacities[k] == 0:
                    continue
                eta = (item.value / item.weight) * (
                    left[k] / problem.capacities[k]
                ) ** cfg.fill_penalty
                attractiveness = pheromone[i][k] ** cfg.alpha * eta**cfg.beta
                moves.append((i, k))
                weights.append(attractiveness)
        if not moves:
            break
        i, k = rng.choices(moves, weights=weights, k=1)[0]
        assignment[i] = k
        left[k] -= problem.items[i].weight
        available.remove(i)

    result = tuple(assignment)
    return Solution(result, assess(problem, result))


def solve_aco(
    problem: Problem, cfg: Config, seed: int = 42, record_snapshots: bool = False
) -> RunResult:
    """MMAS: одна найкраща мураха за ітерацію, випаровування та межі τ."""
    rng = random.Random(seed)
    n, m = len(problem.items), len(problem.capacities)
    tau = [[cfg.tau_max for _ in range(m)] for _ in range(n)]
    best = Solution(tuple([-1] * n), 0)
    history: list[int] = []
    snapshots: list[tuple[int, list[list[float]]]] = []
    selected = {1, 2, 5, 10, 20, cfg.iterations}
    if record_snapshots:
        snapshots.append((0, [row[:] for row in tau]))
    value_bound = max(1, sum(item.value for item in problem.items))

    for iteration in range(1, cfg.iterations + 1):
        colony = [construct_ant(problem, tau, cfg, rng) for _ in range(cfg.ants)]
        iteration_best = max(colony, key=lambda s: s.value)
        if iteration_best.value > best.value:
            best = iteration_best
        # MMAS для максимізації: підсилюється лише найкраща мураха ітерації.
        deposit = cfg.rho * cfg.tau_max * (iteration_best.value / value_bound)
        for i in range(n):
            for k in range(m):
                updated = (1.0 - cfg.rho) * tau[i][k]
                if iteration_best.assignment[i] == k:
                    updated += deposit
                tau[i][k] = max(cfg.tau_min, min(cfg.tau_max, updated))
        history.append(best.value)
        if record_snapshots and iteration in selected:
            snapshots.append((iteration, [row[:] for row in tau]))

    assert assess(problem, best.assignment) == best.value
    return RunResult(best, history, snapshots, tau)


def run_tests() -> None:
    problems = examples()
    cfg = Config()
    for name, problem in problems.items():
        exact = exact_optimum(problem)
        result = solve_aco(problem, cfg, seed=42)
        assert result.solution.value <= exact.value
        assert len(result.history) == cfg.iterations
        assert result.history == sorted(result.history)
        assert all(
            cfg.tau_min <= x <= cfg.tau_max for row in result.pheromone for x in row
        )
        print(
            f"ТЕСТ {name:8}: ACO={result.solution.value:3}, OPT={exact.value:3}, "
            f"valid=так"
        )
    assert exact_optimum(problems["example"]).value == 15
    assert exact_optimum(problems["unfit"]).value == 0
    print("Усі автоматичні перевірки пройдено.")


def plot_demo(
    problem: Problem, result: RunResult, optimum: Solution, cfg: Config, output: Path
) -> None:

    matplotlib.use("Agg")

    output.mkdir(parents=True, exist_ok=True)
    # Графік найкращого значення за ітераціями.
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(
        range(1, len(result.history) + 1),
        result.history,
        marker="o",
        markersize=2,
        label="MMAS (найкраще знайдене)",
    )
    ax.axhline(optimum.value, linestyle="--", label=f"Точний оптимум = {optimum.value}")
    ax.set(
        xlabel="Ітерація", ylabel="Сумарна цінність", title=f"Збіжність: {problem.name}"
    )
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "01_convergence.png", dpi=150)
    plt.close(fig)

    # Матриці феромонів на старті та наприкінці.
    fig, axes = plt.subplots(
        1, 2, figsize=(10, max(4, len(problem.items) * 0.38)), constrained_layout=True
    )
    matrices = [result.snapshots[0][1], result.pheromone]
    for ax, mat, name in zip(axes, matrices, ["Початок", "Завершення"]):
        im = ax.imshow(
            mat, vmin=cfg.tau_min, vmax=cfg.tau_max, cmap="viridis", aspect="auto"
        )
        ax.set(title=name, xlabel="Рюкзак", ylabel="Предмет")
        ax.set_xticks(
            range(len(problem.capacities)),
            [f"K{k + 1}" for k in range(len(problem.capacities))],
        )
        ax.set_yticks(range(len(problem.items)), [i.name for i in problem.items])
    fig.colorbar(im, ax=axes, shrink=0.85, pad=0.03, label="Рівень феромону")
    fig.savefig(output / "02_pheromone.png", dpi=150)
    plt.close(fig)

    # Схема розміщення: кольорові ділянки позначають вагу предметів.
    fig, ax = plt.subplots(figsize=(9, max(3, len(problem.capacities) * 0.9)))
    for k, capacity in enumerate(problem.capacities):
        start = 0
        for i, bag in enumerate(result.solution.assignment):
            if bag != k:
                continue
            item = problem.items[i]
            ax.barh(k, item.weight, left=start, height=0.6, edgecolor="white")
            ax.text(
                start + item.weight / 2,
                k,
                item.name,
                ha="center",
                va="center",
                fontsize=9,
            )
            start += item.weight
        ax.text(capacity + 0.35, k, f"{start}/{capacity}", va="center")
    ax.set_yticks(
        range(len(problem.capacities)),
        [f"K{k + 1}" for k in range(len(problem.capacities))],
    )
    ax.set(
        xlabel="Зайнята вага",
        title=f"Розподіл предметів (цінність = {result.solution.value})",
    )
    ax.set_xlim(0, max(problem.capacities) * 1.3 + 1)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(output / "03_solution.png", dpi=150)
    plt.close(fig)

    # Анімована зміна феромонів і значення розв'язку під час ітерацій.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
    img = ax1.imshow(
        result.snapshots[0][1],
        vmin=cfg.tau_min,
        vmax=cfg.tau_max,
        cmap="viridis",
        aspect="auto",
    )
    ax1.set_xticks(
        range(len(problem.capacities)),
        [f"K{k + 1}" for k in range(len(problem.capacities))],
    )
    ax1.set_yticks(range(len(problem.items)), [item.name for item in problem.items])
    ax1.set(xlabel="Рюкзак", ylabel="Предмет")
    fig.colorbar(img, ax=ax1, shrink=0.75)
    (line,) = ax2.plot([], [], linewidth=2)
    ax2.axhline(optimum.value, linestyle="--", label="Оптимум")
    ax2.set(
        xlim=(0, cfg.iterations),
        ylim=(0, max(1, optimum.value) * 1.1),
        xlabel="Ітерація",
        ylabel="Найкраща цінність",
    )
    ax2.legend()
    ax2.grid(alpha=0.2)

    def animate(frame: int):
        step, mat = result.snapshots[frame]
        img.set_data(mat)
        line.set_data(range(1, step + 1), result.history[:step])
        ax1.set_title(f"Феромони, ітерація {step}")
        ax2.set_title(f"Поточний рекорд: {result.history[step - 1] if step else 0}")
        return img, line

    animation = FuncAnimation(
        fig,
        animate,
        frames=len(result.snapshots),
        interval=700,
        blit=False,
        repeat=True,
    )
    animation.save(output / "05_evolution.gif", writer=PillowWriter(fps=1))
    plt.close(fig)


def experiment(output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    problems = examples()
    settings = {
        "base": Config(),
        "alpha_0.5": replace(Config(), alpha=0.5),
        "alpha_3": replace(Config(), alpha=3.0),
        "beta_1": replace(Config(), beta=1.0),
        "beta_5": replace(Config(), beta=5.0),
        "rho_0.05": replace(Config(), rho=0.05),
        "rho_0.5": replace(Config(), rho=0.5),
    }
    seeds = range(8)
    cases = ["balanced", "uneven", "tight"]
    rows: list[dict] = []
    summary: list[dict] = []
    for case in cases:
        problem = problems[case]
        opt = exact_optimum(problem).value
        for setting, cfg in settings.items():
            scores = []
            histories = []
            for seed in seeds:
                result = solve_aco(problem, cfg, seed)
                val = result.solution.value
                scores.append(val)
                histories.append(result.history)
                rows.append(
                    {
                        "case": case,
                        "setting": setting,
                        "seed": seed,
                        "optimum": opt,
                        "aco_value": val,
                        "gap_pct": round(100 * (opt - val) / opt, 2) if opt else 0,
                    }
                )
            summary.append(
                {
                    "case": case,
                    "setting": setting,
                    "alpha": cfg.alpha,
                    "beta": cfg.beta,
                    "rho": cfg.rho,
                    "mean": round(statistics.mean(scores), 3),
                    "stdev": round(statistics.stdev(scores), 3),
                    "best": max(scores),
                    "worst": min(scores),
                    "hits": sum(score == opt for score in scores),
                    "runs": len(scores),
                    "mean_gap_pct": round(
                        100 * (opt - statistics.mean(scores)) / opt, 2
                    ),
                    "optimum": opt,
                }
            )
            if case == "tight":
                mean_line = [
                    statistics.mean(h[j] for h in histories)
                    for j in range(cfg.iterations)
                ]
                plt.plot(range(1, cfg.iterations + 1), mean_line, label=setting)
        print(f"Експеримент {case}: оптимум = {opt}")
    plt.axhline(
        exact_optimum(problems["tight"]).value,
        linestyle="--",
        color="black",
        linewidth=1,
        label="Оптимум",
    )
    plt.title("Порівняння параметрів MMAS: tight (середнє за 8 запусками)")
    plt.xlabel("Ітерація")
    plt.ylabel("Середня найкраща цінність")
    plt.legend(fontsize=8, ncol=2)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output / "04_parameter_comparison.png", dpi=150)
    plt.close()

    for filename, data in [
        ("06_experimental_results.csv", rows),
        ("07_parameter_summary.csv", summary),
    ]:
        with (output / filename).open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(data[0].keys()))
            writer.writeheader()
            writer.writerows(data)

    print("\nЗведені результати (8 запусків на комбінацію):")
    for row in summary:
        print(
            f"{row['case']:8} | {row['setting']:10} | "
            f"mean={row['mean']:6.2f} | hit={row['hits']}/8 | "
            f"gap={row['mean_gap_pct']:5.2f}%"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Multiple Knapsack + MMAS")
    parser.add_argument(
        "--mode", choices=["all", "demo", "experiment", "tests"], default="all"
    )
    parser.add_argument("--output", default="outputs", help="Каталог для результатів")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    output = Path(args.output)

    if args.mode in ("all", "tests"):
        run_tests()
    if args.mode in ("all", "demo"):
        problem = examples()["tight"]
        cfg = Config()
        optimum = exact_optimum(problem)
        result = solve_aco(problem, cfg, args.seed, record_snapshots=True)
        print(
            f"DEMO {problem.name}: MMAS={result.solution.value}, "
            f"OPT={optimum.value}, assignment={result.solution.assignment}"
        )
        plot_demo(problem, result, optimum, cfg, output)
    if args.mode in ("all", "experiment"):
        experiment(output)
    print(f"Готово. Режим: {args.mode}; файли: {output.resolve()}")


if __name__ == "__main__":
    main()

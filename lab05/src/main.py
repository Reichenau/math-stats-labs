from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from math import atan2, degrees
from matplotlib.patches import Ellipse


SAMPLE_SIZES = [20, 60, 100]
REPEATS = 1000
RANDOM_SEED = 42
RHO_VALUES = [0.0, 0.5, 0.9]
ELLIPSE_PROB = 0.95
MIXTURE_WEIGHTS = (0.9, 0.1)
MIXTURE_COMPONENT_1_COV = np.array([[1.0, 0.9], [0.9, 1.0]])
MIXTURE_COMPONENT_2_COV = np.array([[10.0, -9.0], [-9.0, 10.0]])


def generate_bivariate_normal(n, rho, rng):
    cov = np.array([[1.0, rho], [rho, 1.0]])
    return rng.multivariate_normal(mean=[0.0, 0.0], cov=cov, size=n)


def generate_mixture(n, rng):
    # Линейная комбинация двух независимых выборок для численной корреляции.
    s1 = rng.multivariate_normal([0, 0], MIXTURE_COMPONENT_1_COV, size=n)
    s2 = rng.multivariate_normal([0, 0], MIXTURE_COMPONENT_2_COV, size=n)
    return MIXTURE_WEIGHTS[0] * s1 + MIXTURE_WEIGHTS[1] * s2


def mixture_linear_combination_correlation():
    w1, w2 = MIXTURE_WEIGHTS
    cov1 = MIXTURE_COMPONENT_1_COV
    cov2 = MIXTURE_COMPONENT_2_COV

    var_x = w1**2 * cov1[0, 0] + w2**2 * cov2[0, 0]
    var_y = w1**2 * cov1[1, 1] + w2**2 * cov2[1, 1]
    cov_xy = w1**2 * cov1[0, 1] + w2**2 * cov2[0, 1]
    corr = cov_xy / np.sqrt(var_x * var_y)

    return float(var_x), float(var_y), float(cov_xy), float(corr)


def pearson_corr(x, y):
    return float(np.corrcoef(x, y)[0, 1])


def spearman_corr(x, y):
    return float(stats.spearmanr(x, y).correlation)


def quadrant_corr(x, y):
    mx = np.median(x)
    my = np.median(y)
    n11 = np.sum((x > mx) & (y > my))
    n22 = np.sum((x < mx) & (y < my))
    n12 = np.sum((x > mx) & (y < my))
    n21 = np.sum((x < mx) & (y > my))
    n = x.size
    return float((n11 + n22 - n12 - n21) / n)


def ellipse_parameters(x, y, prob=ELLIPSE_PROB):
    # Возвращает полуоси (a, b) и угол (в градусах) для эллипса равной вероятности
    cov = np.cov(np.vstack([x, y]))
    eigvals, eigvecs = np.linalg.eigh(cov)
    # сортируем по убыванию
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    # масштаб по уровню вероятности (двумерное хи-квадрат)
    k = np.sqrt(stats.chi2.ppf(prob, df=2))
    a = np.sqrt(max(eigvals[0], 0.0)) * k
    b = np.sqrt(max(eigvals[1], 0.0)) * k
    # Аналитический угол главной оси: tan(2a) = 2 cov_xy / (cov_xx - cov_yy)
    angle = 0.5 * degrees(atan2(2.0 * cov[0, 1], cov[0, 0] - cov[1, 1]))
    return float(a), float(b), float(angle)


def save_csv_stats(rows):
    output_dir = Path(__file__).resolve().parents[1] / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "estimates.csv"

    header = ["dist", "n", "stat", "E", "D"]
    lines = [",".join(header)]
    for row in rows:
        lines.append(
            ",".join(
                [
                    str(row["dist"]),
                    str(row["n"]),
                    str(row["stat"]),
                    f"{row['E']:.10f}",
                    f"{row['D']:.10f}",
                ]
            )
        )

    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Результаты сохранены: {output_path}")


def save_csv_ellipses(rows):
    output_dir = Path(__file__).resolve().parents[1] / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "ellipses.csv"

    header = ["dist", "n", "param", "E", "D"]
    lines = [",".join(header)]
    for row in rows:
        lines.append(
            ",".join(
                [
                    str(row["dist"]),
                    str(row["n"]),
                    str(row["param"]),
                    f"{row['E']:.10f}",
                    f"{row['D']:.10f}",
                ]
            )
        )

    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Параметры эллипсов сохранены: {output_path}")


def save_csv_mixture_theory(var_x, var_y, cov_xy, corr):
    output_dir = Path(__file__).resolve().parents[1] / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "mixture_theory.csv"

    header = ["var_x", "var_y", "cov_xy", "corr"]
    line = ",".join(
        [
            f"{var_x:.10f}",
            f"{var_y:.10f}",
            f"{cov_xy:.10f}",
            f"{corr:.10f}",
        ]
    )
    output_path.write_text(
        ",".join(header) + "\n" + line,
        encoding="utf-8",
    )
    print(f"Теоретическая корреляция смеси сохранена: {output_path}")


def add_probability_ellipses(ax, x, y):
    for prob, facecolor, alpha in [(0.5, "white", 0.0), (0.95, "gray", 0.15)]:
        a, b, angle = ellipse_parameters(x, y, prob=prob)
        ellipse = Ellipse(
            xy=(float(np.mean(x)), float(np.mean(y))),
            width=2 * a,
            height=2 * b,
            angle=angle,
            edgecolor="black",
            facecolor=facecolor,
            alpha=alpha,
            linewidth=1.2,
        )
        ax.add_patch(ellipse)


def plot_normals_grouped_by_sample(sample_sizes, rho_values, rng):
    figures_dir = Path(__file__).resolve().parents[1] / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    for n in sample_sizes:
        fig, axes = plt.subplots(1, len(rho_values), figsize=(22, 8))
        for ax, rho in zip(axes, rho_values):
            sample = generate_bivariate_normal(n, rho, rng)
            x = sample[:, 0]
            y = sample[:, 1]

            ax.scatter(x, y, s=12, color="black")
            ax.set_title(f"rho={rho}", fontsize=14)
            ax.grid(True, linestyle=":", alpha=0.7)
            ax.tick_params(axis="both", labelsize=12)
            add_probability_ellipses(ax, x, y)

        axes[0].set_ylabel("y", fontsize=12)
        for ax in axes:
            ax.set_xlabel("x", fontsize=12)

        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig(figures_dir / f"normal_n{n}_by_rho_scatter.png", dpi=300)
        plt.close(fig)


def plot_mixture_grouped_by_sample(sample_sizes, rng):
    figures_dir = Path(__file__).resolve().parents[1] / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, len(sample_sizes), figsize=(22, 8))
    for ax, n in zip(axes, sample_sizes):
        sample = generate_mixture(n, rng)
        x = sample[:, 0]
        y = sample[:, 1]

        ax.scatter(x, y, s=12, color="black")
        ax.set_title(f"n={n}", fontsize=14)
        ax.grid(True, linestyle=":", alpha=0.7)
        ax.tick_params(axis="both", labelsize=12)
        add_probability_ellipses(ax, x, y)

    axes[0].set_ylabel("y", fontsize=12)
    for ax in axes:
        ax.set_xlabel("x", fontsize=12)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(figures_dir / "mixture_by_n_scatter.png", dpi=300)
    plt.close(fig)


def run_experiments():
    rng = np.random.default_rng(RANDOM_SEED)
    rng_visual = np.random.default_rng(RANDOM_SEED + 1)
    plt.style.use("ggplot")

    # Сбор статистики по коэффициентам
    stats_rows = []
    ellipse_rows = []

    # Распределения: нормальные с разными rho и смесь
    distributions = []
    for rho in RHO_VALUES:
        distributions.append((f"Normal rho={rho}", f"rho={rho}", lambda n, r, rho=rho: generate_bivariate_normal(n, rho, r)))
    distributions.append(("Mixture", "mixture", lambda n, r: generate_mixture(n, r)))

    plot_normals_grouped_by_sample(SAMPLE_SIZES, RHO_VALUES, rng_visual)
    plot_mixture_grouped_by_sample(SAMPLE_SIZES, rng_visual)

    for dist_name, file_stub, generator in distributions:

        for n in SAMPLE_SIZES:
            pearsons = []
            spearmans = []
            quadrants = []
            a_list = []
            b_list = []
            ang_list = []
            center_x_list = []
            center_y_list = []

            for _ in range(REPEATS):
                sample = generator(n, rng)
                x = sample[:, 0]
                y = sample[:, 1]

                pearsons.append(pearson_corr(x, y))
                spearmans.append(spearman_corr(x, y))
                quadrants.append(quadrant_corr(x, y))

                a, b, ang = ellipse_parameters(x, y, prob=ELLIPSE_PROB)
                a_list.append(a)
                b_list.append(b)
                ang_list.append(ang)

                center_x_list.append(float(np.mean(x)))
                center_y_list.append(float(np.mean(y)))

            # агрегируем
            def mean_var(arr):
                arr = np.asarray(arr, dtype=float)
                return float(np.mean(arr)), float(np.var(arr, ddof=0))

            m_p, v_p = mean_var(pearsons)
            m_s, v_s = mean_var(spearmans)
            m_q, v_q = mean_var(quadrants)

            stats_rows.append({"dist": dist_name, "n": n, "stat": "pearson", "E": m_p, "D": v_p})
            stats_rows.append({"dist": dist_name, "n": n, "stat": "spearman", "E": m_s, "D": v_s})
            stats_rows.append({"dist": dist_name, "n": n, "stat": "quadrant", "E": m_q, "D": v_q})

            # эллипсы: агрегируем полуоси и углы
            m_a, v_a = mean_var(a_list)
            m_b, v_b = mean_var(b_list)
            m_ang, v_ang = mean_var(ang_list)
            m_cx, v_cx = mean_var(center_x_list)
            m_cy, v_cy = mean_var(center_y_list)

            ellipse_rows.append({"dist": dist_name, "n": n, "param": "a", "E": m_a, "D": v_a})
            ellipse_rows.append({"dist": dist_name, "n": n, "param": "b", "E": m_b, "D": v_b})
            ellipse_rows.append({"dist": dist_name, "n": n, "param": "angle", "E": m_ang, "D": v_ang})
            ellipse_rows.append({"dist": dist_name, "n": n, "param": "center_x", "E": m_cx, "D": v_cx})
            ellipse_rows.append({"dist": dist_name, "n": n, "param": "center_y", "E": m_cy, "D": v_cy})

    save_csv_stats(stats_rows)
    save_csv_ellipses(ellipse_rows)

    var_x, var_y, cov_xy, corr = mixture_linear_combination_correlation()
    save_csv_mixture_theory(var_x, var_y, cov_xy, corr)
    print(
        "Теоретическая корреляция для смеси как линейной комбинации: "
        f"D_x={var_x:.3f}, D_y={var_y:.3f}, cov(X,Y)={cov_xy:.3f}, r={corr:.3f}"
    )


def main():
    run_experiments()


if __name__ == "__main__":
    main()

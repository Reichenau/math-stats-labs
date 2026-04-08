import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from pathlib import Path

# Основная функция для рисования эмпирической гистограммы и теоретической кривой
def vizData(ax, data, x, y, n, bins='auto', is_discrete=False):
    ax.hist(data, bins=bins, density=True, color='lightgray', edgecolor='black', label='Гистограмма выборки')
    if is_discrete:
        ax.plot(x, y, marker='o', linestyle='none', color='black', label='Теоретические вероятности')
    else:
        ax.plot(x, y, linewidth=2, color='black', linestyle='--', label='Теоретическая плотность')
    ax.set_title(f'n={n}', fontsize=14)
    ax.grid(True, linestyle=':', alpha=0.7)
    ax.legend(loc='upper right', fontsize=12)
    ax.tick_params(axis='both', labelsize=14)


# Сохранение графика в PNG
def finalizeFigure(fig, dist_name):
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    figures_dir = Path(__file__).resolve().parents[1] / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)
    file_name = f'{dist_name}.png'
    fig.savefig(figures_dir / file_name, dpi=300)

# Нормальное распределение
def normDistribution(sizes):
    title = "Нормальное распределение $N(0,1)$"
    dist_name = 'norm'
    fig, axes = plt.subplots(1, len(sizes), figsize=(24, 8))
    for ax, n in zip(axes, sizes):
        data = stats.norm.rvs(size=n)
        x = np.linspace(-4, 4, 100)
        y = stats.norm.pdf(x)
        vizData(ax, data, x, y, n)
    finalizeFigure(fig, dist_name)

# Распределение Коши
def cauchyDistribution(sizes):
    title = r"Распределение Коши $C(0,1)$"
    dist_name = 'cauchy'
    fig, axes = plt.subplots(1, len(sizes), figsize=(24, 8))
    for ax, n in zip(axes, sizes):
        data = stats.cauchy.rvs(size=n)
        x = np.linspace(-10, 10, 100)
        y = stats.cauchy.pdf(x)
        data = data[(data >= -10) & (data <= 10)]
        vizData(ax, data, x, y, n)
    finalizeFigure(fig, dist_name)

# Распределение Лапласа
def laplaceDistribution(sizes):
    title = r"Распределение Лапласа $L(0,\frac{1}{\sqrt{2}})$"
    dist_name = 'laplace'
    fig, axes = plt.subplots(1, len(sizes), figsize=(24, 8))
    for ax, n in zip(axes, sizes):
        data = stats.laplace.rvs(loc=0, scale=1 / np.sqrt(2), size=n)
        x = np.linspace(-10, 10, 100)
        y = stats.laplace.pdf(x, loc=0, scale=1 / np.sqrt(2))
        vizData(ax, data, x, y, n)
    finalizeFigure(fig, dist_name)

# Распределение Пуассона (дискретное)
def poissonDistribution(sizes):
    title = r"Распределение Пуассона $P(k,5)$"
    dist_name = 'poisson'
    fig, axes = plt.subplots(1, len(sizes), figsize=(24, 8))
    for ax, n in zip(axes, sizes):
        data = stats.poisson.rvs(mu=5, size=n)
        x = np.arange(0, 25)
        y = stats.poisson.pmf(x, mu=5)
        bins = np.arange(-0.5, 25.5, 1)
        vizData(ax, data, x, y, n, bins=bins, is_discrete=True)
    finalizeFigure(fig, dist_name)

# Равномерное распределение
def uniformDistribution(sizes):
    title = r"Равномерное распределение $U(-\sqrt{3}, \sqrt{3})$"
    dist_name = 'uniform'
    fig, axes = plt.subplots(1, len(sizes), figsize=(24, 8))
    for ax, n in zip(axes, sizes):
        data = stats.uniform.rvs(loc=-np.sqrt(3), scale=2*np.sqrt(3), size=n)
        x = np.linspace(-4, 4, 100)
        y = stats.uniform.pdf(x, loc=-np.sqrt(3), scale=2*np.sqrt(3))
        vizData(ax, data, x, y, n)
    finalizeFigure(fig, dist_name)

# Последовательно запускаются все распределения
def main():
    sizes = [10, 100, 1000]

    normDistribution(sizes)
    cauchyDistribution(sizes)
    laplaceDistribution(sizes)
    poissonDistribution(sizes)
    uniformDistribution(sizes)


if __name__ == "__main__":
    main()

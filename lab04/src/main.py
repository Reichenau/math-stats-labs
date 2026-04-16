from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from scipy.stats import gaussian_kde


SAMPLE_SIZES = [20, 60, 100]
RANDOM_SEED = 42


def getDistributionConfigs():
	return [
		{
			"name": "Normal N(0,1)",
			"file_stub": "norm",
			"continuous": True,
			"x_range": (-4, 4),
			"generator": lambda size, rng: stats.norm.rvs(size=size, random_state=rng),
			"cdf": lambda x: stats.norm.cdf(x),
			"pdf": lambda x: stats.norm.pdf(x),
		},
		{
			"name": "Cauchy C(0,1)",
			"file_stub": "cauchy",
			"continuous": True,
			"x_range": (-4, 4),
			"generator": lambda size, rng: stats.cauchy.rvs(size=size, random_state=rng),
			"cdf": lambda x: stats.cauchy.cdf(x),
			"pdf": lambda x: stats.cauchy.pdf(x),
		},
		{
			"name": "Laplace L(0, 1/sqrt(2))",
			"file_stub": "laplace",
			"continuous": True,
			"x_range": (-4, 4),
			"generator": lambda size, rng: stats.laplace.rvs(
				loc=0,
				scale=1 / np.sqrt(2),
				size=size,
				random_state=rng,
			),
			"cdf": lambda x: stats.laplace.cdf(x, loc=0, scale=1 / np.sqrt(2)),
			"pdf": lambda x: stats.laplace.pdf(x, loc=0, scale=1 / np.sqrt(2)),
		},
		{
			"name": "Poisson P(5)",
			"file_stub": "poisson",
			"continuous": False,
			"x_range": (6, 14),
			"generator": lambda size, rng: stats.poisson.rvs(mu=5, size=size, random_state=rng),
			"cdf": lambda x: stats.poisson.cdf(x, mu=5),
			"pdf": lambda x: stats.poisson.pmf(x, mu=5),
		},
		{
			"name": "Uniform U(-sqrt(3), sqrt(3))",
			"file_stub": "uniform",
			"continuous": True,
			"x_range": (-4, 4),
			"generator": lambda size, rng: stats.uniform.rvs(
				loc=-np.sqrt(3),
				scale=2 * np.sqrt(3),
				size=size,
				random_state=rng,
			),
			"cdf": lambda x: stats.uniform.cdf(x, loc=-np.sqrt(3), scale=2 * np.sqrt(3)),
			"pdf": lambda x: stats.uniform.pdf(x, loc=-np.sqrt(3), scale=2 * np.sqrt(3)),
		},
	]


def makeEcdf(sample):
	values = np.sort(np.asarray(sample, dtype=float))
	y_values = np.arange(1, values.size + 1) / values.size
	return values, y_values


def kdeDensity(sample, x_values):
	try:
		kde = gaussian_kde(np.asarray(sample, dtype=float))
		return kde(x_values)
	except np.linalg.LinAlgError:
		jittered_sample = np.asarray(sample, dtype=float) + np.random.default_rng(RANDOM_SEED).normal(
			0,
			1e-6,
			size=len(sample),
		)
		kde = gaussian_kde(jittered_sample)
		return kde(x_values)


def computeErrorMetrics(sample, config):
	lower_bound, upper_bound = config["x_range"]
	if config["continuous"]:
		x_grid = np.linspace(lower_bound, upper_bound, 400)
	else:
		x_grid = np.arange(lower_bound, upper_bound + 1)

	sorted_sample, ecdf_values = makeEcdf(sample)
	ecdf_at_grid = np.searchsorted(sorted_sample, x_grid, side="right") / sorted_sample.size
	theoretical_cdf = config["cdf"](x_grid)
	ecdf_error = float(np.max(np.abs(ecdf_at_grid - theoretical_cdf)))

	if config["continuous"]:
		density_grid = np.linspace(lower_bound, upper_bound, 400)
	else:
		density_grid = np.arange(lower_bound, upper_bound + 1)

	if config["continuous"]:
		density_values = kdeDensity(sample, density_grid)
		theoretical_density = config["pdf"](density_grid)
		density_error = float(np.mean((density_values - theoretical_density) ** 2))
	else:
		counts = np.array([np.mean(sample == value) for value in density_grid], dtype=float)
		theoretical_density = config["pdf"](density_grid)
		density_error = float(np.mean((counts - theoretical_density) ** 2))

	return ecdf_error, density_error


def plotEcdf(ax, sample, config, sample_size):
	lower_bound, upper_bound = config["x_range"]
	x_min = lower_bound - 0.5 if not config["continuous"] else lower_bound
	x_max = upper_bound + 0.5 if not config["continuous"] else upper_bound
	x_sorted, y_ecdf = makeEcdf(sample)
	x_grid = np.linspace(x_min, x_max, 500)

	# Добавляем граничные точки, чтобы ЭФР явно доходила до 0 и 1 на концах оси.
	x_step = np.concatenate(([x_min], x_sorted, [x_max]))
	y_step = np.concatenate(([0.0], y_ecdf, [1.0]))

	ax.step(x_step, y_step, where="post", color="black", linewidth=1.8, label="Эмпирическая ФР")
	ax.plot(x_grid, config["cdf"](x_grid), color="black", linestyle="--", linewidth=1.8, label="Теоретическая ФР")
	ax.set_xlim(x_min, x_max)
	ax.set_ylim(-0.02, 1.02)
	ax.set_title(f"n={sample_size}", fontsize=14)
	ax.grid(True, linestyle=":", alpha=0.7)
	ax.legend(loc="lower right", fontsize=10)
	ax.tick_params(axis="both", labelsize=12)


def plotDensity(ax, sample, config, sample_size):
	lower_bound, upper_bound = config["x_range"]

	if config["continuous"]:
		x_grid = np.linspace(lower_bound, upper_bound, 400)
		bins = 20
		kde_values = kdeDensity(sample, x_grid)
		theoretical_values = config["pdf"](x_grid)
		ax.hist(sample, bins=bins, density=True, color="lightgray", edgecolor="black", label="Гистограмма выборки")
		ax.plot(x_grid, kde_values, color="black", linewidth=1.8, label="Ядерная оценка плотности")
		ax.plot(x_grid, theoretical_values, color="black", linestyle="--", linewidth=1.8, label="Теоретическая плотность")
		x_min = lower_bound
		x_max = upper_bound
	else:
		pmf_x = np.arange(lower_bound, upper_bound + 1)
		counts = np.array([np.mean(sample == value) for value in pmf_x], dtype=float)
		pmf_values = config["pdf"](pmf_x)
		ax.bar(pmf_x, counts, width=0.8, color="lightgray", edgecolor="black", label="Полигон относительных частот")
		ax.plot(pmf_x, pmf_values, color="black", marker="o", linestyle="none", label="Теоретическая вероятность")
		x_min = lower_bound - 0.5
		x_max = upper_bound + 0.5

	ax.set_xlim(x_min, x_max)
	ax.set_title(f"n={sample_size}", fontsize=14)
	ax.grid(True, linestyle=":", alpha=0.7)
	ax.legend(loc="upper right", fontsize=10)
	ax.tick_params(axis="both", labelsize=12)


def saveDistributionFigure(config, rng):
	samples = []
	result_rows = []

	for sample_size in SAMPLE_SIZES:
		sample = np.asarray(config["generator"](sample_size, rng), dtype=float)
		samples.append(sample)

		ecdf_error, density_error = computeErrorMetrics(sample, config)
		result_rows.append(
			{
				"distribution": config["name"],
				"n": sample_size,
				"ecdf_max_error": ecdf_error,
				"density_mse": density_error,
			}
		)

	figures_dir = Path(__file__).resolve().parents[1] / "figures"
	figures_dir.mkdir(parents=True, exist_ok=True)

	fig_ecdf, axes_ecdf = plt.subplots(1, len(SAMPLE_SIZES), figsize=(22, 8))
	for column_index, sample_size in enumerate(SAMPLE_SIZES):
		plotEcdf(axes_ecdf[column_index], samples[column_index], config, sample_size)
		axes_ecdf[column_index].set_xlabel("x", fontsize=12)
	axes_ecdf[0].set_ylabel("Э. ф. р.", fontsize=12)
	fig_ecdf.tight_layout()
	fig_ecdf.savefig(figures_dir / f"{config['file_stub']}_ecdf.png", dpi=300)
	plt.close(fig_ecdf)

	fig_density, axes_density = plt.subplots(1, len(SAMPLE_SIZES), figsize=(22, 8))
	for column_index, sample_size in enumerate(SAMPLE_SIZES):
		plotDensity(axes_density[column_index], samples[column_index], config, sample_size)
		axes_density[column_index].set_xlabel("x", fontsize=12)
	axes_density[0].set_ylabel("Плотность", fontsize=12)
	fig_density.tight_layout()
	fig_density.savefig(figures_dir / f"{config['file_stub']}_density.png", dpi=300)
	plt.close(fig_density)

	return result_rows


def saveCsv(rows):
	output_dir = Path(__file__).resolve().parents[1] / "data" / "processed"
	output_dir.mkdir(parents=True, exist_ok=True)
	output_path = output_dir / "ecdf_kde_metrics.csv"

	lines = ["distribution,n,ecdf_max_error,density_mse"]
	for row in rows:
		lines.append(
			f"{row['distribution']},{row['n']},{row['ecdf_max_error']:.10f},{row['density_mse']:.10f}"
		)

	output_path.write_text("\n".join(lines), encoding="utf-8")
	print(f"Результаты сохранены: {output_path}")


def printReport(rows):
	current_distribution = None
	for row in rows:
		if row["distribution"] != current_distribution:
			current_distribution = row["distribution"]
			print(f"\nРаспределение: {current_distribution}")
		print(
			f"  n={row['n']}: max|F_n-F| = {row['ecdf_max_error']:.6f}, "
			f"MSE(KDE, f) = {row['density_mse']:.6f}"
		)


def main():
	rng = np.random.default_rng(RANDOM_SEED)
	plt.style.use("default")

	distribution_configs = getDistributionConfigs()
	result_rows = []

	for config in distribution_configs:
		result_rows.extend(saveDistributionFigure(config, rng))

	printReport(result_rows)
	saveCsv(result_rows)


if __name__ == "__main__":
	main()

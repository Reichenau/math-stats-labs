from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats


SAMPLE_SIZES = [20, 100]
REPEATS = 1000
POISSON_LAMBDA = 5
RANDOM_SEED = 42


def getDistributionConfigs():
	return [
		{
			"name": "Normal N(0,1)",
			"file_stub": "norm",
			"generator": lambda size, rng: stats.norm.rvs(size=size, random_state=rng),
		},
		{
			"name": "Cauchy C(0,1)",
			"file_stub": "cauchy",
			"generator": lambda size, rng: stats.cauchy.rvs(size=size, random_state=rng),
		},
		{
			"name": "Laplace L(0, 1/sqrt(2))",
			"file_stub": "laplace",
			"generator": lambda size, rng: stats.laplace.rvs(
				loc=0,
				scale=1 / np.sqrt(2),
				size=size,
				random_state=rng,
			),
		},
		{
			"name": f"Poisson P({POISSON_LAMBDA})",
			"file_stub": "poisson",
			"generator": lambda size, rng: stats.poisson.rvs(
				mu=POISSON_LAMBDA,
				size=size,
				random_state=rng,
			),
		},
		{
			"name": "Uniform U(-sqrt(3), sqrt(3))",
			"file_stub": "uniform",
			"generator": lambda size, rng: stats.uniform.rvs(
				loc=-np.sqrt(3),
				scale=2 * np.sqrt(3),
				size=size,
				random_state=rng,
			),
		},
	]


def tukeyBounds(sample):
	q1, q3 = np.quantile(sample, [0.25, 0.75], method="midpoint")
	iqr = q3 - q1
	lower_bound = q1 - 1.5 * iqr
	upper_bound = q3 + 1.5 * iqr
	return lower_bound, upper_bound


def outlierShare(sample):
	lower_bound, upper_bound = tukeyBounds(sample)
	outlier_count = np.sum((sample < lower_bound) | (sample > upper_bound))
	return float(outlier_count / sample.size)


def meanOutlierShare(generator, sample_size, repeats, rng):
	shares = []
	for _ in range(repeats):
		sample = np.asarray(generator(sample_size, rng), dtype=float)
		shares.append(outlierShare(sample))
	return float(np.mean(shares))


def saveOutlierCsv(rows):
	output_dir = Path(__file__).resolve().parents[1] / "data" / "processed"
	output_dir.mkdir(parents=True, exist_ok=True)
	output_path = output_dir / "outlier_share.csv"

	lines = ["distribution,n,mean_outlier_share"]
	for row in rows:
		lines.append(
			f"{row['distribution']},{row['n']},{row['mean_outlier_share']:.10f}"
		)

	output_path.write_text("\n".join(lines), encoding="utf-8")
	print(f"Результаты сохранены: {output_path}")


def saveBoxplotFigure(dist_name, file_stub, generator, sample_sizes, rng):
	samples = [np.asarray(generator(size, rng), dtype=float) for size in sample_sizes]

	fig, ax = plt.subplots(figsize=(10, 6))
	boxplot = ax.boxplot(
		samples,
		patch_artist=True,
		labels=[f"n={size}" for size in sample_sizes],
		widths=0.55,
		showfliers=True,
	)

	colors = ["white", "gray"]
	for i, box in enumerate(boxplot["boxes"]):
		box.set(facecolor=colors[i % len(colors)], edgecolor="black")
	for median in boxplot["medians"]:
		median.set(color="black", linewidth=1.6)
	for whisker in boxplot["whiskers"]:
		whisker.set(color="black", linewidth=1.2)
	for cap in boxplot["caps"]:
		cap.set(color="black", linewidth=1.2)
	for flier in boxplot["fliers"]:
		flier.set(marker="o", markerfacecolor="black", markeredgecolor="black", markersize=4)

	ax.set_ylabel("Значения выборки", fontsize=12)
	ax.grid(True, linestyle=":", alpha=0.7)
	ax.tick_params(axis="both", labelsize=12)

	figures_dir = Path(__file__).resolve().parents[1] / "figures"
	figures_dir.mkdir(parents=True, exist_ok=True)
	output_path = figures_dir / f"{file_stub}_boxplot.png"
	fig.tight_layout()
	fig.savefig(output_path, dpi=300)
	plt.close(fig)


def saveOutlierShareFigure(rows):
	grouped = {}
	for row in rows:
		grouped.setdefault(row["distribution"], {})[row["n"]] = row["mean_outlier_share"]

	distributions = list(grouped.keys())
	x_positions = np.arange(len(distributions))
	bar_width = 0.35

	fig, ax = plt.subplots(figsize=(12, 6))
	colors = ["white", "gray"]
	for index, sample_size in enumerate(SAMPLE_SIZES):
		values = [grouped[dist][sample_size] for dist in distributions]
		shift = (index - 0.5) * bar_width
		ax.bar(
			x_positions + shift,
			values,
			width=bar_width,
			color=colors[index % len(colors)],
			edgecolor="black",
			label=f"n={sample_size}",
		)

	ax.set_ylabel("Доля выбросов", fontsize=12)
	ax.set_xticks(x_positions)
	ax.set_xticklabels(distributions, rotation=15, ha="right", fontsize=11)
	ax.grid(True, axis="y", linestyle=":", alpha=0.7)
	ax.legend(loc="upper right", fontsize=11)
	ax.tick_params(axis="y", labelsize=11)

	figures_dir = Path(__file__).resolve().parents[1] / "figures"
	figures_dir.mkdir(parents=True, exist_ok=True)
	output_path = figures_dir / "outlier_share.png"
	fig.tight_layout()
	fig.savefig(output_path, dpi=300)
	plt.close(fig)


def printReport(rows):
	current_distribution = None
	for row in rows:
		if row["distribution"] != current_distribution:
			current_distribution = row["distribution"]
			print(f"\nРаспределение: {current_distribution}")
		print(f"  n={row['n']}: средняя доля выбросов = {row['mean_outlier_share']:.6f}")


def main():
	rng = np.random.default_rng(RANDOM_SEED)
	plt.style.use("ggplot")

	distribution_configs = getDistributionConfigs()
	result_rows = []

	for config in distribution_configs:
		saveBoxplotFigure(
			dist_name=config["name"],
			file_stub=config["file_stub"],
			generator=config["generator"],
			sample_sizes=SAMPLE_SIZES,
			rng=rng,
		)

		for sample_size in SAMPLE_SIZES:
			mean_share = meanOutlierShare(
				generator=config["generator"],
				sample_size=sample_size,
				repeats=REPEATS,
				rng=rng,
			)
			result_rows.append(
				{
					"distribution": config["name"],
					"n": sample_size,
					"mean_outlier_share": mean_share,
				}
			)

	printReport(result_rows)
	saveOutlierCsv(result_rows)
	saveOutlierShareFigure(result_rows)


if __name__ == "__main__":
	main()

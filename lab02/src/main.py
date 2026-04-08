from pathlib import Path

import numpy as np
import scipy.stats as stats


SAMPLE_SIZES = [10, 100, 1000]
REPEATS = 1000
TRIM_RATIO = 0.1


def calculateCharacteristics(sample):
	"""Вычисляет выборочные характеристики положения для одной выборки."""
	sorted_sample = np.sort(sample)
	sample_size = sorted_sample.size

	sample_mean = float(np.mean(sorted_sample))
	sample_median = float(np.median(sorted_sample))
	sample_midrange = float((sorted_sample[0] + sorted_sample[-1]) / 2.0)

	q1, q3 = np.quantile(sorted_sample, [0.25, 0.75], method="midpoint")
	sample_midquartile = float((q1 + q3) / 2.0)

	trim_count = int(np.floor(sample_size * TRIM_RATIO))
	if trim_count > 0 and (2 * trim_count) < sample_size:
		trimmed_sample = sorted_sample[trim_count:-trim_count]
	else:
		trimmed_sample = sorted_sample
	sample_trimmed_mean = float(np.mean(trimmed_sample))

	return {
		"z_bar": sample_mean,
		"z_med": sample_median,
		"z_r": sample_midrange,
		"z_q": sample_midquartile,
		"z_tr": sample_trimmed_mean,
	}


def estimateMoments(values):
	"""Оценивает E(z) и D(z)=E(z^2)-E(z)^2 по серии реализаций."""
	values_array = np.asarray(values, dtype=float)
	mean_value = float(np.mean(values_array))
	second_moment = float(np.mean(values_array ** 2))
	variance_value = float(second_moment - mean_value ** 2)
	return mean_value, variance_value


def runExperiment(dist_name, generator, rng):
	"""Проводит серию экспериментов для одного распределения."""
	rows = []

	for sample_size in SAMPLE_SIZES:
		stats_storage = {
			"z_bar": [],
			"z_med": [],
			"z_r": [],
			"z_q": [],
			"z_tr": [],
		}

		for _ in range(REPEATS):
			sample = generator(sample_size, rng)
			characteristics = calculateCharacteristics(sample)
			for key, value in characteristics.items():
				stats_storage[key].append(value)

		for characteristic_name, characteristic_values in stats_storage.items():
			expected_value, variance_value = estimateMoments(characteristic_values)
			rows.append(
				{
					"distribution": dist_name,
					"n": sample_size,
					"characteristic": characteristic_name,
					"E(z)": expected_value,
					"D(z)": variance_value,
				}
			)

	return rows


def formatValue(value):
	"""Единый формат чисел для консольного отчета."""
	return f"{value: .6f}"


def printReport(rows):
	"""Печатает сводную таблицу оценок в консоль."""
	current_distribution = None
	current_size = None

	for row in rows:
		if row["distribution"] != current_distribution:
			current_distribution = row["distribution"]
			print(f"\nРаспределение: {current_distribution}")
		if row["n"] != current_size:
			current_size = row["n"]
			print(f"  n = {current_size}")
			print("    characteristic      E(z)          D(z)")

		characteristic_name = row["characteristic"]
		expected_value = formatValue(row["E(z)"])
		variance_value = formatValue(row["D(z)"])
		print(f"    {characteristic_name:<15} {expected_value:>12} {variance_value:>12}")


def saveCsv(rows):
	"""Сохраняет итоговые результаты в CSV-файл."""
	output_dir = Path(__file__).resolve().parents[1] / "data" / "processed"
	output_dir.mkdir(parents=True, exist_ok=True)
	output_path = output_dir / "estimates.csv"

	header = ["distribution", "n", "characteristic", "E(z)", "D(z)"]
	lines = [",".join(header)]
	for row in rows:
		line = ",".join(
			[
				str(row["distribution"]),
				str(row["n"]),
				str(row["characteristic"]),
				f"{row['E(z)']:.10f}",
				f"{row['D(z)']:.10f}",
			]
		)
		lines.append(line)

	output_path.write_text("\n".join(lines), encoding="utf-8")
	print(f"\nРезультаты сохранены: {output_path}")


def main():
	rng = np.random.default_rng(42)

	distributions = [
		("Normal N(0,1)", lambda n, local_rng: stats.norm.rvs(size=n, random_state=local_rng)),
		("Cauchy C(0,1)", lambda n, local_rng: stats.cauchy.rvs(size=n, random_state=local_rng)),
		(
			"Laplace L(0, 1/sqrt(2))",
			lambda n, local_rng: stats.laplace.rvs(
				loc=0,
				scale=1 / np.sqrt(2),
				size=n,
				random_state=local_rng,
			),
		),
		("Poisson P(10)", lambda n, local_rng: stats.poisson.rvs(mu=10, size=n, random_state=local_rng)),
		(
			"Uniform U(-sqrt(3), sqrt(3))",
			lambda n, local_rng: stats.uniform.rvs(
				loc=-np.sqrt(3),
				scale=2 * np.sqrt(3),
				size=n,
				random_state=local_rng,
			),
		),
	]

	all_rows = []
	for dist_name, generator in distributions:
		all_rows.extend(runExperiment(dist_name, generator, rng))

	printReport(all_rows)
	saveCsv(all_rows)


if __name__ == "__main__":
	main()

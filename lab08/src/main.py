from csv import writer
from pathlib import Path

import numpy as np
from scipy import stats


RANDOM_SEED = 42
ALPHA = 0.05
SAMPLE_SIZE_1 = 20
SAMPLE_SIZE_2 = 100


def generate_normal_sample(n, rng):
	return rng.normal(loc=0.0, scale=1.0, size=n)


def sample_mean(sample):
	return float(np.mean(sample))


def sample_variance_biased(sample):
	return float(np.var(sample, ddof=0))


def sample_variance_unbiased(sample):
	return float(np.var(sample, ddof=1))


def mean_confidence_interval(sample, alpha=ALPHA):
	n = sample.size
	mean_value = sample_mean(sample)
	variance_value = sample_variance_biased(sample)
	standard_error = float(np.sqrt(variance_value / (n - 1)))
	critical_value = float(stats.t.ppf(1.0 - alpha / 2.0, df=n - 1))
	margin = critical_value * standard_error

	return {
		"n": n,
		"mean": mean_value,
		"lower": float(mean_value - margin),
		"upper": float(mean_value + margin),
		"critical_value": critical_value,
	}


def variance_confidence_interval(sample, alpha=ALPHA):
	n = sample.size
	variance_value = sample_variance_biased(sample)
	lower_quantile = float(stats.chi2.ppf(1.0 - alpha / 2.0, df=n - 1))
	upper_quantile = float(stats.chi2.ppf(alpha / 2.0, df=n - 1))
	lower_bound = n * variance_value / lower_quantile
	upper_bound = n * variance_value / upper_quantile

	return {
		"n": n,
		"variance": variance_value,
		"lower": float(lower_bound),
		"upper": float(upper_bound),
		"lower_quantile": lower_quantile,
		"upper_quantile": upper_quantile,
	}


def fisher_variance_test(sample_1, sample_2, alpha=ALPHA):
	variance_1 = sample_variance_unbiased(sample_1)
	variance_2 = sample_variance_unbiased(sample_2)
	df_1 = sample_1.size - 1
	df_2 = sample_2.size - 1
	statistic = float(variance_1 / variance_2)
	lower_critical = float(stats.f.ppf(alpha / 2.0, df_1, df_2))
	upper_critical = float(stats.f.ppf(1.0 - alpha / 2.0, df_1, df_2))
	cdf_value = float(stats.f.cdf(statistic, df_1, df_2))
	p_value = float(2.0 * min(cdf_value, 1.0 - cdf_value))
	accepted = lower_critical <= statistic <= upper_critical

	return {
		"statistic": statistic,
		"df_1": df_1,
		"df_2": df_2,
		"lower_critical": lower_critical,
		"upper_critical": upper_critical,
		"p_value": p_value,
		"accepted": accepted,
		"variance_1": variance_1,
		"variance_2": variance_2,
	}


def save_rows_csv(file_path, header, rows):
	file_path.parent.mkdir(parents=True, exist_ok=True)
	with file_path.open("w", newline="", encoding="utf-8") as csv_file:
		csv_writer = writer(csv_file)
		csv_writer.writerow(header)
		for row in rows:
			csv_writer.writerow(row)


def save_samples(sample_1, sample_2):
	output_path = Path(__file__).resolve().parents[1] / "data" / "raw" / "samples.csv"
	rows = []
	for index in range(max(sample_1.size, sample_2.size)):
		first_value = f"{sample_1[index]:.10f}" if index < sample_1.size else ""
		second_value = f"{sample_2[index]:.10f}" if index < sample_2.size else ""
		rows.append([index + 1, first_value, second_value])
	save_rows_csv(output_path, ["index", "sample_1", "sample_2"], rows)
	print(f"Выборки сохранены: {output_path}")


def save_interval_summary(mean_interval_1, variance_interval_1, mean_interval_2, variance_interval_2):
	output_path = Path(__file__).resolve().parents[1] / "data" / "processed" / "confidence_intervals.csv"
	rows = [
		[
			"sample_1",
			mean_interval_1["n"],
			f"{mean_interval_1['mean']:.10f}",
			f"{mean_interval_1['lower']:.10f}",
			f"{mean_interval_1['upper']:.10f}",
			f"{variance_interval_1['variance']:.10f}",
			f"{variance_interval_1['lower']:.10f}",
			f"{variance_interval_1['upper']:.10f}",
		],
		[
			"sample_2",
			mean_interval_2["n"],
			f"{mean_interval_2['mean']:.10f}",
			f"{mean_interval_2['lower']:.10f}",
			f"{mean_interval_2['upper']:.10f}",
			f"{variance_interval_2['variance']:.10f}",
			f"{variance_interval_2['lower']:.10f}",
			f"{variance_interval_2['upper']:.10f}",
		],
	]
	save_rows_csv(
		output_path,
		["sample", "n", "mean", "mean_lower", "mean_upper", "variance", "variance_lower", "variance_upper"],
		rows,
	)
	print(f"Интервальные оценки сохранены: {output_path}")


def save_fisher_result(result):
	output_path = Path(__file__).resolve().parents[1] / "data" / "processed" / "fisher_test_result.csv"
	rows = [[
		f"{result['statistic']:.10f}",
		result["df_1"],
		result["df_2"],
		f"{result['lower_critical']:.10f}",
		f"{result['upper_critical']:.10f}",
		f"{result['p_value']:.10f}",
		"accept" if result["accepted"] else "reject",
	]]
	save_rows_csv(
		output_path,
		["f_statistic", "df_1", "df_2", "lower_critical", "upper_critical", "p_value", "decision"],
		rows,
	)
	print(f"Результат F-теста сохранен: {output_path}")


def print_summary(mean_interval_1, variance_interval_1, mean_interval_2, variance_interval_2, fisher_result):
	print("\nДоверительные интервалы")
	print(
		f"  Выборка 1 (n={mean_interval_1['n']}): "
		f"E[X] in [{mean_interval_1['lower']:.6f}; {mean_interval_1['upper']:.6f}], "
		f"D[X] in [{variance_interval_1['lower']:.6f}; {variance_interval_1['upper']:.6f}]"
	)
	print(
		f"  Выборка 2 (n={mean_interval_2['n']}): "
		f"E[X] in [{mean_interval_2['lower']:.6f}; {mean_interval_2['upper']:.6f}], "
		f"D[X] in [{variance_interval_2['lower']:.6f}; {variance_interval_2['upper']:.6f}]"
	)
	decision = "не отвергается" if fisher_result["accepted"] else "отвергается"
	print("\nF-критерий")
	print(f"  F = {fisher_result['statistic']:.6f}")
	print(f"  Критическая область: ({fisher_result['lower_critical']:.6f}; {fisher_result['upper_critical']:.6f})")
	print(f"  p-value = {fisher_result['p_value']:.6f}")
	print(f"  Решение: H0 {decision}")


def main():
	rng = np.random.default_rng(RANDOM_SEED)
	sample_1 = generate_normal_sample(SAMPLE_SIZE_1, rng)
	sample_2 = generate_normal_sample(SAMPLE_SIZE_2, rng)

	mean_interval_1 = mean_confidence_interval(sample_1, alpha=ALPHA)
	variance_interval_1 = variance_confidence_interval(sample_1, alpha=ALPHA)
	mean_interval_2 = mean_confidence_interval(sample_2, alpha=ALPHA)
	variance_interval_2 = variance_confidence_interval(sample_2, alpha=ALPHA)
	fisher_result = fisher_variance_test(sample_1, sample_2, alpha=ALPHA)

	save_samples(sample_1, sample_2)
	save_interval_summary(mean_interval_1, variance_interval_1, mean_interval_2, variance_interval_2)
	save_fisher_result(fisher_result)
	print_summary(mean_interval_1, variance_interval_1, mean_interval_2, variance_interval_2, fisher_result)


if __name__ == "__main__":
	main()

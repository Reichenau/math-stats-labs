from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


RANDOM_SEED = 42
ALPHA = 0.05
MAIN_SAMPLE_SIZE = 100
SENSITIVITY_SAMPLE_SIZE = 20
SENSITIVITY_REPEATS = 1000


def generate_normal_sample(n, rng):
	return rng.normal(loc=0.0, scale=1.0, size=n)


def generate_uniform_sample(n, rng):
	# Равномерное распределение с нулевым средним и единичной дисперсией
	bound = np.sqrt(3.0)
	return rng.uniform(low=-bound, high=bound, size=n)


def generate_laplace_sample(n, rng):
	# Распределение Лапласа с нулевым средним и единичной дисперсией
	scale = 1.0 / np.sqrt(2.0)
	return rng.laplace(loc=0.0, scale=scale, size=n)


def estimate_normal_parameters(sample):
	mu_hat = float(np.mean(sample))
	sigma_hat = float(np.sqrt(np.mean((sample - mu_hat) ** 2)))
	return mu_hat, sigma_hat


def choose_bin_count(n):
	# Базовый выбор по формуле Стерджеса: k ~= 1 + 3.3 * log10(n)
	return max(4, int(round(1.0 + 3.3 * np.log10(n))))


def build_interval_edges(sample, k):
	x_min = float(np.min(sample))
	x_max = float(np.max(sample))

	if np.isclose(x_min, x_max):
		eps = 1e-6
		x_min -= eps
		x_max += eps

	step = (x_max - x_min) / k
	internal_edges = x_min + step * np.arange(1, k)
	return np.concatenate(([-np.inf], internal_edges, [np.inf]))


def merge_small_frequency_bins(observed, expected, edges, min_freq=5):
	"""Объединяет интервалы с ожидаемыми частотами < min_freq."""
	k = len(observed)
	merged_obs = list(observed)
	merged_exp = list(expected)
	merged_edges = list(edges)
	
	# Сначала объединяем край слева
	while merged_exp[0] < min_freq and len(merged_obs) > 1:
		merged_obs[0] = merged_obs[0] + merged_obs[1]
		merged_exp[0] = merged_exp[0] + merged_exp[1]
		merged_edges.pop(1)
		merged_obs.pop(1)
		merged_exp.pop(1)
	
	# Затем объединяем край справа
	while merged_exp[-1] < min_freq and len(merged_obs) > 1:
		merged_obs[-2] = merged_obs[-2] + merged_obs[-1]
		merged_exp[-2] = merged_exp[-2] + merged_exp[-1]
		merged_edges.pop(-2)
		merged_obs.pop(-1)
		merged_exp.pop(-1)
	
	# Объединяем внутренние интервалы с малыми частотами
	i = 1
	while i < len(merged_obs) - 1:
		if merged_exp[i] < min_freq:
			# Объединяем с правым соседом
			merged_obs[i] = merged_obs[i] + merged_obs[i + 1]
			merged_exp[i] = merged_exp[i] + merged_exp[i + 1]
			merged_edges.pop(i + 1)
			merged_obs.pop(i + 1)
			merged_exp.pop(i + 1)
			# Не увеличиваем i, чтобы проверить новый объединённый интервал
		else:
			i += 1
	
	return np.array(merged_obs), np.array(merged_exp), np.array(merged_edges)


def chi_square_normal_test(sample, alpha=ALPHA):
	n = sample.size
	mu_hat, sigma_hat = estimate_normal_parameters(sample)
	k = choose_bin_count(n)
	edges = build_interval_edges(sample, k)
	probabilities = stats.norm.cdf(edges[1:], loc=mu_hat, scale=sigma_hat) - stats.norm.cdf(
		edges[:-1], loc=mu_hat, scale=sigma_hat
	)
	probabilities = np.clip(probabilities, 1e-12, None)
	probabilities = probabilities / np.sum(probabilities)
	expected = n * probabilities

	observed, _ = np.histogram(sample, bins=edges)
	
	# Объединяем интервалы с малыми ожидаемыми частотами
	observed, expected, edges = merge_small_frequency_bins(observed, expected, edges, min_freq=5)
	k = len(observed)
	
	# Число степеней свободы: k интервалов минус 1 минус 2 параметра (mu, sigma)
	degrees_of_freedom = max(1, k - 3)

	statistic = float(np.sum((observed - expected) ** 2 / expected))
	critical_value = float(stats.chi2.ppf(1.0 - alpha, df=degrees_of_freedom))
	p_value = float(1.0 - stats.chi2.cdf(statistic, df=degrees_of_freedom))
	accepted = statistic < critical_value

	return {
		"n": n,
		"mu_hat": mu_hat,
		"sigma_hat": sigma_hat,
		"k": k,
		"degrees_of_freedom": degrees_of_freedom,
		"statistic": statistic,
		"critical_value": critical_value,
		"p_value": p_value,
		"accepted": accepted,
		"observed": observed,
		"expected": expected,
		"edges": edges,
	}


def build_breakdown_rows(sample, scenario_name, alpha=ALPHA):
	result = chi_square_normal_test(sample, alpha=alpha)
	n = result["n"]
	observed = result["observed"]
	expected = result["expected"]
	edges = result["edges"]
	rows = []

	for idx in range(result["k"]):
		interval_left = float(edges[idx])
		interval_right = float(edges[idx + 1])
		observed_count = int(observed[idx])
		expected_count = float(expected[idx])
		probability = float(expected_count / n)
		diff = float(observed_count - expected_count)
		contribution = float(diff**2 / expected_count)

		rows.append(
			{
				"scenario": scenario_name,
				"n": n,
				"mu_hat": result["mu_hat"],
				"sigma_hat": result["sigma_hat"],
				"bin": idx + 1,
				"interval_left": interval_left,
				"interval_right": interval_right,
				"observed": observed_count,
				"probability": probability,
				"expected": expected_count,
				"difference": diff,
				"contribution": contribution,
				"chi2": result["statistic"],
				"critical": result["critical_value"],
				"p_value": result["p_value"],
				"decision": "accept" if result["accepted"] else "reject",
			}
		)

	rows.append(
		{
			"scenario": scenario_name,
			"n": n,
			"mu_hat": result["mu_hat"],
			"sigma_hat": result["sigma_hat"],
			"bin": "sum",
			"interval_left": "",
			"interval_right": "",
			"observed": int(np.sum(observed)),
			"probability": float(np.sum(expected) / n),
			"expected": float(np.sum(expected)),
			"difference": float(np.sum(observed) - np.sum(expected)),
			"contribution": float(np.sum((observed - expected) ** 2 / expected)),
			"chi2": result["statistic"],
			"critical": result["critical_value"],
			"p_value": result["p_value"],
			"decision": "accept" if result["accepted"] else "reject",
		}
	)

	return rows


def save_breakdown_csv(rows):
	output_dir = Path(__file__).resolve().parents[1] / "data" / "processed"
	output_dir.mkdir(parents=True, exist_ok=True)
	output_path = output_dir / "chi_square_breakdown.csv"

	header = [
		"scenario",
		"n",
		"mu_hat",
		"sigma_hat",
		"bin",
		"interval_left",
		"interval_right",
		"observed",
		"probability",
		"expected",
		"difference",
		"contribution",
		"chi2",
		"critical",
		"p_value",
		"decision",
	]
	lines = [",".join(header)]
	for row in rows:
		lines.append(
			",".join(
				[
					str(row["scenario"]),
					str(row["n"]),
					f"{row['mu_hat']:.10f}",
					f"{row['sigma_hat']:.10f}",
					str(row["bin"]),
					str(row["interval_left"]),
					str(row["interval_right"]),
					str(row["observed"]),
					f"{row['probability']:.10f}",
					f"{row['expected']:.10f}",
					f"{row['difference']:.10f}",
					f"{row['contribution']:.10f}",
					f"{row['chi2']:.10f}",
					f"{row['critical']:.10f}",
					f"{row['p_value']:.10f}",
					str(row['decision']),
				]
			)
		)

	output_path.write_text("\n".join(lines), encoding="utf-8")
	print(f"Данные разбиений сохранены: {output_path}")


def save_single_result_csv(result):
	output_dir = Path(__file__).resolve().parents[1] / "data" / "processed"
	output_dir.mkdir(parents=True, exist_ok=True)
	output_path = output_dir / "chi_square_result.csv"

	header = ["n", "mu_hat", "sigma_hat", "k", "df", "chi2", "critical", "p_value", "decision"]
	decision = "accept" if result["accepted"] else "reject"
	line = ",".join(
		[
			str(result["n"]),
			f"{result['mu_hat']:.10f}",
			f"{result['sigma_hat']:.10f}",
			str(result["k"]),
			str(result["degrees_of_freedom"]),
			f"{result['statistic']:.10f}",
			f"{result['critical_value']:.10f}",
			f"{result['p_value']:.10f}",
			decision,
		]
	)
	output_path.write_text(",".join(header) + "\n" + line, encoding="utf-8")
	print(f"Результаты сохранены: {output_path}")


def evaluate_sensitivity(distribution_name, generator, n, repeats, rng):
	rejections = 0
	statistics_values = []
	mu_values = []
	sigma_values = []

	for _ in range(repeats):
		sample = generator(n, rng)
		result = chi_square_normal_test(sample, alpha=ALPHA)
		rejections += int(not result["accepted"])
		statistics_values.append(result["statistic"])
		mu_values.append(result["mu_hat"])
		sigma_values.append(result["sigma_hat"])

	statistics_values = np.asarray(statistics_values, dtype=float)
	mu_values = np.asarray(mu_values, dtype=float)
	sigma_values = np.asarray(sigma_values, dtype=float)

	return {
		"distribution": distribution_name,
		"n": n,
		"repeats": repeats,
		"rejection_rate": float(rejections / repeats),
		"mean_statistic": float(np.mean(statistics_values)),
		"mean_mu_hat": float(np.mean(mu_values)),
		"mean_sigma_hat": float(np.mean(sigma_values)),
		"std_statistic": float(np.std(statistics_values, ddof=0)),
	}


def save_sensitivity_csv(rows):
	output_dir = Path(__file__).resolve().parents[1] / "data" / "processed"
	output_dir.mkdir(parents=True, exist_ok=True)
	output_path = output_dir / "chi_square_sensitivity.csv"

	header = ["distribution", "n", "repeats", "rejection_rate", "mean_chi2", "mean_mu_hat", "mean_sigma_hat", "std_chi2"]
	lines = [",".join(header)]
	for row in rows:
		lines.append(
			",".join(
				[
					str(row["distribution"]),
					str(row["n"]),
					str(row["repeats"]),
					f"{row['rejection_rate']:.10f}",
					f"{row['mean_statistic']:.10f}",
					f"{row['mean_mu_hat']:.10f}",
					f"{row['mean_sigma_hat']:.10f}",
					f"{row['std_statistic']:.10f}",
				]
			)
		)

	output_path.write_text("\n".join(lines), encoding="utf-8")
	print(f"Результаты чувствительности сохранены: {output_path}")


def save_sample_csv(sample_rows):
	output_dir = Path(__file__).resolve().parents[1] / "data" / "processed"
	output_dir.mkdir(parents=True, exist_ok=True)
	output_path = output_dir / "chi_square_samples.csv"

	header = ["index", "normal", "uniform", "laplace"]
	lines = [",".join(header)]
	for idx, row in enumerate(sample_rows, start=1):
		lines.append(
			",".join(
				[
					str(idx),
					f"{row['normal']:.10f}",
					f"{row['uniform']:.10f}",
					f"{row['laplace']:.10f}",
				]
			)
		)

	output_path.write_text("\n".join(lines), encoding="utf-8")
	print(f"Выборки сохранены: {output_path}")


def plot_histogram_with_test(ax, sample, title, alpha=ALPHA):
	result = chi_square_normal_test(sample, alpha=alpha)
	mu_hat = result["mu_hat"]
	sigma_hat = result["sigma_hat"]

	# Используем интервалы, по которым рассчитан критерий χ² (после объединения)
	edges = np.array(result.get("edges", []), dtype=float)
	# Заменим бесконечности на конечные границы для построения гистограммы
	if edges.size > 0:
		finite_edges = edges.copy()
		data_min, data_max = float(np.min(sample)), float(np.max(sample))
		range_span = max(1e-6, data_max - data_min)
		if np.isneginf(finite_edges[0]):
			finite_edges[0] = data_min - 0.1 * range_span
		if np.isposinf(finite_edges[-1]):
			finite_edges[-1] = data_max + 0.1 * range_span
		bins_to_use = finite_edges
	else:
		# На случай, если edges не доступны — откат к обычному выбору
		bins_to_use = choose_bin_count(sample.size)

	ax.hist(sample, bins=bins_to_use, density=True, color="#d9d9d9", edgecolor="black", alpha=0.9)
	x_grid = np.linspace(np.min(sample) - 0.5, np.max(sample) + 0.5, 400)
	ax.plot(x_grid, stats.norm.pdf(x_grid, loc=mu_hat, scale=sigma_hat), color="black", linewidth=1.8, label="N(μ̂, σ̂)")

	# Если использованы численные границы интервалов — добавить линии раздела бинов
	if edges.size > 0:
		for e in bins_to_use:
			ax.axvline(e, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
	ax.set_title(title, fontsize=13)
	ax.set_xlabel("x", fontsize=12)
	ax.set_ylabel("density", fontsize=12)
	ax.grid(True, linestyle=":", alpha=0.7)
	ax.tick_params(axis="both", labelsize=11)
	ax.legend(loc="upper right", fontsize=9)
	decision_text = "H0 accepted" if result["accepted"] else "H0 rejected"
	ax.text(
		0.03,
		0.97,
		f"μ̂={mu_hat:.3f}\nσ̂={sigma_hat:.3f}\nχ²={result['statistic']:.3f}\n{decision_text}",
		transform=ax.transAxes,
		va="top",
		ha="left",
		fontsize=10,
		bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor="gray"),
	)
	return result


def save_sensitivity_figure(rows):
	figures_dir = Path(__file__).resolve().parents[1] / "figures"
	figures_dir.mkdir(parents=True, exist_ok=True)

	labels_map = {
		"uniform": "Равномерное",
		"laplace": "Лапласа",
	}
	labels = [labels_map.get(row["distribution"], row["distribution"]) for row in rows]
	rates = [row["rejection_rate"] for row in rows]

	fig, ax = plt.subplots(figsize=(8, 6))
	bars = ax.bar(labels, rates, color=["#6b6b6b", "#b0b0b0"], edgecolor="black")
	ax.set_ylim(0.0, 1.0)
	ax.set_xlabel("Распределение", fontsize=12)
	ax.set_ylabel("Доля отвержений", fontsize=12)
	ax.grid(True, axis="y", linestyle=":", alpha=0.7)
	for bar, rate in zip(bars, rates):
		ax.text(bar.get_x() + bar.get_width() / 2, rate + 0.03, f"{rate:.3f}", ha="center", va="bottom", fontsize=10)
	fig.tight_layout()
	output_path = figures_dir / "chi_square_sensitivity.png"
	fig.savefig(output_path, dpi=300)
	plt.close(fig)
	print(f"График сохранен: {output_path}")


def save_figures(normal_sample, uniform_sample, laplace_sample):
	figures_dir = Path(__file__).resolve().parents[1] / "figures"
	figures_dir.mkdir(parents=True, exist_ok=True)

	fig, axes = plt.subplots(1, 3, figsize=(20, 6))
	plot_histogram_with_test(axes[0], normal_sample, "Нормальная выборка, n=100")
	plot_histogram_with_test(axes[1], uniform_sample, "Равномерная выборка, n=20")
	plot_histogram_with_test(axes[2], laplace_sample, "Лапласовская выборка, n=20")
	fig.tight_layout()
	fig.savefig(figures_dir / "chi_square_samples.png", dpi=300)
	plt.close(fig)
	print(f"График сохранен: {figures_dir / 'chi_square_samples.png'}")


def print_result(result, label):
	decision = "не отвергается" if result["accepted"] else "отвергается"
	print(f"\n{label}")
	print(f"  n = {result['n']}")
	print(f"  mu_hat = {result['mu_hat']:.6f}")
	print(f"  sigma_hat = {result['sigma_hat']:.6f}")
	print(f"  chi2 = {result['statistic']:.6f}")
	print(f"  critical = {result['critical_value']:.6f}")
	print(f"  p-value = {result['p_value']:.6f}")
	print(f"  Решение: H0 {decision}")


def print_sensitivity_summary(rows):
	print("\nЧувствительность критерия")
	for row in rows:
		print(
			f"  {row['distribution']:<8} n={row['n']:<2} "
			f"reject_rate={row['rejection_rate']:.3f} mean_chi2={row['mean_statistic']:.3f}"
		)


def main():
	rng = np.random.default_rng(RANDOM_SEED)
	plt.style.use("ggplot")

	normal_sample = generate_normal_sample(MAIN_SAMPLE_SIZE, rng)
	normal_result = chi_square_normal_test(normal_sample, alpha=ALPHA)
	print_result(normal_result, "Проверка основной выборки")
	save_single_result_csv(normal_result)
	breakdown_rows = []
	breakdown_rows.extend(build_breakdown_rows(normal_sample, "normal", alpha=ALPHA))

	uniform_sample = generate_uniform_sample(SENSITIVITY_SAMPLE_SIZE, rng)
	laplace_sample = generate_laplace_sample(SENSITIVITY_SAMPLE_SIZE, rng)
	breakdown_rows.extend(build_breakdown_rows(uniform_sample, "uniform", alpha=ALPHA))
	breakdown_rows.extend(build_breakdown_rows(laplace_sample, "laplace", alpha=ALPHA))
	sample_rows = [
		{
			"normal": float(vn),
			"uniform": float(vu),
			"laplace": float(vl),
		}
		for vn, vu, vl in zip(normal_sample[:SENSITIVITY_SAMPLE_SIZE], uniform_sample, laplace_sample)
	]
	save_breakdown_csv(breakdown_rows)
	save_sample_csv(sample_rows)
	save_figures(normal_sample, uniform_sample, laplace_sample)

	sensitivity_rows = []
	sensitivity_rows.append(
		evaluate_sensitivity("uniform", generate_uniform_sample, SENSITIVITY_SAMPLE_SIZE, SENSITIVITY_REPEATS, rng)
	)
	sensitivity_rows.append(
		evaluate_sensitivity("laplace", generate_laplace_sample, SENSITIVITY_SAMPLE_SIZE, SENSITIVITY_REPEATS, rng)
	)
	save_sensitivity_csv(sensitivity_rows)
	save_sensitivity_figure(sensitivity_rows)
	print_sensitivity_summary(sensitivity_rows)


if __name__ == "__main__":
	main()

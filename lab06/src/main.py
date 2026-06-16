from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize


RANDOM_SEED = 42
TRUE_A = 2.0
TRUE_B = 2.0


def generate_x_values():
	# Равномерная сетка [-1.8, 2.0] с шагом 0.2 (20 точек)
	return np.round(np.arange(-1.8, 2.0 + 1e-12, 0.2), 10)


def generate_response(x_values, rng):
	noise = rng.normal(loc=0.0, scale=1.0, size=x_values.size)
	return TRUE_A + TRUE_B * x_values + noise


def add_outliers(y_values):
	y_out = np.array(y_values, dtype=float, copy=True)
	y_out[0] += 10.0
	y_out[-1] -= 10.0
	return y_out


def fit_ols(x_values, y_values):
	x_mean = float(np.mean(x_values))
	y_mean = float(np.mean(y_values))
	s_xx = float(np.sum((x_values - x_mean) ** 2))
	s_xy = float(np.sum((x_values - x_mean) * (y_values - y_mean)))

	b_hat = s_xy / s_xx
	a_hat = y_mean - b_hat * x_mean
	return float(a_hat), float(b_hat)


def fit_lad(x_values, y_values):
	# Инициализируемся МНК и минимизируем сумму модулей отклонений
	a0, b0 = fit_ols(x_values, y_values)

	def objective(params):
		a_hat, b_hat = params
		residuals = y_values - (a_hat + b_hat * x_values)
		return float(np.sum(np.abs(residuals)))

	result = minimize(objective, x0=np.array([a0, b0], dtype=float), method="Nelder-Mead")
	if not result.success:
		raise RuntimeError(f"Не удалось найти решение МНМ: {result.message}")

	return float(result.x[0]), float(result.x[1])


def parameter_errors(estimate, true_value):
	abs_error = float(abs(estimate - true_value))
	rel_error_pct = float(abs_error / abs(true_value) * 100.0)
	return abs_error, rel_error_pct


def evaluate_methods(x_values, y_values, scenario_name):
	rows = []
	methods = [
		("МНК", fit_ols),
		("МНМ", fit_lad),
	]

	for method_name, estimator in methods:
		a_hat, b_hat = estimator(x_values, y_values)
		delta_a, delta_a_pct = parameter_errors(a_hat, TRUE_A)
		delta_b, delta_b_pct = parameter_errors(b_hat, TRUE_B)

		rows.append(
			{
				"scenario": scenario_name,
				"method": method_name,
				"a": a_hat,
				"delta_a": delta_a,
				"delta_a_pct": delta_a_pct,
				"b": b_hat,
				"delta_b": delta_b,
				"delta_b_pct": delta_b_pct,
			}
		)

	return rows


def save_estimates_csv(rows):
	output_dir = Path(__file__).resolve().parents[1] / "data" / "processed"
	output_dir.mkdir(parents=True, exist_ok=True)
	output_path = output_dir / "regression_estimates.csv"

	header = ["scenario", "method", "a", "delta_a", "delta_a_pct", "b", "delta_b", "delta_b_pct"]
	lines = [",".join(header)]

	for row in rows:
		lines.append(
			",".join(
				[
					str(row["scenario"]),
					str(row["method"]),
					f"{row['a']:.10f}",
					f"{row['delta_a']:.10f}",
					f"{row['delta_a_pct']:.10f}",
					f"{row['b']:.10f}",
					f"{row['delta_b']:.10f}",
					f"{row['delta_b_pct']:.10f}",
				]
			)
		)

	output_path.write_text("\n".join(lines), encoding="utf-8")
	print(f"Результаты сохранены: {output_path}")


def save_samples_csv(x_values, y_clean, y_outliers):
	output_dir = Path(__file__).resolve().parents[1] / "data" / "processed"
	output_dir.mkdir(parents=True, exist_ok=True)
	output_path = output_dir / "regression_samples.csv"

	header = ["index", "x", "y_clean", "y_outliers"]
	lines = [",".join(header)]

	for idx, (x_val, y_c, y_o) in enumerate(zip(x_values, y_clean, y_outliers), start=1):
		lines.append(f"{idx},{x_val:.10f},{y_c:.10f},{y_o:.10f}")

	output_path.write_text("\n".join(lines), encoding="utf-8")
	print(f"Выборки сохранены: {output_path}")


def draw_lines(ax, x_values, y_values):
	a_ols, b_ols = fit_ols(x_values, y_values)
	a_lad, b_lad = fit_lad(x_values, y_values)

	x_grid = np.linspace(np.min(x_values), np.max(x_values), 300)
	y_true = TRUE_A + TRUE_B * x_grid
	y_ols = a_ols + b_ols * x_grid
	y_lad = a_lad + b_lad * x_grid

	ax.scatter(x_values, y_values, color="black", s=28, label="Данные")
	ax.plot(x_grid, y_true, color="black", linestyle="--", linewidth=1.8, label="Истинная прямая")
	ax.plot(x_grid, y_ols, color="black", linewidth=1.8, label="МНК")
	ax.plot(x_grid, y_lad, color="gray", linewidth=1.8, label="МНМ")
	ax.set_xlabel("x", fontsize=12)
	ax.set_ylabel("y", fontsize=12)
	ax.grid(True, linestyle=":", alpha=0.7)
	ax.tick_params(axis="both", labelsize=12)
	ax.legend(loc="upper left", fontsize=10)


def save_regression_figures(x_values, y_clean, y_outliers):
	figures_dir = Path(__file__).resolve().parents[1] / "figures"
	figures_dir.mkdir(parents=True, exist_ok=True)

	fig_clean, ax_clean = plt.subplots(figsize=(9, 7))
	draw_lines(ax_clean, x_values, y_clean)
	fig_clean.tight_layout()
	output_clean = figures_dir / "regression_clean.png"
	fig_clean.savefig(output_clean, dpi=300)
	plt.close(fig_clean)

	fig_outliers, ax_outliers = plt.subplots(figsize=(9, 7))
	draw_lines(ax_outliers, x_values, y_outliers)
	fig_outliers.tight_layout()
	output_outliers = figures_dir / "regression_outliers.png"
	fig_outliers.savefig(output_outliers, dpi=300)
	plt.close(fig_outliers)

	print(f"График сохранен: {output_clean}")
	print(f"График сохранен: {output_outliers}")


def print_report(rows):
	current_scenario = None
	for row in rows:
		if row["scenario"] != current_scenario:
			current_scenario = row["scenario"]
			print(f"\nСценарий: {current_scenario}")
			print("  method      a         Δa       δa,%       b         Δb       δb,%")

		print(
			f"  {row['method']:<8} "
			f"{row['a']:>8.4f} "
			f"{row['delta_a']:>8.4f} "
			f"{row['delta_a_pct']:>9.4f} "
			f"{row['b']:>8.4f} "
			f"{row['delta_b']:>8.4f} "
			f"{row['delta_b_pct']:>9.4f}"
		)


def main():
	rng = np.random.default_rng(RANDOM_SEED)
	plt.style.use("ggplot")

	x_values = generate_x_values()
	y_clean = generate_response(x_values, rng)
	y_outliers = add_outliers(y_clean)

	all_rows = []
	all_rows.extend(evaluate_methods(x_values, y_clean, "clean"))
	all_rows.extend(evaluate_methods(x_values, y_outliers, "outliers"))

	print_report(all_rows)
	save_estimates_csv(all_rows)
	save_samples_csv(x_values, y_clean, y_outliers)
	save_regression_figures(x_values, y_clean, y_outliers)


if __name__ == "__main__":
	main()


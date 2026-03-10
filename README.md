# math-stats-labs
Лабораторные работы по курсу «Математическая статистика» СПбПУ (весна 2026 г.)

## Структура проекта

	math-stats-labs/
	|-- README.md
	|-- requirements.txt
	|-- lab01/
	|   |-- data/
	|   |   |-- raw/
	|   |   `-- processed/
	|   |-- src/
	|   |   `-- main.py
	|   |-- figures/
	|   `-- report/
	|       `-- report.tex
	|-- lab02/
	|   |-- data/
	|   |-- src/
	|   |-- figures/
	|   `-- report/
	|-- lab03/
	|-- lab04/
	|-- lab05/
	|-- lab06/
	|-- lab07/
	`-- lab08/

## Установка зависимостей

	python -m pip install -r requirements.txt

## Запуск кода лабораторной

Пример для первой лабораторной:

	python lab01/src/main.py

## Сборка отчета

Пример для первой лабораторной (из корня репозитория):

	pdflatex -output-directory=lab01/report lab01/report/report.tex

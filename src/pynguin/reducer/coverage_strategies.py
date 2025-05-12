from pathlib import Path
import coverage
import pytest

import io
import contextlib

import importlib
import sys
import shutil


class ResultCollector:
    def __init__(self):
        self.passed_tests = []
        self.failed_tests = []
        self.xfailed_tests = []

    def pytest_runtest_logreport(self, report):
        if report.when == "call":
            report_node_id = report.nodeid
            node_name = "test_case_" + report_node_id.split("test_case_")[1]
            if report.outcome == "passed":
                self.passed_tests.append(node_name)
            elif report.outcome == "failed":
                self.failed_tests.append(node_name)
            elif report.outcome == "skipped":
                self.xfailed_tests.append(node_name)


# Перезагрузка модуля для очистки кеша
def reload_module(module_name):
    if module_name in sys.modules:
        importlib.reload(sys.modules[module_name])


def find_project_root(start_dir: Path):
    current_dir = start_dir.resolve()

    while current_dir != current_dir.parent:
        potential_root = current_dir / '.pytest_cache'
        if potential_root.exists() and potential_root.is_dir():
            return current_dir
        current_dir = current_dir.parent

    return None


# Очистка кэша pytest, если он существует
def pytest_cache_cleanup(file_path):
    pytest_cache_dir = file_path.parent / ".pytest_cache"
    if pytest_cache_dir.exists():
        shutil.rmtree(pytest_cache_dir)

    project_root = find_project_root(file_path)

    if project_root:
        pytest_cache_dir = project_root / ".pytest_cache"
        if pytest_cache_dir.exists():
            shutil.rmtree(pytest_cache_dir)


# Очистка кешей и запуск тестов
def run_tests_with_cache_cleanup(path_file_with_tests, path_target_file):
    # Перезагрузка модуля с целью сброса состояния
    target_module = path_target_file.stem
    reload_module(target_module)

    # Убираем внутренний кэш pytest
    pytest_cache_cleanup(path_file_with_tests)
    return run_tests(path_file_with_tests)


def run_tests(file_with_tests_path):
    collector = ResultCollector()

    # Перенаправление вывода
    str_io = io.StringIO()
    with contextlib.redirect_stdout(str_io):
        # Запуск pytest с дополнительным хук-классом для сбора результатов
        pytest_args = [
            str(file_with_tests_path),
            "-q", "--tb=short", "-rP", "--disable-warnings"
        ]
        pytest.main(pytest_args, plugins=[collector])

    pytest_cache_cleanup(file_with_tests_path)
    return collector.passed_tests, collector.failed_tests, collector.xfailed_tests


def get_coverage(target_file: Path, file_with_tests: Path):
    cov = coverage.Coverage(branch=True, data_file=None)
    cov.erase()
    cov.start()

    try:
        passed_names, failed_names, xfailed_names = run_tests_with_cache_cleanup(file_with_tests, target_file)
    finally:
        cov.stop()

    # Получение данных покрытия
    cov_data = cov.get_data()

    # Удаляем объект для гарантии сброса
    del cov

    # Получаем все возможные ветви
    branches = cov_data.arcs(str(target_file)) or []

    # Получаем покрытые строки
    covered_lines = set(cov_data.lines(str(target_file)) or [])

    # Статистика по ветвям
    branch_stats = {}

    for start, end in branches:
        if start not in branch_stats:
            branch_stats[start] = {"total": 0, "covered": 0}

        branch_stats[start]["total"] += 1

        # Ветвь покрыта, если и start, и end находятся среди покрытых строк
        if start in covered_lines and abs(end) in covered_lines:
            branch_stats[start]["covered"] += 1

    filtered_branch_stats = {branch: stats for branch, stats in branch_stats.items() if stats["total"] > 1}

    return {
        "branch_stats": filtered_branch_stats,
        "passed_names": passed_names,
        "failed_names": failed_names,
        "xfailed_names": xfailed_names
    }


def print_stats_without_branches(logger, statistics):
    passed_names = statistics["passed_names"]
    failed_names = statistics["failed_names"]
    xfailed_names = statistics["xfailed_names"]

    logger.info(f"\nPassed tests:  {passed_names}")
    logger.info(f"Failed tests:  {failed_names}")
    logger.info(f"Xfailed tests: {xfailed_names}")


def print_stats_with_branches(logger, statistics):
    branch_stats = statistics["branch_stats"]

    logger.info("Branch coverage statistics:")
    for branch, stats in branch_stats.items():
        covered = stats["covered"]
        total = stats["total"]
        logger.info(f"Branch {branch} - {covered}/{total}")


def print_stats(logger, statistics):
    print_stats_with_branches(logger, statistics)
    print_stats_without_branches(logger, statistics)


def compare_coverage(first, second):
    branch_stats_first = first["branch_stats"]
    branch_stats_second = second["branch_stats"]

    for branch, first_stats in branch_stats_first.items():
        if branch not in branch_stats_second:
            return False

        second_stats = branch_stats_second[branch]

        if second_stats["covered"] < first_stats["covered"]:
            return False

    # Все элементы из second должны присутствовать и в first по соответствующим позициям
    compare_passed = set(second["passed_names"]).issubset(set(first["passed_names"]))
    compare_failed = set(second["failed_names"]).issubset(set(first["failed_names"]))
    compare_xfailed = set(second["xfailed_names"]).issubset(set(first["xfailed_names"]))

    return compare_passed and compare_failed and compare_xfailed

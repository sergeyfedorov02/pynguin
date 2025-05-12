from pathlib import Path

from pynguin.reducer.coverage_strategies import get_coverage, print_stats_without_branches
from pynguin.reducer.reducing_strategies import run_reduce, get_reducing_percents_stats, print_reducing_stats
from pynguin.reducer.reducing_env import create_reducing_env, remove_reducing_env

import time


class Reducer:
    def __init__(self, target_file, path_for_result, logger):
        self.target_file = target_file
        self.path_for_result = path_for_result
        self.logger = logger

    def reduce(self):
        copy_target_file_path, flag_bad_name = create_reducing_env(self.target_file, self.path_for_result)

        start_time = time.time()
        original_cov_stat = _get_coverage(copy_target_file_path, self.path_for_result, self.logger)
        initial_time = time.time() - start_time

        self.logger.info("Test cases statistic:")
        print_stats_without_branches(self.logger, original_cov_stat)

        start_time = time.time()
        _reduce(copy_target_file_path, self.path_for_result, original_cov_stat, self.logger, initial_time)
        time_to_reduce = time.time() - start_time

        _print_reducing_stats(self.path_for_result, self.logger, time_to_reduce)

        remove_reducing_env(copy_target_file_path, self.path_for_result, flag_bad_name)


def _get_coverage(target_file: Path, path_for_result: Path, logger):
    try:
        return get_coverage(target_file, path_for_result)
    except Exception as e:
        logger.exception(f"Error: {e}")


def _reduce(target_file: Path, path_for_result: Path, coverage_statistic, logger, original_run_time):
    max_run_time = original_run_time + 1
    try:
        run_reduce(target_file, path_for_result, coverage_statistic, logger, max_run_time)
    except Exception as e:
        logger.exception(f"Error: {e}")


def _print_reducing_stats(path_for_result: Path, logger, time_to_reduce):
    reducing_tests_file = path_for_result.parent / f"reducing_{path_for_result.name}"

    try:
        reducing_stats = get_reducing_percents_stats(path_for_result, reducing_tests_file)
        print_reducing_stats(logger, reducing_stats, time_to_reduce)
    except Exception as e:
        logger.exception(f"Error: {e}")

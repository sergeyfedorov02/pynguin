import os
import subprocess
import unittest
from pathlib import Path
import json
import re

from pynguin.reducer.reducing_strategies import get_reducing_raw_stats, calculate_llm_tokens


def find_files(source_directory):
    for dirpath, dir_names, filenames in os.walk(source_directory):
        dir_names[:] = [d for d in dir_names if d != '__pycache__']
        for filename in filenames:
            yield os.path.join(dirpath, filename)


def save_stats_to_json(data, stat_path):
    with open(stat_path, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=4)


def calculate_total_tokens(data, reducing_file_path):
    llm_reducing_file_path = reducing_file_path.parent / f"llm_{reducing_file_path.name}"

    original_tokens = 0
    reducing_tokens = 0

    for func_stats in data["functions_stats"].values():
        original_tokens += func_stats["original_test_lines"]
        reducing_tokens += func_stats["reducing_test_lines"]

    llm_reducing_tokens = -1
    if Path(llm_reducing_file_path).exists():
        llm_reducing_tokens = calculate_llm_tokens(llm_reducing_file_path)

    new_data = {
        "original_tree_tests_count": data["original_tree_tests_count"],
        "reducing_tree_tests_count": data["reducing_tree_tests_count"],
        "total_original_tokens": data["total_original_tokens"],
        "original_tokens": original_tokens,
        "reducing_tokens": reducing_tokens,
        "llm_reducing_tokens": llm_reducing_tokens,
        "functions_stats": data["functions_stats"],
        "time_to_reduce": data["time_to_reduce"]
    }

    if llm_reducing_tokens is None or llm_reducing_tokens == -1:
        del new_data["llm_reducing_tokens"]

    return new_data


def create_statistic_report(current_file_path, time_to_reduce):
    directory, filename = os.path.split(current_file_path)
    basename, file_extension = os.path.splitext(filename)

    test_file_path = Path(os.path.join(directory.replace("source", "results"), f"test_{basename}.py"))
    reducing_test_file_path = Path(os.path.join(directory.replace("source", "results"), f"reducing_test_{basename}.py"))

    stat_path = os.path.join(directory.replace("source", "result_statistics"), f"{basename}.json")
    new_stat_dir = os.path.dirname(stat_path)
    os.makedirs(new_stat_dir, exist_ok=True)

    try:
        raw_stats = get_reducing_raw_stats(test_file_path, reducing_test_file_path)
        raw_stats["time_to_reduce"] = time_to_reduce

        updated_raw_stats = calculate_total_tokens(raw_stats, reducing_test_file_path)

        save_stats_to_json(updated_raw_stats, stat_path)
    except Exception as e:
        error_data = {
            "error": str(e),
            "message": "Failed to collect statistics."
        }
        save_stats_to_json(error_data, stat_path)


class TestReducingModule(unittest.TestCase):

    def run_main_with_args(self, file_path):
        base_path = file_path.split("\\source\\")[0] + "\\source\\"

        relative_path = file_path[len(base_path):]
        relative_path = relative_path.lstrip("\\")

        path_parts = relative_path.split(os.sep)

        file_name = path_parts[-1]

        module_name = os.path.splitext(file_name)[0]

        project_path = base_path + ("\\".join(path_parts[:-1])) if len(path_parts) > 1 else base_path

        output_path = base_path.replace("source", "results") + ("\\".join(path_parts[:-1])) if len(
            path_parts) > 1 else base_path.replace("source", "results")

        command = [
            'python', '-m', 'pynguin.cli',
            '--algorithm', 'DYNAMOSA',
            '--project-path', project_path,
            '--output-path', output_path,
            '--module-name', module_name,
            '--maximum_search_time', '60',
            '--seed', '42',
            '--statistics-backend', 'CONSOLE',
            '-v',
            '--reducer_module', 'True'
        ]
        result = subprocess.run(command, capture_output=True, text=True)

        time_pattern = r"Time to reduce:\s+.*\n\s+([\d.]+)"
        match = re.search(time_pattern, result.stdout)
        time_to_reduce = float(match.group(1))
        time_to_reduce = float(f"{time_to_reduce:.2f}")

        create_statistic_report(file_path, time_to_reduce)

        return result

    def test_file_processing(self):
        # source_dir = Path(__file__).resolve().parent / 'source' / 'data_structures' / 'hashing' / 'for_test'
        # source_dir = Path(__file__).resolve().parent / 'source' / 'graphs' / 'for_test'
        # source_dir = Path(__file__).resolve().parent / 'source' / 'data_structures' / 'arrays' / 'for_test'
        # source_dir = Path(__file__).resolve().parent / 'source'

        # source_dir = Path(__file__).resolve().parent / 'source' / 'sorts' / 'for_test'
        # source_dir = Path(__file__).resolve().parent / 'source' / 'sorts' / 'for_test2'
        # source_dir = Path(__file__).resolve().parent / 'source' / 'dynamic_programming' / 'for_test'
        # source_dir = Path(__file__).resolve().parent / 'source' / 'graphs' / 'for_test2'
        # source_dir = Path(__file__).resolve().parent / 'source' / 'data_structures' / 'heap' / 'for_test'
        # source_dir = Path(__file__).resolve().parent / 'source' / 'data_structures' / 'heap' / 'for_test2'
        # source_dir = Path(__file__).resolve().parent / 'source' / 'data_structures' / 'binary_tree' / 'for_test'

        source_dir = Path(__file__).resolve().parent / 'source' / 'data_structures' / 'binary_tree' / 'for_test'

        # Проходим по всем файлам в source
        for file_path in find_files(source_dir):
            try:
                self.run_main_with_args(file_path)
            except Exception as e:
                print(f"\nIn file {file_path}")
                print(f"Error: {e}")


if __name__ == '__main__':
    unittest.main()

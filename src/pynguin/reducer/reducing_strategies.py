import ast
from _ast import AST
from pathlib import Path

import copy
import shutil
import tempfile

import multiprocessing

import json

import requests
import re

from pynguin.reducer.coverage_strategies import get_coverage, compare_coverage, pytest_cache_cleanup


def extract_comments(source_code):
    comments = []

    for line in source_code.splitlines():
        line = line.strip()
        if line.startswith('#'):
            # Сохраняем комментарий целиком
            comments.append(line)

    return comments


def remove_pyc_and_cache(file_path):
    # Удаляем .pyc файл
    pyc_file = file_path.with_suffix('.pyc')
    if pyc_file.exists():
        pyc_file.unlink()

    # Удаляем __pycache__ директорию
    pycache_dir = file_path.parent / "__pycache__"
    if pycache_dir.exists():
        shutil.rmtree(pycache_dir)

    pytest_cache_cleanup(file_path)


def integrate_node_into_tree(overall_tree, original_node, updated_node):
    overall_tree_copy = copy.deepcopy(overall_tree)

    # Если замена на верхнем уровне
    if hasattr(original_node, 'name'):
        class ReplaceOrRemoveNode(ast.NodeTransformer):
            def visit_FunctionDef(self, node):
                # Если узел совпадает с original_node -> заменить или удалить
                if node.name == original_node.name:
                    if len(updated_node.body) == 0:
                        return None  # Удалить узел, если тело пустое
                    return updated_node  # Иначе заменить узел
                return node

        # Модифицируем переданное дерево
        new_tree = ReplaceOrRemoveNode().visit(overall_tree_copy)
        ast.fix_missing_locations(new_tree)

        return new_tree

    # Если замена на нижнем уровне
    else:
        def update_node(search_node):
            new_body = []
            end_flag = False

            for node in search_node.body:
                if end_flag:
                    new_body.append(node)

                else:
                    if node.lineno == original_node.lineno:
                        new_body.append(updated_node)
                        end_flag = True
                        continue

                    if hasattr(node, 'body') and isinstance(node.body, list):
                        update_result = update_node(node)
                        node.body = update_result[0]
                        end_flag = update_result[1]

                    new_body.append(node)

            return new_body, end_flag

        updated_body = update_node(overall_tree_copy)[0]
        overall_tree_copy.body = updated_body
        return overall_tree_copy


def is_changed_node_acceptable(overall_tree, test_node, changed_node, original_cov_report, target_file_path,
                               file_with_tests_path, tmp_file_path):
    # Применяем изменения в дереве AST
    modified_tree = integrate_node_into_tree(overall_tree, test_node, changed_node)

    # Сохраняем дерево во временный файл для проверки покрытия
    with tmp_file_path.open("w", encoding="utf-8") as f:
        f.write(ast.unparse(modified_tree))

    # Проверяем покрытие на внесенных изменениях
    remove_pyc_and_cache(tmp_file_path)
    try:
        modified_cov_report = get_coverage(target_file_path, tmp_file_path)
    finally:
        if tmp_file_path.exists():
            tmp_file_path.unlink()  # Удаляем временный файл

    is_acceptable = compare_coverage(original_cov_report, modified_cov_report)

    return is_acceptable


def dead_test_case_elimination(overall_tree, test_node, original_cov_report, target_file_path, file_with_tests_path,
                               time_to_run):
    empty_node = copy.deepcopy(test_node)
    empty_node.body.clear()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".py", dir=file_with_tests_path.parent) as tmp_file:
        tmp_file_path = Path(tmp_file.name)

    # TODO
    is_acceptable = run_with_timeout(
        func=is_changed_node_acceptable,
        timeout=time_to_run,
        overall_tree=overall_tree,
        test_node=test_node,
        changed_node=empty_node,
        original_cov_report=original_cov_report,
        target_file_path=target_file_path,
        file_with_tests_path=file_with_tests_path,
        tmp_file_path=tmp_file_path
    )

    # is_acceptable = is_changed_node_acceptable(
    #     overall_tree=overall_tree,
    #     test_node=test_node,
    #     changed_node=empty_node,
    #     original_cov_report=original_cov_report,
    #     target_file_path=target_file_path,
    #     file_with_tests_path=file_with_tests_path,
    #     tmp_file_path=tmp_file_path
    # )

    if tmp_file_path.exists():
        tmp_file_path.unlink()

    if is_acceptable:
        return True
    return False


def find_function_body(tree, function_name) -> list[AST] | None:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            return node.body
    return None


def restore_asserts(current_node, original_tree):
    overall_body = current_node.body
    original_body = find_function_body(original_tree, current_node.name)

    restored_asserts_test = copy.deepcopy(current_node)
    restored_asserts_test.body.clear()

    original_start_index = 0

    for i in range(len(overall_body)):
        # Узел по текущему индексу
        current_statement = overall_body[i]
        next_statement = overall_body[i + 1] if i + 1 < len(overall_body) else None

        # Флаг для отслеживания окончания добавления assert узлов
        adding_asserts = False

        for j in range(original_start_index, len(original_body)):
            stmt = original_body[j]

            if stmt.lineno == current_statement.lineno:
                # Начало добавления assert узлов после текущего узла
                restored_asserts_test.body.append(current_statement)
                adding_asserts = True

            elif next_statement and stmt.lineno == next_statement.lineno:
                # Завершаем добавление перед следующим узлом
                adding_asserts = False
                original_start_index = j
                break
            elif adding_asserts:
                if isinstance(stmt, ast.Assert):
                    restored_asserts_test.body.append(stmt)
                else:
                    # Если среди совпадающих узлов обнаружен не assert, останавливаем добавление
                    adding_asserts = False
                    break
            else:
                # Если next_statement отсутствует или не найдено, обновляем original_start_index
                original_start_index = len(original_body)

    return restored_asserts_test


def wrapper(func, result_queue, **kwargs):
    try:
        result = func(**kwargs)
        result_queue.put(result)
    except Exception as e:
        result_queue.put(e)


def run_with_timeout(func, timeout, **kwargs):
    result_queue = multiprocessing.SimpleQueue()
    process = multiprocessing.Process(target=wrapper, args=(func, result_queue), kwargs=kwargs)

    process.start()  # запуск
    process.join(timeout)  # ожидание завершения

    if process.is_alive():
        process.terminate()  # принудительное завершение
        process.join()
        return False

    if not result_queue.empty():
        result = result_queue.get()
        if isinstance(result, Exception):
            raise result
        return result

    return False


def hierarchical_delta_debugging(test_node, target_file_path, original_cov_report, overall_tree, file_with_tests_path,
                                 time_to_run):
    def process_node_body(node, tree):
        body = node.body

        if not body:
            return node

        reduced_node = copy.deepcopy(node)
        n = 2

        while n <= len(body):
            chunk_size = max(1, len(body) // n)
            chunks = [body[i:i + chunk_size] for i in range(0, len(body), chunk_size)]

            reduced = False

            for chunk_index, chunk in enumerate(chunks):
                new_body = [stmt for stmt in body if stmt not in chunk]
                reduced_node.body = new_body

                with tempfile.NamedTemporaryFile(delete=False, suffix=".py",
                                                 dir=file_with_tests_path.parent) as tmp_file:
                    tmp_file_path = Path(tmp_file.name)

                # TODO
                is_acceptable = run_with_timeout(
                    func=is_changed_node_acceptable,
                    timeout=time_to_run,
                    overall_tree=tree,
                    test_node=node,
                    changed_node=reduced_node,
                    original_cov_report=original_cov_report,
                    target_file_path=target_file_path,
                    file_with_tests_path=file_with_tests_path,
                    tmp_file_path=tmp_file_path
                )

                # is_acceptable = is_changed_node_acceptable(
                #     overall_tree=tree,
                #     test_node=node,
                #     changed_node=reduced_node,
                #     original_cov_report=original_cov_report,
                #     target_file_path=target_file_path,
                #     file_with_tests_path=file_with_tests_path,
                #     tmp_file_path=tmp_file_path
                # )

                if tmp_file_path.exists():
                    tmp_file_path.unlink()

                if is_acceptable:
                    body = new_body
                    reduced = True

                    # Сохраняем изменения
                    tree = integrate_node_into_tree(tree, node, reduced_node)
                    node = copy.deepcopy(reduced_node)
                    break

            if reduced:
                n = max(n - 1, 2)
            else:
                if n == len(body):
                    break
                n = min(n * 2, len(body))

        reduced_node.body = body
        return reduced_node

    def process_node_recursively(node, tree, node_name):
        test_tree = copy.deepcopy(tree)

        # Сначала обрабатываем тело узла
        reduced_node_ast = process_node_body(node, test_tree)
        test_tree = integrate_node_into_tree(test_tree, node, reduced_node_ast)

        # Рекурсивно обрабатываем прямых потомков в теле узла
        reduced_node_copy = copy.deepcopy(reduced_node_ast)
        for stmt in reduced_node_copy.body:
            if hasattr(stmt, 'body') and isinstance(stmt.body, list):
                reduced_stmt = process_node_recursively(stmt, test_tree, node_name)
                test_tree = integrate_node_into_tree(test_tree, stmt, reduced_stmt)

                reduced_node_ast.body = find_function_body(test_tree, node_name)

        return reduced_node_ast

    # Начинаем с обработки корневого узла
    return process_node_recursively(test_node, overall_tree, test_node.name)


def modify_value(value):
    if isinstance(value.value, bool):
        return value

    elif isinstance(value.value, int):
        value.value = 1 if value.value > 0 else -1

    elif isinstance(value.value, float):
        value.value = 0.1 if value.value > 0 else -0.1

    elif isinstance(value.value, complex):
        return value

    elif isinstance(value.value, str):
        value.value = "test"

    return value


def creating_specific_modifications(test_node, target_file_path, original_cov_report, overall_tree,
                                    file_with_tests_path, time_to_run):
    test_node_copy = copy.deepcopy(test_node)
    for current_node in ast.iter_child_nodes(test_node_copy):
        if isinstance(current_node, ast.Assign):
            for target, value in zip(current_node.targets, [current_node.value]):

                check_flag = False
                old_value = 0
                new_value = 0
                node_to_modify = None

                if isinstance(value, ast.UnaryOp):
                    value_operand = value.operand
                    if isinstance(value.op, ast.USub) and isinstance(value_operand, ast.Constant):
                        old_value = copy.deepcopy(value_operand)
                        node_to_modify = value_operand
                        new_value = modify_value(value_operand)
                        check_flag = True

                elif isinstance(value, ast.Constant):
                    old_value = copy.deepcopy(value)
                    node_to_modify = value
                    new_value = modify_value(value)
                    check_flag = True

                if check_flag:
                    if old_value.value != new_value.value:

                        with tempfile.NamedTemporaryFile(delete=False, suffix=".py",
                                                         dir=file_with_tests_path.parent) as tmp_file:
                            tmp_file_path = Path(tmp_file.name)

                        # TODO
                        is_acceptable = run_with_timeout(
                            func=is_changed_node_acceptable,
                            timeout=time_to_run,
                            overall_tree=overall_tree,
                            test_node=test_node,
                            changed_node=test_node_copy,
                            original_cov_report=original_cov_report,
                            target_file_path=target_file_path,
                            file_with_tests_path=file_with_tests_path,
                            tmp_file_path=tmp_file_path
                        )

                        # is_acceptable = is_changed_node_acceptable(
                        #     overall_tree=overall_tree,
                        #     test_node=test_node,
                        #     changed_node=test_node_copy,
                        #     original_cov_report=original_cov_report,
                        #     target_file_path=target_file_path,
                        #     file_with_tests_path=file_with_tests_path,
                        #     tmp_file_path=tmp_file_path
                        # )

                        if tmp_file_path.exists():
                            tmp_file_path.unlink()

                        # Сохраняем изменения или возвращаем предыдущее значение
                        if is_acceptable:
                            overall_tree = integrate_node_into_tree(overall_tree, test_node,
                                                                    test_node_copy)
                            test_node = copy.deepcopy(test_node_copy)
                        else:
                            node_to_modify.value = old_value.value

    return test_node


def try_create_ast(file_path):
    with file_path.open("r", encoding="utf-8") as f:
        source_code = f.read()

    comments = extract_comments(source_code)

    try:
        ast_tree = ast.parse(source_code)
    except SyntaxError as e:
        raise ValueError(f"Syntax error in file {file_path}: {e}")

    return {
        "ast_tree": ast_tree,
        "comments": comments
    }


def run_reduce(target_file_path: Path, file_with_tests_path: Path, original_coverage_report, logger, time_to_run):
    # Получим AST структуру
    creating_ast_structure = try_create_ast(file_with_tests_path)
    overall_tree = creating_ast_structure["ast_tree"]
    comments = creating_ast_structure["comments"]

    reducing_tests_path = file_with_tests_path.parent / f"reducing_{file_with_tests_path.name}"

    overall_tree_copy = copy.deepcopy(overall_tree)

    test_cases_count = sum(1 for node in ast.walk(overall_tree_copy) if isinstance(node, ast.FunctionDef))
    current_test_case = 0

    # Проходим по всем узлам дерева и минимизируем функции
    for node in ast.iter_child_nodes(overall_tree_copy):
        if isinstance(node, ast.FunctionDef):

            current_test_case += 1
            logger.info(f"Reducing <{node.name}>: {current_test_case}/{test_cases_count}")

            # Если тест никак не влияет на покрытие -> удаляем его сразу
            is_test_dead = dead_test_case_elimination(
                overall_tree=overall_tree,
                test_node=node,
                original_cov_report=original_coverage_report,
                target_file_path=target_file_path,
                file_with_tests_path=file_with_tests_path,
                time_to_run=time_to_run
            )

            if is_test_dead:
                empty_node = copy.deepcopy(node)
                empty_node.body.clear()
                overall_tree = integrate_node_into_tree(overall_tree, node, empty_node)
                continue

            # Тест как-то влияет на покрытие -> delta-debugging
            reduced_test_ast = hierarchical_delta_debugging(
                test_node=node,
                target_file_path=target_file_path,
                original_cov_report=original_coverage_report,
                overall_tree=overall_tree,
                file_with_tests_path=file_with_tests_path,
                time_to_run=time_to_run
            )

            # Обновляем общее дерево
            overall_tree = integrate_node_into_tree(overall_tree, node, reduced_test_ast)

    # Возвращаем значимые asserts
    for node in ast.iter_child_nodes(overall_tree):
        if isinstance(node, ast.FunctionDef):
            restored_asserts_ast = restore_asserts(
                current_node=node,
                original_tree=overall_tree_copy
            )
            overall_tree = integrate_node_into_tree(overall_tree, node, restored_asserts_ast)

    # Применяем специфичные модификации по замене assign
    logger.info(f"Creating specific modifications...")
    for node in ast.iter_child_nodes(overall_tree):
        if isinstance(node, ast.FunctionDef):
            specific_changes = copy.deepcopy(node)
            try:
                specific_changes = creating_specific_modifications(
                    test_node=node,
                    target_file_path=target_file_path,
                    original_cov_report=original_coverage_report,
                    overall_tree=overall_tree,
                    file_with_tests_path=file_with_tests_path,
                    time_to_run=time_to_run
                )
            except ValueError:
                pass
            overall_tree = integrate_node_into_tree(overall_tree, node, specific_changes)

    # Сохраняем финальное дерево в файл
    with reducing_tests_path.open("w", encoding="utf-8") as reduced_file:
        for comment in comments:
            reduced_file.write(comment + "\n")
        reduced_file.write(ast.unparse(overall_tree))

    # Взаимодействие с LLM
    logger.info(f"Application of LLM...")
    try:
        reducing_with_llm(
            target_file_path=target_file_path,
            reducing_tests_path=reducing_tests_path,
            original_cov_report=original_coverage_report,
            time_to_run=time_to_run
        )
        logger.info(f"Interaction with LLM was successful")
    except ValueError:
        logger.info(f"Interaction with LLM was NOT successful")
        pass

    remove_pyc_and_cache(reducing_tests_path)


def reducing_with_llm(target_file_path, reducing_tests_path, original_cov_report, time_to_run):
    llm_reducing_tests_path = reducing_tests_path.parent / f"llm_{reducing_tests_path.name}"

    with target_file_path.open("r", encoding="utf-8") as f:
        target_file_code = f.read()

    with reducing_tests_path.open("r", encoding="utf-8") as f:
        reducing_file_code = f.read()

    for i in range(3):
        try:
            response = make_request_llm(target_file_code, reducing_file_code)
        except requests.RequestException:
            pass
            continue

        if response.status_code == 200:
            result = response.json()
            result_answer = result['choices'][0]['message']['content']

            code_match = re.search(r"```python(.*?)```", result_answer, re.DOTALL)

            if code_match:
                llm_reducing_code = code_match.group(1).strip()

                # Проверяем, что не изменилось покрытие
                with tempfile.NamedTemporaryFile(delete=False, suffix=".py",
                                                 dir=reducing_tests_path.parent) as tmp_file:
                    tmp_file_path = Path(tmp_file.name)

                # TODO
                is_acceptable = run_with_timeout(
                    func=check_coverage_after_llm_reducing,
                    timeout=time_to_run,
                    target_file_path=target_file_path,
                    original_cov_report=original_cov_report,
                    llm_reducing_code=llm_reducing_code,
                    tmp_file_path=tmp_file_path
                )

                # is_acceptable = check_coverage_after_llm_reducing(
                #     target_file_path=target_file_path,
                #     original_cov_report=original_cov_report,
                #     llm_reducing_code=llm_reducing_code,
                #     tmp_file_path=tmp_file_path
                # )

                if tmp_file_path.exists():
                    tmp_file_path.unlink()

                if is_acceptable:
                    with llm_reducing_tests_path.open("w", encoding="utf-8") as llm_reducing_file:
                        llm_reducing_file.write(llm_reducing_code)
                    return True

    return False


def check_coverage_after_llm_reducing(target_file_path, original_cov_report, llm_reducing_code, tmp_file_path):
    with tmp_file_path.open("w", encoding="utf-8") as f:
        f.write(llm_reducing_code)

    # Проверяем покрытие на внесенных изменениях
    remove_pyc_and_cache(tmp_file_path)
    try:
        modified_cov_report = get_coverage(target_file_path, tmp_file_path)
    finally:
        if tmp_file_path.exists():
            tmp_file_path.unlink()

    is_acceptable = compare_coverage(original_cov_report, modified_cov_report)

    return is_acceptable


def make_request_llm(target_code, reducing_code):
    api_key = "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"
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    prompt = f"Improve the readability of the following code with Python tests. " \
             f"Don't change the name of the tests. " \
             f"In response, only the corrected code with tests was sent, and highlight it in python '''." \
             f"Do not change the count of tests tests. " \
             f"Try reducing the count of tokens in the tests themselves." \
             f"Add comments to clarify what is happening in the test. " \
             f"The file for which the tests were written:\n{target_code}\n" \
             f"The file with the tests itself:\n {reducing_code}"
    data = json.dumps({
        "model": "deepseek/deepseek-chat-v3-0324:free",
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ]

    })
    response = requests.post(
        url=url,
        headers=headers,
        data=data
    )
    return response


def calculate_llm_tokens(llm_reducing_file_path):
    llm_reducing_tree = try_create_ast(llm_reducing_file_path)["ast_tree"]
    llm_tokens = 0

    for node in ast.iter_child_nodes(llm_reducing_tree):
        if isinstance(node, ast.FunctionDef):
            llm_tokens += count_lines(node.body)

    return llm_tokens


def count_lines(body):
    count = 0
    for node in body:
        count += 1
        if hasattr(node, 'body') and isinstance(node.body, list):
            count += count_lines(node.body)
    return count


def get_reducing_raw_stats(original_tests_file_path, reducing_tests_file_path):
    original_tree = try_create_ast(original_tests_file_path)["ast_tree"]
    reducing_tree = try_create_ast(reducing_tests_file_path)["ast_tree"]

    original_tree_tests_count = sum(1 for node in ast.walk(original_tree) if isinstance(node, ast.FunctionDef))
    reducing_tree_tests_count = 0

    functions_stats = {}

    for node in ast.iter_child_nodes(reducing_tree):
        if isinstance(node, ast.FunctionDef):
            reducing_tree_tests_count += 1

            reducing_test_lines = count_lines(node.body)

            original_test_body = find_function_body(original_tree, node.name)
            original_test_lines = count_lines(original_test_body)

            functions_stats[f"{node.name}"] = {
                "original_test_lines": original_test_lines,
                "reducing_test_lines": reducing_test_lines,
            }

    return {
        "original_tree_tests_count": original_tree_tests_count,
        "reducing_tree_tests_count": reducing_tree_tests_count,
        "functions_stats": functions_stats,
    }


def get_reducing_percents_stats(original_tests_file_path, reducing_tests_file_path):
    raw_stats = get_reducing_raw_stats(original_tests_file_path, reducing_tests_file_path)

    original_tree_tests_count = raw_stats["original_tree_tests_count"]
    reducing_tree_tests_count = raw_stats["reducing_tree_tests_count"]
    functions_stats = raw_stats["functions_stats"]

    tests_lines_stat = {}
    for function_name, stats in functions_stats.items():
        original_test_lines = stats["original_test_lines"]
        reducing_test_lines = stats["reducing_test_lines"]

        tests_lines_stat[function_name] = (1 - reducing_test_lines / original_test_lines) * 100

    tests_count_stat = (1 - reducing_tree_tests_count / original_tree_tests_count) * 100

    return tests_count_stat, tests_lines_stat


def print_reducing_stats(logger, reducing_statistic, time_to_reduce):
    logger.info("Reducing statistic:")

    tests_percent = round(reducing_statistic[0], 2)
    logger.info(f"The number of tests has been reduced by {tests_percent} %")

    logger.info("Statistics for remaining tests:")
    for test_line_stat in reducing_statistic[1]:
        test_case_percent = round(reducing_statistic[1][test_line_stat], 2)
        logger.info(f"Test with name \"{test_line_stat}\" was reduced by {test_case_percent} %")

    logger.info(f"Time to reduce: {time_to_reduce}")

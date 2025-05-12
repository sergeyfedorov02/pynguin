from pathlib import Path
import os
import shutil

import random
import string


def change_expr_in_file(file_path, old_expr, new_expr, flag_bad_name):
    with file_path.open('r', encoding='utf-8') as file:
        content = file.readlines()

    if flag_bad_name is not None:
        updated_content = [line.replace(old_expr, new_expr) if 'import' in line else line for line in content]
    else:
        updated_content = [line.replace(old_expr, new_expr) for line in content]

    with file_path.open('w', encoding='utf-8') as file:
        file.writelines(updated_content)


def create_reducing_env(target_file: Path, path_for_result: Path):
    old_name = target_file.stem
    new_name = target_file.stem + "_2"

    bad_name_flag = None

    if old_name[:2] == "__" or old_name[-2:] == "__":
        characters = string.ascii_letters + string.digits
        random_string = ''.join(random.choices(characters, k=10))
        new_name = "very_good_name" + random_string
        bad_name_flag = old_name

    change_expr_in_file(path_for_result, old_name, new_name, bad_name_flag)

    destination_directory = path_for_result.parent
    copy_target_file_path = destination_directory / f"{new_name}{target_file.suffix}"
    shutil.copy(target_file, copy_target_file_path)

    change_expr_in_file(copy_target_file_path, old_name, new_name, bad_name_flag)

    return copy_target_file_path, bad_name_flag


def remove_reducing_env(target_file: Path, path_for_result: Path, bad_name_flag):
    old_name = target_file.stem
    new_name = target_file.stem[:-2]

    if bad_name_flag is not None:
        new_name = bad_name_flag

    os.remove(target_file)

    change_expr_in_file(path_for_result, old_name, new_name, bad_name_flag)

    reducing_file_path = path_for_result.parent / f"reducing_{path_for_result.name}"
    change_expr_in_file(reducing_file_path, old_name, new_name, bad_name_flag)

    llm_reducing_file_path = path_for_result.parent / f"llm_reducing_{path_for_result.name}"
    if Path(llm_reducing_file_path).exists():
        change_expr_in_file(llm_reducing_file_path, old_name, new_name, bad_name_flag)

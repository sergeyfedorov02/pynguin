# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import pytest
import greedy_min_vertex_cover as module_0


def test_case_0():
    bytes_0 = b"\xb6\xce|"
    dict_0 = {bytes_0: bytes_0, bytes_0: bytes_0}
    set_0 = module_0.greedy_min_vertex_cover(dict_0)


def test_case_1():
    dict_0 = {}
    set_0 = module_0.greedy_min_vertex_cover(dict_0)


@pytest.mark.xfail(strict=True)
def test_case_2():
    none_type_0 = None
    module_0.greedy_min_vertex_cover(none_type_0)


def test_case_3():
    tuple_0 = ()
    dict_0 = {tuple_0: tuple_0}
    set_0 = module_0.greedy_min_vertex_cover(dict_0)
    str_0 = ",:Y4mzPi<:^_P4O"
    dict_1 = {str_0: str_0, tuple_0: dict_0, str_0: set_0}
    set_1 = module_0.greedy_min_vertex_cover(dict_1)


@pytest.mark.xfail(strict=True)
def test_case_4():
    str_0 = "vnk|^n;^)q>r"
    str_1 = "Te"
    dict_0 = {str_0: str_1, str_1: str_1, str_0: str_1, str_1: str_0}
    module_0.greedy_min_vertex_cover(dict_0)


def test_case_5():
    tuple_0 = ()
    dict_0 = {tuple_0: tuple_0}
    str_0 = "VzX~~\nyLgSd}zVKT&m"
    dict_1 = {str_0: str_0, tuple_0: dict_0}
    set_0 = module_0.greedy_min_vertex_cover(dict_1)
    set_1 = module_0.greedy_min_vertex_cover(dict_0)

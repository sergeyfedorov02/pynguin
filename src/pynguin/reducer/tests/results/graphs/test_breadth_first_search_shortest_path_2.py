# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import pytest
import breadth_first_search_shortest_path_2 as module_0


@pytest.mark.xfail(strict=True)
def test_case_0():
    bytes_0 = b"\xdb\x193e\x93\x06E] h}\\@"
    list_0 = module_0.bfs_shortest_path(bytes_0, bytes_0, bytes_0)
    assert module_0.demo_graph == {
        "A": ["B", "C", "E"],
        "B": ["A", "D", "E"],
        "C": ["A", "F", "G"],
        "D": ["B"],
        "E": ["A", "B", "D"],
        "F": ["C"],
        "G": ["C"],
    }
    module_0.bfs_shortest_path(bytes_0, list_0, bytes_0)


@pytest.mark.xfail(strict=True)
def test_case_1():
    complex_0 = -1225 - 1703.8853741896612j
    list_0 = [complex_0]
    module_0.bfs_shortest_path(complex_0, complex_0, list_0)


def test_case_2():
    str_0 = "u"
    int_0 = module_0.bfs_shortest_path_distance(str_0, str_0, str_0)
    assert int_0 == 0
    assert module_0.demo_graph == {
        "A": ["B", "C", "E"],
        "B": ["A", "D", "E"],
        "C": ["A", "F", "G"],
        "D": ["B"],
        "E": ["A", "B", "D"],
        "F": ["C"],
        "G": ["C"],
    }


def test_case_3():
    none_type_0 = None
    int_0 = module_0.bfs_shortest_path_distance(none_type_0, none_type_0, none_type_0)
    assert int_0 == -1
    assert module_0.demo_graph == {
        "A": ["B", "C", "E"],
        "B": ["A", "D", "E"],
        "C": ["A", "F", "G"],
        "D": ["B"],
        "E": ["A", "B", "D"],
        "F": ["C"],
        "G": ["C"],
    }


def test_case_4():
    str_0 = ".+"
    none_type_0 = None
    str_1 = 'iVd0! iLnFTWd5?t#x"'
    dict_0 = {str_0: none_type_0}
    int_0 = module_0.bfs_shortest_path_distance(dict_0, none_type_0, str_1)
    assert int_0 == -1
    assert module_0.demo_graph == {
        "A": ["B", "C", "E"],
        "B": ["A", "D", "E"],
        "C": ["A", "F", "G"],
        "D": ["B"],
        "E": ["A", "B", "D"],
        "F": ["C"],
        "G": ["C"],
    }
    str_2 = 't{Y"TH=Q'
    dict_1 = {str_2: str_2, str_0: str_0}
    int_1 = module_0.bfs_shortest_path_distance(dict_1, str_0, str_2)
    assert int_1 == -1


def test_case_5():
    str_0 = "u"
    list_0 = module_0.bfs_shortest_path(str_0, str_0, str_0)
    assert module_0.demo_graph == {
        "A": ["B", "C", "E"],
        "B": ["A", "D", "E"],
        "C": ["A", "F", "G"],
        "D": ["B"],
        "E": ["A", "B", "D"],
        "F": ["C"],
        "G": ["C"],
    }
    none_type_0 = None
    int_0 = module_0.bfs_shortest_path_distance(list_0, str_0, none_type_0)
    assert int_0 == -1


@pytest.mark.xfail(strict=True)
def test_case_6():
    str_0 = 'i0! iLnFTWd5?t#x"'
    dict_0 = {str_0: str_0, str_0: str_0, str_0: str_0}
    module_0.bfs_shortest_path(dict_0, str_0, dict_0)


@pytest.mark.xfail(strict=True)
def test_case_7():
    str_0 = "u"
    int_0 = 31
    dict_0 = {str_0: str_0, str_0: int_0, int_0: str_0}
    module_0.bfs_shortest_path_distance(dict_0, str_0, int_0)


@pytest.mark.xfail(strict=True)
def test_case_8():
    bool_0 = False
    none_type_0 = None
    int_0 = module_0.bfs_shortest_path_distance(bool_0, none_type_0, bool_0)
    assert int_0 == -1
    assert module_0.demo_graph == {
        "A": ["B", "C", "E"],
        "B": ["A", "D", "E"],
        "C": ["A", "F", "G"],
        "D": ["B"],
        "E": ["A", "B", "D"],
        "F": ["C"],
        "G": ["C"],
    }
    dict_0 = {bool_0: bool_0, bool_0: bool_0, bool_0: bool_0}
    bool_1 = False
    int_1 = module_0.bfs_shortest_path_distance(dict_0, bool_0, bool_1)
    assert int_1 == 0
    list_0 = module_0.bfs_shortest_path(bool_1, bool_1, bool_1)
    int_2 = module_0.bfs_shortest_path_distance(dict_0, bool_0, none_type_0)
    assert int_2 == -1
    int_3 = module_0.bfs_shortest_path_distance(dict_0, none_type_0, int_1)
    assert int_3 == -1
    list_1 = module_0.bfs_shortest_path(none_type_0, int_2, int_3)
    dict_1 = {int_3: int_2, bool_0: list_1, int_2: bool_1}
    list_2 = module_0.bfs_shortest_path(dict_1, bool_0, int_0)
    list_3 = module_0.bfs_shortest_path(dict_0, none_type_0, none_type_0)
    module_0.bfs_shortest_path(dict_0, bool_1, none_type_0)


def test_case_9():
    dict_0 = {}
    list_0 = module_0.bfs_shortest_path(dict_0, dict_0, dict_0)
    assert module_0.demo_graph == {
        "A": ["B", "C", "E"],
        "B": ["A", "D", "E"],
        "C": ["A", "F", "G"],
        "D": ["B"],
        "E": ["A", "B", "D"],
        "F": ["C"],
        "G": ["C"],
    }
    bool_0 = True
    str_0 = ".+"
    str_1 = 'i0! iLnFTWd5?t#x"'
    str_2 = "V"
    list_1 = [str_0, str_1, str_2]
    dict_1 = {str_2: str_2, str_0: str_0}
    int_0 = module_0.bfs_shortest_path_distance(dict_0, str_1, bool_0)
    assert int_0 == -1
    int_1 = module_0.bfs_shortest_path_distance(dict_1, str_0, str_2)
    assert int_1 == -1
    bool_1 = False
    list_2 = module_0.bfs_shortest_path(bool_1, bool_1, bool_1)
    list_3 = module_0.bfs_shortest_path(dict_1, str_2, list_1)


def test_case_10():
    str_0 = "f"
    str_1 = "V"
    dict_0 = {str_1: str_1, str_0: str_1}
    int_0 = module_0.bfs_shortest_path_distance(dict_0, str_0, str_1)
    assert int_0 == 1
    assert module_0.demo_graph == {
        "A": ["B", "C", "E"],
        "B": ["A", "D", "E"],
        "C": ["A", "F", "G"],
        "D": ["B"],
        "E": ["A", "B", "D"],
        "F": ["C"],
        "G": ["C"],
    }

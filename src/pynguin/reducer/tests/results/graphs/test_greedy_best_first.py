# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import pytest
import greedy_best_first as module_0


def test_case_0():
    bool_0 = False
    bool_1 = True
    list_0 = [bool_0, bool_1, bool_1]
    list_1 = [list_0]
    tuple_0 = (bool_0, bool_1)
    greedy_best_first_0 = module_0.GreedyBestFirst(list_1, tuple_0, tuple_0)
    assert (
        f"{type(greedy_best_first_0).__module__}.{type(greedy_best_first_0).__qualname__}"
        == "greedy_best_first.GreedyBestFirst"
    )
    assert greedy_best_first_0.grid == [[False, True, True]]
    assert (
        f"{type(greedy_best_first_0.start).__module__}.{type(greedy_best_first_0.start).__qualname__}"
        == "greedy_best_first.Node"
    )
    assert (
        f"{type(greedy_best_first_0.target).__module__}.{type(greedy_best_first_0.target).__qualname__}"
        == "greedy_best_first.Node"
    )
    assert (
        f"{type(greedy_best_first_0.open_nodes).__module__}.{type(greedy_best_first_0.open_nodes).__qualname__}"
        == "builtins.list"
    )
    assert len(greedy_best_first_0.open_nodes) == 1
    assert greedy_best_first_0.closed_nodes == []
    assert greedy_best_first_0.reached is False
    assert module_0.delta == ([-1, 0], [0, -1], [1, 0], [0, 1])
    var_0 = greedy_best_first_0.search()
    assert greedy_best_first_0.open_nodes == []
    assert greedy_best_first_0.reached is True


def test_case_1():
    bool_0 = False
    list_0 = [bool_0, bool_0, bool_0]
    list_1 = [list_0]
    bool_1 = True
    tuple_0 = (bool_0, bool_1)
    greedy_best_first_0 = module_0.GreedyBestFirst(list_1, tuple_0, list_0)
    assert (
        f"{type(greedy_best_first_0).__module__}.{type(greedy_best_first_0).__qualname__}"
        == "greedy_best_first.GreedyBestFirst"
    )
    assert greedy_best_first_0.grid == [[False, False, False]]
    assert (
        f"{type(greedy_best_first_0.start).__module__}.{type(greedy_best_first_0.start).__qualname__}"
        == "greedy_best_first.Node"
    )
    assert (
        f"{type(greedy_best_first_0.target).__module__}.{type(greedy_best_first_0.target).__qualname__}"
        == "greedy_best_first.Node"
    )
    assert (
        f"{type(greedy_best_first_0.open_nodes).__module__}.{type(greedy_best_first_0.open_nodes).__qualname__}"
        == "builtins.list"
    )
    assert len(greedy_best_first_0.open_nodes) == 1
    assert greedy_best_first_0.closed_nodes == []
    assert greedy_best_first_0.reached is False
    assert module_0.delta == ([-1, 0], [0, -1], [1, 0], [0, 1])
    var_0 = greedy_best_first_0.search()
    assert len(greedy_best_first_0.closed_nodes) == 1
    assert greedy_best_first_0.reached is True


def test_case_2():
    none_type_0 = None
    int_0 = -2328
    list_0 = [int_0, int_0]
    list_1 = [list_0, list_0, list_0, list_0]
    bool_0 = True
    bool_1 = True
    tuple_0 = (bool_0, bool_1)
    greedy_best_first_0 = module_0.GreedyBestFirst(list_1, tuple_0, tuple_0)
    assert (
        f"{type(greedy_best_first_0).__module__}.{type(greedy_best_first_0).__qualname__}"
        == "greedy_best_first.GreedyBestFirst"
    )
    assert greedy_best_first_0.grid == [
        [-2328, -2328],
        [-2328, -2328],
        [-2328, -2328],
        [-2328, -2328],
    ]
    assert (
        f"{type(greedy_best_first_0.start).__module__}.{type(greedy_best_first_0.start).__qualname__}"
        == "greedy_best_first.Node"
    )
    assert (
        f"{type(greedy_best_first_0.target).__module__}.{type(greedy_best_first_0.target).__qualname__}"
        == "greedy_best_first.Node"
    )
    assert (
        f"{type(greedy_best_first_0.open_nodes).__module__}.{type(greedy_best_first_0.open_nodes).__qualname__}"
        == "builtins.list"
    )
    assert len(greedy_best_first_0.open_nodes) == 1
    assert greedy_best_first_0.closed_nodes == []
    assert greedy_best_first_0.reached is False
    assert module_0.delta == ([-1, 0], [0, -1], [1, 0], [0, 1])
    list_2 = greedy_best_first_0.retrace_path(none_type_0)


@pytest.mark.xfail(strict=True)
def test_case_3():
    dict_0 = {}
    none_type_0 = None
    module_0.Node(dict_0, none_type_0, dict_0, none_type_0, dict_0, none_type_0)


@pytest.mark.xfail(strict=True)
def test_case_4():
    int_0 = -1007
    tuple_0 = (int_0, int_0)
    list_0 = [tuple_0, tuple_0, tuple_0, tuple_0]
    list_1 = [tuple_0]
    module_0.GreedyBestFirst(list_0, tuple_0, list_1)


def test_case_5():
    bool_0 = False
    list_0 = [bool_0, bool_0]
    list_1 = [list_0]
    bool_1 = True
    tuple_0 = (bool_1, bool_1)
    greedy_best_first_0 = module_0.GreedyBestFirst(list_1, tuple_0, list_0)
    assert (
        f"{type(greedy_best_first_0).__module__}.{type(greedy_best_first_0).__qualname__}"
        == "greedy_best_first.GreedyBestFirst"
    )
    assert greedy_best_first_0.grid == [[False, False]]
    assert (
        f"{type(greedy_best_first_0.start).__module__}.{type(greedy_best_first_0.start).__qualname__}"
        == "greedy_best_first.Node"
    )
    assert (
        f"{type(greedy_best_first_0.target).__module__}.{type(greedy_best_first_0.target).__qualname__}"
        == "greedy_best_first.Node"
    )
    assert (
        f"{type(greedy_best_first_0.open_nodes).__module__}.{type(greedy_best_first_0.open_nodes).__qualname__}"
        == "builtins.list"
    )
    assert len(greedy_best_first_0.open_nodes) == 1
    assert greedy_best_first_0.closed_nodes == []
    assert greedy_best_first_0.reached is False
    assert module_0.delta == ([-1, 0], [0, -1], [1, 0], [0, 1])
    var_0 = greedy_best_first_0.search()
    assert greedy_best_first_0.open_nodes == []
    assert len(greedy_best_first_0.closed_nodes) == 2
    assert greedy_best_first_0.reached is True
    list_2 = greedy_best_first_0.search()


def test_case_6():
    bool_0 = True
    list_0 = [bool_0, bool_0, bool_0]
    list_1 = [list_0]
    bool_1 = False
    tuple_0 = (bool_0, bool_1)
    greedy_best_first_0 = module_0.GreedyBestFirst(list_1, tuple_0, list_0)
    assert (
        f"{type(greedy_best_first_0).__module__}.{type(greedy_best_first_0).__qualname__}"
        == "greedy_best_first.GreedyBestFirst"
    )
    assert greedy_best_first_0.grid == [[True, True, True]]
    assert (
        f"{type(greedy_best_first_0.start).__module__}.{type(greedy_best_first_0.start).__qualname__}"
        == "greedy_best_first.Node"
    )
    assert (
        f"{type(greedy_best_first_0.target).__module__}.{type(greedy_best_first_0.target).__qualname__}"
        == "greedy_best_first.Node"
    )
    assert (
        f"{type(greedy_best_first_0.open_nodes).__module__}.{type(greedy_best_first_0.open_nodes).__qualname__}"
        == "builtins.list"
    )
    assert len(greedy_best_first_0.open_nodes) == 1
    assert greedy_best_first_0.closed_nodes == []
    assert greedy_best_first_0.reached is False
    assert module_0.delta == ([-1, 0], [0, -1], [1, 0], [0, 1])
    var_0 = greedy_best_first_0.search()
    assert greedy_best_first_0.open_nodes == []
    assert len(greedy_best_first_0.closed_nodes) == 1


def test_case_7():
    bool_0 = False
    list_0 = [bool_0, bool_0]
    list_1 = [list_0]
    bool_1 = True
    tuple_0 = (bool_1, bool_1)
    greedy_best_first_0 = module_0.GreedyBestFirst(list_1, tuple_0, list_0)
    assert (
        f"{type(greedy_best_first_0).__module__}.{type(greedy_best_first_0).__qualname__}"
        == "greedy_best_first.GreedyBestFirst"
    )
    assert greedy_best_first_0.grid == [[False, False]]
    assert (
        f"{type(greedy_best_first_0.start).__module__}.{type(greedy_best_first_0.start).__qualname__}"
        == "greedy_best_first.Node"
    )
    assert (
        f"{type(greedy_best_first_0.target).__module__}.{type(greedy_best_first_0.target).__qualname__}"
        == "greedy_best_first.Node"
    )
    assert (
        f"{type(greedy_best_first_0.open_nodes).__module__}.{type(greedy_best_first_0.open_nodes).__qualname__}"
        == "builtins.list"
    )
    assert len(greedy_best_first_0.open_nodes) == 1
    assert greedy_best_first_0.closed_nodes == []
    assert greedy_best_first_0.reached is False
    assert module_0.delta == ([-1, 0], [0, -1], [1, 0], [0, 1])
    var_0 = greedy_best_first_0.search()
    assert greedy_best_first_0.open_nodes == []
    assert len(greedy_best_first_0.closed_nodes) == 2
    assert greedy_best_first_0.reached is True


def test_case_8():
    bool_0 = False
    list_0 = [bool_0, bool_0, bool_0]
    list_1 = [list_0, list_0]
    bool_1 = True
    tuple_0 = (bool_1, bool_1)
    greedy_best_first_0 = module_0.GreedyBestFirst(list_1, tuple_0, list_0)
    assert (
        f"{type(greedy_best_first_0).__module__}.{type(greedy_best_first_0).__qualname__}"
        == "greedy_best_first.GreedyBestFirst"
    )
    assert greedy_best_first_0.grid == [[False, False, False], [False, False, False]]
    assert (
        f"{type(greedy_best_first_0.start).__module__}.{type(greedy_best_first_0.start).__qualname__}"
        == "greedy_best_first.Node"
    )
    assert (
        f"{type(greedy_best_first_0.target).__module__}.{type(greedy_best_first_0.target).__qualname__}"
        == "greedy_best_first.Node"
    )
    assert (
        f"{type(greedy_best_first_0.open_nodes).__module__}.{type(greedy_best_first_0.open_nodes).__qualname__}"
        == "builtins.list"
    )
    assert len(greedy_best_first_0.open_nodes) == 1
    assert greedy_best_first_0.closed_nodes == []
    assert greedy_best_first_0.reached is False
    assert module_0.delta == ([-1, 0], [0, -1], [1, 0], [0, 1])
    var_0 = greedy_best_first_0.search()
    assert len(greedy_best_first_0.open_nodes) == 3
    assert len(greedy_best_first_0.closed_nodes) == 2
    assert greedy_best_first_0.reached is True


def test_case_9():
    bool_0 = False
    list_0 = [bool_0, bool_0, bool_0, bool_0]
    list_1 = [list_0, list_0]
    bool_1 = True
    tuple_0 = (bool_1, bool_1)
    greedy_best_first_0 = module_0.GreedyBestFirst(list_1, tuple_0, list_0)
    assert (
        f"{type(greedy_best_first_0).__module__}.{type(greedy_best_first_0).__qualname__}"
        == "greedy_best_first.GreedyBestFirst"
    )
    assert greedy_best_first_0.grid == [
        [False, False, False, False],
        [False, False, False, False],
    ]
    assert (
        f"{type(greedy_best_first_0.start).__module__}.{type(greedy_best_first_0.start).__qualname__}"
        == "greedy_best_first.Node"
    )
    assert (
        f"{type(greedy_best_first_0.target).__module__}.{type(greedy_best_first_0.target).__qualname__}"
        == "greedy_best_first.Node"
    )
    assert (
        f"{type(greedy_best_first_0.open_nodes).__module__}.{type(greedy_best_first_0.open_nodes).__qualname__}"
        == "builtins.list"
    )
    assert len(greedy_best_first_0.open_nodes) == 1
    assert greedy_best_first_0.closed_nodes == []
    assert greedy_best_first_0.reached is False
    assert module_0.delta == ([-1, 0], [0, -1], [1, 0], [0, 1])
    var_0 = greedy_best_first_0.search()
    assert len(greedy_best_first_0.open_nodes) == 3
    assert len(greedy_best_first_0.closed_nodes) == 2
    assert greedy_best_first_0.reached is True
    var_1 = greedy_best_first_0.search()
    assert len(greedy_best_first_0.open_nodes) == 2
    assert len(greedy_best_first_0.closed_nodes) == 3
    var_2 = greedy_best_first_0.search()
    assert greedy_best_first_0.open_nodes == []
    assert len(greedy_best_first_0.closed_nodes) == 7

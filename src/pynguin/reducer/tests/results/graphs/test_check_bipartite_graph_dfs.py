# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import pytest
import check_bipartite_graph_dfs as module_0


def test_case_0():
    dict_0 = {}
    var_0 = module_0.check_bipartite_dfs(dict_0)
    assert var_0 is True


@pytest.mark.xfail(strict=True)
def test_case_1():
    str_0 = "[akum)9k"
    module_0.check_bipartite_dfs(str_0)


@pytest.mark.xfail(strict=True)
def test_case_2():
    none_type_0 = None
    module_0.check_bipartite_dfs(none_type_0)


@pytest.mark.xfail(strict=True)
def test_case_3():
    list_0 = []
    list_1 = [list_0]
    var_0 = module_0.check_bipartite_dfs(list_1)
    assert var_0 is True
    none_type_0 = None
    module_0.check_bipartite_dfs(none_type_0)


@pytest.mark.xfail(strict=True)
def test_case_4():
    bool_0 = False
    dict_0 = {bool_0: bool_0}
    list_0 = [dict_0, bool_0, bool_0]
    tuple_0 = (dict_0, list_0, bool_0, dict_0)
    module_0.check_bipartite_dfs(tuple_0)


@pytest.mark.xfail(strict=True)
def test_case_5():
    bool_0 = True
    set_0 = {bool_0, bool_0, bool_0, bool_0}
    tuple_0 = (set_0, set_0)
    var_0 = module_0.check_bipartite_dfs(tuple_0)
    assert var_0 is False
    int_0 = 2542
    module_0.check_bipartite_dfs(int_0)

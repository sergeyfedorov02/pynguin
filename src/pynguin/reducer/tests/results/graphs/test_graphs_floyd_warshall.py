# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import pytest
import graphs_floyd_warshall as module_0


def test_case_0():
    int_0 = -399
    var_0 = module_0.floyd_warshall(int_0, int_0)


@pytest.mark.xfail(strict=True)
def test_case_1():
    bool_0 = True
    module_0.floyd_warshall(bool_0, bool_0)


@pytest.mark.xfail(strict=True)
def test_case_2():
    none_type_0 = None
    module_0.floyd_warshall(none_type_0, none_type_0)


@pytest.mark.xfail(strict=True)
def test_case_3():
    bool_0 = False
    var_0 = module_0.floyd_warshall(bool_0, bool_0)
    list_0 = [var_0, bool_0, bool_0, bool_0]
    bool_1 = True
    module_0.floyd_warshall(list_0, bool_1)

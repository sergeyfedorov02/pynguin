# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import pytest
import dijkstra_2 as module_0

@pytest.mark.xfail(strict=True)
def test_case_0():
    int_0 = 892
    bytes_0 = b'\x10\x0b,'
    module_0.print_dist(bytes_0, int_0)

@pytest.mark.xfail(strict=True)
def test_case_3():
    bool_0 = False
    int_0 = 2020
    module_0.dijkstra(bool_0, int_0, bool_0)

@pytest.mark.xfail(strict=True)
def test_case_4():
    bool_0 = False
    module_0.dijkstra(bool_0, bool_0, bool_0)

@pytest.mark.xfail(strict=True)
def test_case_5():
    bytes_0 = b'\x08\xcf\xa9'
    module_0.min_dist(bytes_0, bytes_0, bytes_0)

@pytest.mark.xfail(strict=True)
def test_case_7():
    bool_0 = False
    var_0 = module_0.min_dist(bool_0, bool_0, bool_0)
    assert var_0 == -1
    var_2 = module_0.print_dist(var_0, bool_0)

@pytest.mark.xfail(strict=True)
def test_case_8():
    str_0 = 'E }.Y4;sk{&RS'
    int_0 = 3547
    module_0.min_dist(str_0, str_0, int_0)
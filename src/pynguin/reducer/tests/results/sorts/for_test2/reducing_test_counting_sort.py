# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import pytest
import counting_sort as module_0
import builtins as module_1

@pytest.mark.xfail(strict=True)
def test_case_0():
    str_0 = 'test'
    list_0 = []
    var_0 = module_0.counting_sort(list_0)
    var_1 = module_0.counting_sort_string(str_0)
    assert var_1 == '\r!!"#+3@BQRU^als{'
    module_0.counting_sort_string(none_type_0)

@pytest.mark.xfail(strict=True)
def test_case_1():
    object_0 = module_1.object()
    tuple_0 = (object_0, object_0)
    module_0.counting_sort(tuple_0)

@pytest.mark.xfail(strict=True)
def test_case_4():
    int_0 = 1
    set_0 = {int_0, int_0}
    module_0.counting_sort(set_0)
# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import pytest
import articulation_points as module_0


def test_case_0():
    dict_0 = {}
    var_0 = module_0.compute_ap(dict_0)


@pytest.mark.xfail(strict=True)
def test_case_1():
    str_0 = "g"
    module_0.compute_ap(str_0)


@pytest.mark.xfail(strict=True)
def test_case_2():
    none_type_0 = None
    module_0.compute_ap(none_type_0)


def test_case_3():
    tuple_0 = ()
    tuple_1 = (tuple_0,)
    var_0 = module_0.compute_ap(tuple_1)


@pytest.mark.xfail(strict=True)
def test_case_4():
    bool_0 = True
    dict_0 = {bool_0: bool_0, bool_0: bool_0, bool_0: bool_0}
    list_0 = [dict_0, dict_0, bool_0]
    module_0.compute_ap(list_0)

# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import pytest
import strongly_connected_components as module_0


def test_case_0():
    list_0 = []
    bool_0 = False
    bool_1 = True
    list_1 = [bool_1, bool_0]
    dict_0 = {bool_1: list_0, bool_0: list_0, bool_0: list_1, bool_0: list_0}
    list_2 = module_0.strongly_connected_components(dict_0)


def test_case_1():
    bool_0 = False
    list_0 = [bool_0]
    bool_1 = False
    bool_2 = True
    dict_0 = {bool_0: list_0, bool_1: list_0, bool_2: list_0, bool_0: list_0}
    list_1 = module_0.strongly_connected_components(dict_0)


@pytest.mark.xfail(strict=True)
def test_case_2():
    list_0 = []
    bool_0 = True
    bool_1 = True
    list_1 = [bool_1, bool_0]
    dict_0 = {bool_1: list_0, bool_0: list_0, bool_0: list_1, bool_0: list_0}
    module_0.strongly_connected_components(dict_0)


@pytest.mark.xfail(strict=True)
def test_case_3():
    list_0 = []
    module_0.strongly_connected_components(list_0)


@pytest.mark.xfail(strict=True)
def test_case_4():
    str_0 = "Ju,Rn\\\\a&K8&(##)2NY"
    tuple_0 = (str_0,)
    list_0 = [tuple_0]
    bool_0 = False
    module_0.find_components(list_0, bool_0, list_0)


@pytest.mark.xfail(strict=True)
def test_case_5():
    bool_0 = False
    list_0 = [bool_0, bool_0]
    list_1 = [bool_0]
    dict_0 = {bool_0: list_0, bool_0: list_0, bool_0: list_1, bool_0: list_0}
    module_0.find_components(dict_0, bool_0, list_0)


@pytest.mark.xfail(strict=True)
def test_case_6():
    bool_0 = False
    int_0 = 961
    int_1 = 410
    int_2 = -185
    list_0 = [int_0, int_1, int_2, int_2]
    dict_0 = {bool_0: list_0}
    module_0.strongly_connected_components(dict_0)


@pytest.mark.xfail(strict=True)
def test_case_7():
    bool_0 = True
    list_0 = [bool_0]
    bool_1 = True
    bool_2 = False
    bool_3 = True
    dict_0 = {bool_0: list_0, bool_2: list_0, bool_3: list_0, bool_0: list_0}
    list_1 = module_0.strongly_connected_components(dict_0)
    dict_1 = {bool_0: list_0, bool_1: list_0}
    module_0.strongly_connected_components(dict_1)


@pytest.mark.xfail(strict=True)
def test_case_8():
    bool_0 = True
    bool_1 = False
    bool_2 = False
    list_0 = [bool_0, bool_1, bool_2]
    bool_3 = True
    dict_0 = {bool_0: list_0, bool_3: list_0, bool_0: list_0, bool_1: list_0}
    list_1 = module_0.strongly_connected_components(dict_0)
    str_0 = ""
    module_0.topology_sort(str_0, str_0, str_0)

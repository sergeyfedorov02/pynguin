# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import pytest
import tarjans_scc as module_0

@pytest.mark.xfail(strict=True)
def test_case_0():
    str_0 = '\x0b~?'
    module_0.tarjan(str_0)

@pytest.mark.xfail(strict=True)
def test_case_2():
    bool_0 = True
    module_0.create_graph(bool_0, bool_0)

def test_case_3():
    dict_0 = {}
    var_0 = module_0.tarjan(dict_0)
    bool_0 = False
    var_2 = module_0.create_graph(bool_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_4():
    bytes_0 = b'0\x9f5\xef\x8ae\xf0m\x1e;6\x1b[R,\x03f\xa7P\x03'
    module_0.create_graph(bytes_0, bytes_0)

def test_case_8():
    bool_0 = True
    set_0 = {bool_0, bool_0, bool_0, bool_0}
    list_0 = [set_0, set_0, set_0, set_0, set_0]
    var_0 = module_0.tarjan(list_0)
    var_1 = module_0.tarjan(var_0)
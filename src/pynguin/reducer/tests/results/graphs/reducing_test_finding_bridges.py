# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import finding_bridges as module_0

def test_case_2():
    bool_0 = False
    bool_1 = True
    list_0 = [bool_0, bool_1, bool_0]
    dict_0 = {bool_0: list_0, bool_1: list_0, bool_1: list_0, bool_0: list_0}
    list_1 = module_0.compute_bridges(dict_0)
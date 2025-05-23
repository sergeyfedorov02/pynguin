# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import pytest
import a_simple_sorts as module_0

def test_case_0():
    list_0 = []
    var_0 = module_0.shell_sort(list_0)
    list_1 = module_0.bead_sort(list_0)
    list_3 = module_0.exchange_sort(list_1)

@pytest.mark.xfail(strict=True)
def test_case_5():
    list_0 = []
    list_2 = module_0.binary_insertion_sort(list_0)
    complex_0 = 2838.604 - 3120.56j
    list_3 = [list_2, complex_0, list_2]
    module_0.selection_sort(list_3)

@pytest.mark.xfail(strict=True)
def test_case_6():
    none_type_0 = None
    module_0.shell_sort(none_type_0)

@pytest.mark.xfail(strict=True)
def test_case_7():
    complex_0 = 2024.13147 - 1758.9747j
    list_0 = [complex_0]
    module_0.bucket_sort(list_0)

def test_case_15():
    bool_0 = False

@pytest.mark.xfail(strict=True)
def test_case_18():
    none_type_0 = None
    module_0.selection_sort(none_type_0)

@pytest.mark.xfail(strict=True)
def test_case_19():
    dict_0 = {}
    none_type_0 = None
    list_0 = [none_type_0, dict_0, none_type_0, dict_0]
    module_0.bubble_sort(list_0)

@pytest.mark.xfail(strict=True)
def test_case_20():
    bool_0 = False
    dict_0 = {bool_0: bool_0, bool_0: bool_0}
    list_0 = module_0.cocktail_shaker_sort(dict_0)
    list_1 = [list_0]
    module_0.bucket_sort(list_1)

@pytest.mark.xfail(strict=True)
def test_case_23():
    bytes_0 = b'\x0cJJ\x87\xa3\xef\x13\x831\xc4\xa0'
    module_0.selection_sort(bytes_0)

@pytest.mark.xfail(strict=True)
def test_case_24():
    str_0 = '\\,b5lDGc4-a'
    module_0.comb_sort(str_0)

@pytest.mark.xfail(strict=True)
def test_case_25():
    complex_0 = 2024.13147 - 1758.9747j
    list_0 = [complex_0]
    module_0.comb_sort(list_0)

@pytest.mark.xfail(strict=True)
def test_case_26():
    str_0 = "R'\t (:s!syT!"
    module_0.double_sort(str_0)

@pytest.mark.xfail(strict=True)
def test_case_27():
    str_0 = '5P6+*oK)A9N'
    module_0.exchange_sort(str_0)

def test_case_28():
    bool_0 = False
    int_0 = 1579
    bool_1 = True
    list_0 = [int_0, bool_1, bool_0, bool_1]
    list_1 = module_0.gnome_sort(list_0)
    list_4 = [bool_0, int_0, bool_1]
    list_5 = module_0.exchange_sort(list_4)

@pytest.mark.xfail(strict=True)
def test_case_29():
    float_0 = -2655.6
    module_0.double_sort(float_0)

@pytest.mark.xfail(strict=True)
def test_case_30():
    int_0 = -1882
    list_0 = [int_0]
    list_1 = module_0.binary_insertion_sort(list_0)
    list_2 = []
    var_0 = module_0.shell_sort(list_2)
    list_3 = module_0.comb_sort(list_2)
    list_4 = module_0.bead_sort(list_2)
    list_6 = [list_4, list_3, list_1, list_3, list_4]
    list_7 = module_0.comb_sort(list_6)
    list_13 = [var_0, list_1, var_0, var_0]

@pytest.mark.xfail(strict=True)
def test_case_32():
    str_0 = '(r&>x'
    module_0.shell_sort(str_0)

def test_case_33():
    list_0 = []
    var_0 = module_0.shell_sort(list_0)
    list_4 = module_0.comb_sort(var_0)
    list_5 = module_0.comb_sort(list_0)
    list_9 = [var_0, var_0, var_0, var_0]
    list_10 = module_0.cocktail_shaker_sort(list_4)
    list_14 = [list_10, list_5, list_9, list_10]

def test_case_37():
    list_0 = []
    var_0 = module_0.shell_sort(list_0)
    list_2 = module_0.comb_sort(list_0)

@pytest.mark.xfail(strict=True)
def test_case_38():
    int_0 = 2595
    list_0 = [int_0, int_0]
    list_1 = module_0.exchange_sort(list_0)
    list_2 = []
    var_0 = module_0.double_sort(list_2)

def test_case_39():
    int_0 = 2595
    list_0 = [int_0, int_0]
    list_1 = module_0.exchange_sort(list_0)
    list_2 = []
    var_0 = module_0.double_sort(list_2)
    var_1 = module_0.shell_sort(list_1)
    list_4 = module_0.cocktail_shaker_sort(var_1)
    list_5 = module_0.comb_sort(list_2)
    list_6 = module_0.gnome_sort(list_5)
    var_2 = module_0.double_sort(list_2)
    list_8 = [var_0, var_0, var_0, var_0]
    var_3 = module_0.shell_sort(list_1)

def test_case_40():
    bytes_0 = b'd\x90\x83,\x9a\xf6$@\ncP\xdf\xd2\xee\xa4'
    with pytest.raises(TypeError):
        module_0.bead_sort(bytes_0)

@pytest.mark.xfail(strict=True)
def test_case_41():
    str_0 = '\tq:'
    module_0.double_sort(str_0)
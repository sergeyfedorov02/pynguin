# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import pytest
import quick_sort_3_partition as module_0


def test_case_0():
    bool_0 = False
    bool_1 = True
    none_type_0 = None
    none_type_1 = module_0.quick_sort_3partition(none_type_0, bool_1, bool_0)


@pytest.mark.xfail(strict=True)
def test_case_1():
    tuple_0 = ()
    list_0 = [tuple_0]
    bool_0 = False
    bool_1 = True
    module_0.quick_sort_3partition(list_0, bool_0, bool_1)


@pytest.mark.xfail(strict=True)
def test_case_2():
    bool_0 = True
    int_0 = 102
    list_0 = [int_0, bool_0, int_0]
    bool_1 = False
    bool_2 = True
    none_type_0 = module_0.quick_sort_lomuto_partition(list_0, bool_1, bool_2)
    none_type_1 = module_0.quick_sort_3partition(bool_0, int_0, int_0)
    str_0 = "2'a,g\"G#z"
    module_0.quick_sort_3partition(str_0, bool_0, int_0)


def test_case_3():
    bool_0 = False
    list_0 = [bool_0, bool_0, bool_0]
    list_1 = [list_0, bool_0, bool_0, bool_0]
    bool_1 = True
    none_type_0 = module_0.quick_sort_lomuto_partition(list_1, bool_1, bool_1)


@pytest.mark.xfail(strict=True)
def test_case_4():
    dict_0 = {}
    list_0 = [dict_0, dict_0, dict_0]
    bool_0 = False
    bool_1 = True
    module_0.lomuto_partition(list_0, bool_0, bool_1)


def test_case_5():
    list_0 = []
    list_1 = module_0.three_way_radix_quicksort(list_0)


@pytest.mark.xfail(strict=True)
def test_case_6():
    bool_0 = False
    list_0 = [bool_0, bool_0, bool_0]
    list_1 = [list_0, bool_0, bool_0, bool_0]
    none_type_0 = module_0.quick_sort_lomuto_partition(list_0, bool_0, bool_0)
    module_0.three_way_radix_quicksort(list_1)


@pytest.mark.xfail(strict=True)
def test_case_7():
    bool_0 = False
    bool_1 = True
    str_0 = "*K =j0T=VA1,W%lfhyZ"
    module_0.quick_sort_3partition(str_0, bool_0, bool_1)


@pytest.mark.xfail(strict=True)
def test_case_8():
    dict_0 = {}
    list_0 = [dict_0, dict_0, dict_0]
    bool_0 = True
    int_0 = module_0.lomuto_partition(list_0, bool_0, bool_0)
    int_1 = -134
    module_0.quick_sort_lomuto_partition(dict_0, dict_0, int_1)


@pytest.mark.xfail(strict=True)
def test_case_9():
    str_0 = "u\r"
    list_0 = module_0.three_way_radix_quicksort(str_0)
    list_1 = [list_0, list_0, list_0]
    bool_0 = False
    bool_1 = True
    none_type_0 = module_0.quick_sort_lomuto_partition(list_1, bool_0, bool_1)
    str_1 = "*K =j0T=VA1,W%lfhyZ"
    module_0.quick_sort_3partition(str_1, bool_0, bool_1)


@pytest.mark.xfail(strict=True)
def test_case_10():
    str_0 = "u\r"
    list_0 = module_0.three_way_radix_quicksort(str_0)
    list_1 = module_0.three_way_radix_quicksort(list_0)
    bool_0 = False
    bool_1 = True
    none_type_0 = module_0.quick_sort_lomuto_partition(list_1, bool_0, bool_1)
    str_1 = "*K =j0T=VA1,W%lfhyZ"
    module_0.quick_sort_3partition(str_1, bool_0, bool_1)


@pytest.mark.xfail(strict=True)
def test_case_11():
    bool_0 = False
    bool_1 = True
    str_0 = "\\*K =j0T=VA1,W%lfhyZ"
    module_0.quick_sort_3partition(str_0, bool_0, bool_1)


@pytest.mark.xfail(strict=True)
def test_case_12():
    bool_0 = True
    int_0 = 102
    list_0 = [bool_0, int_0, bool_0, int_0]
    bool_1 = False
    bool_2 = True
    none_type_0 = module_0.quick_sort_lomuto_partition(list_0, bool_1, bool_2)
    none_type_1 = module_0.quick_sort_3partition(bool_0, int_0, int_0)
    str_0 = "2'a,g\"G#z"
    module_0.quick_sort_3partition(str_0, bool_0, int_0)


def test_case_13():
    str_0 = "u\r"
    list_0 = module_0.three_way_radix_quicksort(str_0)
    bool_0 = True
    bool_1 = False
    none_type_0 = module_0.quick_sort_3partition(list_0, bool_1, bool_0)

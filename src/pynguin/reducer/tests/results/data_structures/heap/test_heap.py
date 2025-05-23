# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import pytest
import heap as module_0


def test_case_0():
    heap_0 = module_0.Heap()
    assert heap_0.h == []
    assert heap_0.heap_size == 0
    str_0 = "Te"
    none_type_0 = heap_0.build_max_heap(str_0)
    assert heap_0.h == ["e", "T"]
    str_1 = heap_0.__repr__()
    assert str_1 == "['e', 'T']"
    int_0 = 3652
    var_0 = heap_0.parent_index(int_0)
    assert var_0 == 1825
    bool_0 = heap_0.__eq__(var_0)
    heap_1 = module_0.Heap()
    assert heap_1.h == []
    assert heap_1.heap_size == 0


@pytest.mark.xfail(strict=True)
def test_case_1():
    bool_0 = True
    list_0 = [bool_0]
    heap_0 = module_0.Heap()
    assert heap_0.h == []
    assert heap_0.heap_size == 0
    bool_1 = False
    var_0 = heap_0.parent_index(bool_1)
    var_1 = heap_0.left_child_idx(bool_0)
    none_type_0 = heap_0.insert(list_0)
    assert heap_0.heap_size == 1
    heap_0.insert(var_1)


@pytest.mark.xfail(strict=True)
def test_case_2():
    int_0 = -15
    heap_0 = module_0.Heap()
    assert heap_0.h == []
    assert heap_0.heap_size == 0
    var_0 = heap_0.right_child_idx(int_0)
    assert var_0 == -28
    int_1 = -474
    heap_1 = module_0.Heap()
    assert heap_1.heap_size == 0
    heap_1.max_heapify(int_1)


def test_case_3():
    int_0 = 2907
    heap_0 = module_0.Heap()
    assert heap_0.h == []
    assert heap_0.heap_size == 0
    none_type_0 = heap_0.max_heapify(int_0)


def test_case_4():
    heap_0 = module_0.Heap()
    assert heap_0.h == []
    assert heap_0.heap_size == 0
    set_0 = {heap_0, heap_0, heap_0, heap_0}
    none_type_0 = heap_0.build_max_heap(set_0)
    heap_1 = module_0.Heap()
    assert heap_1.h == []
    assert heap_1.heap_size == 0
    none_type_1 = heap_1.heap_sort()
    int_0 = 4291
    var_0 = heap_0.extract_max()
    assert heap_0.h == []
    assert heap_0.heap_size == 0
    assert f"{type(var_0).__module__}.{type(var_0).__qualname__}" == "heap.Heap"
    assert var_0.h == []
    assert var_0.heap_size == 0
    bool_0 = False
    none_type_2 = heap_0.max_heapify(bool_0)
    none_type_3 = heap_0.right_child_idx(int_0)
    none_type_4 = heap_1.max_heapify(int_0)
    heap_2 = module_0.Heap()
    assert heap_2.heap_size == 0
    heap_3 = module_0.Heap()
    assert heap_3.heap_size == 0
    none_type_5 = heap_3.max_heapify(int_0)
    var_1 = heap_0.right_child_idx(int_0)
    with pytest.raises(Exception):
        heap_0.extract_max()


def test_case_5():
    heap_0 = module_0.Heap()
    assert heap_0.h == []
    assert heap_0.heap_size == 0
    with pytest.raises(Exception):
        heap_0.extract_max()


def test_case_6():
    bool_0 = False
    heap_0 = module_0.Heap()
    assert heap_0.h == []
    assert heap_0.heap_size == 0
    none_type_0 = heap_0.insert(bool_0)
    assert heap_0.h == [False]
    assert heap_0.heap_size == 1


def test_case_7():
    heap_0 = module_0.Heap()
    assert heap_0.h == []
    assert heap_0.heap_size == 0
    none_type_0 = heap_0.heap_sort()


def test_case_8():
    heap_0 = module_0.Heap()
    assert heap_0.h == []
    assert heap_0.heap_size == 0


def test_case_9():
    heap_0 = module_0.Heap()
    assert heap_0.h == []
    assert heap_0.heap_size == 0
    none_type_0 = heap_0.insert(heap_0)
    assert (
        f"{type(heap_0.h).__module__}.{type(heap_0.h).__qualname__}" == "builtins.list"
    )
    assert len(heap_0.h) == 1
    assert heap_0.heap_size == 1
    var_0 = heap_0.extract_max()
    assert heap_0.h == []
    assert heap_0.heap_size == 0
    assert f"{type(var_0).__module__}.{type(var_0).__qualname__}" == "heap.Heap"
    assert var_0.h == []
    assert var_0.heap_size == 0
    none_type_1 = var_0.insert(heap_0)
    assert heap_0.heap_size == 1
    assert var_0.heap_size == 1
    bool_0 = var_0.__eq__(var_0)
    none_type_2 = var_0.heap_sort()
    var_1 = heap_0.extract_max()
    assert heap_0.h == []
    assert var_0.h == []
    assert f"{type(var_1).__module__}.{type(var_1).__qualname__}" == "heap.Heap"
    assert var_1.h == []
    assert var_1.heap_size == 0
    bool_1 = var_0.__gt__(none_type_1)


def test_case_10():
    heap_0 = module_0.Heap()
    assert heap_0.h == []
    assert heap_0.heap_size == 0
    str_0 = "Te"
    none_type_0 = heap_0.heap_sort()
    none_type_1 = heap_0.build_max_heap(str_0)
    assert heap_0.h == ["e", "T"]
    str_1 = heap_0.__repr__()
    assert str_1 == "['e', 'T']"
    var_0 = heap_0.extract_max()
    assert var_0 == "e"
    assert heap_0.h == ["T"]
    assert heap_0.heap_size == 1
    bool_0 = var_0.__eq__(str_0)
    int_0 = 3652
    var_1 = heap_0.parent_index(int_0)
    assert var_1 == 1825
    none_type_2 = heap_0.heap_sort()
    bool_1 = none_type_0.__gt__(str_0)
    none_type_3 = heap_0.heap_sort()


@pytest.mark.xfail(strict=True)
def test_case_11():
    heap_0 = module_0.Heap()
    assert heap_0.h == []
    assert heap_0.heap_size == 0
    str_0 = "Te\r"
    none_type_0 = heap_0.build_max_heap(str_0)
    assert heap_0.h == ["e", "T", "\r"]
    str_1 = heap_0.__repr__()
    assert str_1 == "['e', 'T', '\\r']"
    int_0 = 3652
    var_0 = heap_0.parent_index(int_0)
    assert var_0 == 1825
    str_2 = heap_0.__repr__()
    assert str_2 == "['e', 'T', '\\r']"
    var_1 = heap_0.extract_max()
    assert var_1 == "e"
    assert heap_0.h == ["T", "\r"]
    assert heap_0.heap_size == 2
    none_type_1 = heap_0.heap_sort()
    assert heap_0.h == ["\r", "T"]
    heap_0.max_heapify(none_type_1)


@pytest.mark.xfail(strict=True)
def test_case_12():
    heap_0 = module_0.Heap()
    assert heap_0.h == []
    assert heap_0.heap_size == 0
    str_0 = "Te"
    none_type_0 = heap_0.build_max_heap(str_0)
    assert heap_0.h == ["e", "T"]
    str_1 = heap_0.__repr__()
    assert str_1 == "['e', 'T']"
    none_type_1 = None
    heap_0.insert(none_type_1)


@pytest.mark.xfail(strict=True)
def test_case_13():
    heap_0 = module_0.Heap()
    assert heap_0.h == []
    assert heap_0.heap_size == 0
    str_0 = "Te\r"
    none_type_0 = heap_0.build_max_heap(str_0)
    assert heap_0.h == ["e", "T", "\r"]
    str_1 = heap_0.__repr__()
    assert str_1 == "['e', 'T', '\\r']"
    var_0 = heap_0.extract_max()
    assert var_0 == "e"
    assert heap_0.h == ["T", "\r"]
    assert heap_0.heap_size == 2
    none_type_1 = heap_0.insert(str_0)
    assert heap_0.h == ["Te\r", "\r", "T"]
    assert heap_0.heap_size == 3
    bool_0 = True
    var_0.max_heapify(bool_0)


@pytest.mark.xfail(strict=True)
def test_case_14():
    heap_0 = module_0.Heap()
    assert heap_0.h == []
    assert heap_0.heap_size == 0
    str_0 = "Te\r"
    none_type_0 = heap_0.build_max_heap(str_0)
    assert heap_0.h == ["e", "T", "\r"]
    str_1 = heap_0.__repr__()
    assert str_1 == "['e', 'T', '\\r']"
    none_type_1 = heap_0.insert(str_0)
    assert heap_0.h == ["e", "Te\r", "\r", "T"]
    assert heap_0.heap_size == 4
    heap_1 = module_0.Heap()
    assert heap_1.h == []
    assert heap_1.heap_size == 0
    var_0 = heap_0.extract_max()
    assert var_0 == "e"
    assert heap_0.h == ["Te\r", "T", "\r"]
    assert heap_0.heap_size == 3
    heap_0.max_heapify(none_type_1)

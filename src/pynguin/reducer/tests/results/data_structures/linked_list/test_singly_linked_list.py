# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import pytest
import singly_linked_list as module_0


def test_case_0():
    linked_list_0 = module_0.LinkedList()
    assert len(linked_list_0) == 0
    none_type_0 = linked_list_0.insert_head(linked_list_0)
    assert len(linked_list_0) == 1
    var_0 = linked_list_0.delete_tail()
    assert len(linked_list_0) == 0
    assert (
        f"{type(var_0).__module__}.{type(var_0).__qualname__}"
        == "singly_linked_list.LinkedList"
    )
    assert len(var_0) == 0
    none_type_1 = var_0.print_list()


def test_case_1():
    bool_0 = False
    linked_list_0 = module_0.LinkedList()
    assert len(linked_list_0) == 0
    with pytest.raises(IndexError):
        linked_list_0.delete_nth(bool_0)


@pytest.mark.xfail(strict=True)
def test_case_2():
    linked_list_0 = module_0.LinkedList()
    assert len(linked_list_0) == 0
    none_type_0 = linked_list_0.insert_tail(linked_list_0)
    assert len(linked_list_0) == 1
    none_type_1 = linked_list_0.reverse()
    none_type_2 = linked_list_0.insert_tail(linked_list_0)
    assert len(linked_list_0) == 2
    none_type_3 = linked_list_0.reverse()
    var_0 = linked_list_0.delete_tail()
    assert len(linked_list_0) == 1
    assert (
        f"{type(var_0).__module__}.{type(var_0).__qualname__}"
        == "singly_linked_list.LinkedList"
    )
    assert len(var_0) == 1
    var_0.print_list()


def test_case_3():
    linked_list_0 = module_0.LinkedList()
    assert len(linked_list_0) == 0
    int_0 = linked_list_0.__len__()
    assert int_0 == 0
    bool_0 = True
    with pytest.raises(ValueError):
        linked_list_0.__getitem__(bool_0)


def test_case_4():
    int_0 = -395
    linked_list_0 = module_0.LinkedList()
    assert len(linked_list_0) == 0
    with pytest.raises(ValueError):
        linked_list_0.__getitem__(int_0)


def test_case_5():
    bool_0 = False
    none_type_0 = None
    node_0 = module_0.Node(none_type_0)
    linked_list_0 = module_0.LinkedList()
    assert len(linked_list_0) == 0
    with pytest.raises(ValueError):
        linked_list_0.__setitem__(bool_0, none_type_0)


@pytest.mark.xfail(strict=True)
def test_case_6():
    linked_list_0 = module_0.LinkedList()
    assert len(linked_list_0) == 0
    none_type_0 = linked_list_0.insert_head(linked_list_0)
    assert len(linked_list_0) == 1
    none_type_0.print_list()


def test_case_7():
    linked_list_0 = module_0.LinkedList()
    assert len(linked_list_0) == 0
    linked_list_1 = module_0.LinkedList()
    assert len(linked_list_1) == 0
    int_0 = -84
    none_type_0 = None
    with pytest.raises(IndexError):
        linked_list_1.insert_nth(int_0, none_type_0)


@pytest.mark.xfail(strict=True)
def test_case_8():
    linked_list_0 = module_0.LinkedList()
    assert len(linked_list_0) == 0
    int_0 = linked_list_0.__len__()
    assert int_0 == 0
    linked_list_0.delete_tail()


def test_case_9():
    linked_list_0 = module_0.LinkedList()
    assert len(linked_list_0) == 0
    none_type_0 = linked_list_0.insert_tail(linked_list_0)
    assert len(linked_list_0) == 1
    none_type_1 = linked_list_0.reverse()
    bool_0 = False
    var_0 = linked_list_0.__getitem__(bool_0)
    assert (
        f"{type(var_0).__module__}.{type(var_0).__qualname__}"
        == "singly_linked_list.LinkedList"
    )
    assert len(var_0) == 1
    var_1 = var_0.__iter__()


@pytest.mark.xfail(strict=True)
def test_case_10():
    linked_list_0 = module_0.LinkedList()
    assert len(linked_list_0) == 0
    none_type_0 = linked_list_0.reverse()
    none_type_1 = linked_list_0.insert_tail(linked_list_0)
    assert len(linked_list_0) == 1
    none_type_2 = linked_list_0.reverse()
    linked_list_0.__getitem__(none_type_0)


def test_case_11():
    bool_0 = False
    none_type_0 = None
    node_0 = module_0.Node(none_type_0)
    str_0 = node_0.__repr__()
    assert str_0 == "Node(None)"
    linked_list_0 = module_0.LinkedList()
    assert len(linked_list_0) == 0
    with pytest.raises(ValueError):
        linked_list_0.__setitem__(bool_0, none_type_0)


def test_case_12():
    linked_list_0 = module_0.LinkedList()
    assert len(linked_list_0) == 0
    int_0 = linked_list_0.__len__()
    assert int_0 == 0
    none_type_0 = linked_list_0.insert_tail(int_0)
    assert len(linked_list_0) == 1
    bool_0 = True
    with pytest.raises(ValueError):
        linked_list_0.__getitem__(bool_0)


@pytest.mark.xfail(strict=True)
def test_case_13():
    linked_list_0 = module_0.LinkedList()
    assert len(linked_list_0) == 0
    linked_list_0.delete_head()


@pytest.mark.xfail(strict=True)
def test_case_14():
    linked_list_0 = module_0.LinkedList()
    assert len(linked_list_0) == 0
    bool_0 = linked_list_0.is_empty()
    none_type_0 = linked_list_0.print_list()
    var_0 = linked_list_0.__iter__()
    linked_list_1 = module_0.LinkedList()
    linked_list_1.delete_tail()


@pytest.mark.xfail(strict=True)
def test_case_15():
    bool_0 = True
    linked_list_0 = module_0.LinkedList()
    assert len(linked_list_0) == 0
    none_type_0 = linked_list_0.insert_head(bool_0)
    assert len(linked_list_0) == 1
    none_type_1 = linked_list_0.insert_head(none_type_0)
    assert len(linked_list_0) == 2
    var_0 = linked_list_0.delete_tail()
    assert var_0 is True
    assert len(linked_list_0) == 1
    var_0.print_list()


def test_case_16():
    linked_list_0 = module_0.LinkedList()
    assert len(linked_list_0) == 0
    int_0 = 1412
    linked_list_1 = module_0.LinkedList()
    assert len(linked_list_1) == 0
    int_1 = linked_list_1.__len__()
    assert int_1 == 0
    int_2 = linked_list_1.__len__()
    with pytest.raises(IndexError):
        linked_list_1.insert_nth(int_0, int_0)


@pytest.mark.xfail(strict=True)
def test_case_17():
    linked_list_0 = module_0.LinkedList()
    assert len(linked_list_0) == 0
    none_type_0 = linked_list_0.insert_tail(linked_list_0)
    assert len(linked_list_0) == 1
    none_type_1 = linked_list_0.reverse()
    none_type_2 = linked_list_0.insert_tail(linked_list_0)
    assert len(linked_list_0) == 2
    none_type_3 = linked_list_0.reverse()
    var_0 = linked_list_0.delete_tail()
    assert len(linked_list_0) == 1
    assert (
        f"{type(var_0).__module__}.{type(var_0).__qualname__}"
        == "singly_linked_list.LinkedList"
    )
    assert len(var_0) == 1
    var_1 = linked_list_0.delete_head()
    assert len(linked_list_0) == 0
    assert len(var_0) == 0
    assert (
        f"{type(var_1).__module__}.{type(var_1).__qualname__}"
        == "singly_linked_list.LinkedList"
    )
    assert len(var_1) == 0
    str_0 = var_1.__repr__()
    linked_list_0.delete_head()


@pytest.mark.xfail(strict=True)
def test_case_18():
    linked_list_0 = module_0.LinkedList()
    assert len(linked_list_0) == 0
    int_0 = linked_list_0.__len__()
    assert int_0 == 0
    none_type_0 = linked_list_0.insert_head(int_0)
    assert len(linked_list_0) == 1
    none_type_1 = linked_list_0.insert_tail(int_0)
    assert len(linked_list_0) == 2
    bool_0 = True
    var_0 = linked_list_0.__getitem__(bool_0)
    assert var_0 == 0
    var_0.reverse()


@pytest.mark.xfail(strict=True)
def test_case_19():
    linked_list_0 = module_0.LinkedList()
    assert len(linked_list_0) == 0
    none_type_0 = linked_list_0.insert_tail(linked_list_0)
    assert len(linked_list_0) == 1
    none_type_1 = linked_list_0.reverse()
    none_type_2 = linked_list_0.insert_tail(linked_list_0)
    assert len(linked_list_0) == 2
    none_type_3 = linked_list_0.insert_tail(none_type_2)
    assert len(linked_list_0) == 3
    none_type_4 = linked_list_0.reverse()
    var_0 = linked_list_0.delete_tail()
    assert len(linked_list_0) == 2
    assert (
        f"{type(var_0).__module__}.{type(var_0).__qualname__}"
        == "singly_linked_list.LinkedList"
    )
    assert len(var_0) == 2
    var_1 = var_0.delete_head()
    assert len(linked_list_0) == 1
    assert len(var_0) == 1
    str_0 = var_1.__repr__()
    var_1.delete_head()


@pytest.mark.xfail(strict=True)
def test_case_20():
    bool_0 = True
    linked_list_0 = module_0.LinkedList()
    assert len(linked_list_0) == 0
    none_type_0 = linked_list_0.insert_head(bool_0)
    assert len(linked_list_0) == 1
    none_type_1 = linked_list_0.insert_head(none_type_0)
    assert len(linked_list_0) == 2
    var_0 = linked_list_0.__getitem__(bool_0)
    assert var_0 is True
    var_0.print_list()


def test_case_21():
    linked_list_0 = module_0.LinkedList()
    assert len(linked_list_0) == 0
    none_type_0 = linked_list_0.insert_tail(linked_list_0)
    assert len(linked_list_0) == 1
    int_0 = linked_list_0.__len__()
    assert int_0 == 1
    none_type_1 = linked_list_0.insert_tail(linked_list_0)
    assert len(linked_list_0) == 2
    none_type_2 = linked_list_0.reverse()
    var_0 = linked_list_0.delete_tail()
    assert len(linked_list_0) == 1
    assert (
        f"{type(var_0).__module__}.{type(var_0).__qualname__}"
        == "singly_linked_list.LinkedList"
    )
    assert len(var_0) == 1
    none_type_3 = linked_list_0.insert_tail(none_type_1)
    assert len(linked_list_0) == 2
    assert len(var_0) == 2
    bool_0 = False
    var_1 = linked_list_0.__getitem__(bool_0)
    assert (
        f"{type(var_1).__module__}.{type(var_1).__qualname__}"
        == "singly_linked_list.LinkedList"
    )
    assert len(var_1) == 2
    var_2 = linked_list_0.delete_tail()
    assert len(linked_list_0) == 1
    assert len(var_0) == 1
    assert len(var_1) == 1
    var_3 = var_1.delete_head()
    assert len(linked_list_0) == 0
    assert len(var_0) == 0
    assert len(var_1) == 0
    assert (
        f"{type(var_3).__module__}.{type(var_3).__qualname__}"
        == "singly_linked_list.LinkedList"
    )
    assert len(var_3) == 0
    str_0 = var_2.__repr__()
    assert str_0 == "None"
    int_1 = -624
    with pytest.raises(ValueError):
        var_3.__setitem__(int_1, none_type_3)


@pytest.mark.xfail(strict=True)
def test_case_22():
    linked_list_0 = module_0.LinkedList()
    assert len(linked_list_0) == 0
    none_type_0 = linked_list_0.insert_tail(linked_list_0)
    assert len(linked_list_0) == 1
    int_0 = linked_list_0.__len__()
    assert int_0 == 1
    none_type_1 = linked_list_0.insert_tail(linked_list_0)
    assert len(linked_list_0) == 2
    none_type_2 = linked_list_0.reverse()
    var_0 = linked_list_0.delete_tail()
    assert len(linked_list_0) == 1
    assert (
        f"{type(var_0).__module__}.{type(var_0).__qualname__}"
        == "singly_linked_list.LinkedList"
    )
    assert len(var_0) == 1
    none_type_3 = linked_list_0.insert_tail(none_type_1)
    assert len(linked_list_0) == 2
    assert len(var_0) == 2
    bool_0 = False
    var_1 = linked_list_0.__getitem__(bool_0)
    assert (
        f"{type(var_1).__module__}.{type(var_1).__qualname__}"
        == "singly_linked_list.LinkedList"
    )
    assert len(var_1) == 2
    none_type_4 = var_1.reverse()
    var_2 = var_0.__iter__()
    str_0 = var_2.__repr__()
    none_type_5 = var_0.__setitem__(bool_0, none_type_4)
    var_2.delete_tail()


@pytest.mark.xfail(strict=True)
def test_case_23():
    linked_list_0 = module_0.LinkedList()
    assert len(linked_list_0) == 0
    none_type_0 = linked_list_0.insert_tail(linked_list_0)
    assert len(linked_list_0) == 1
    int_0 = linked_list_0.__len__()
    assert int_0 == 1
    none_type_1 = linked_list_0.insert_tail(linked_list_0)
    assert len(linked_list_0) == 2
    none_type_2 = linked_list_0.reverse()
    var_0 = linked_list_0.delete_tail()
    assert len(linked_list_0) == 1
    assert (
        f"{type(var_0).__module__}.{type(var_0).__qualname__}"
        == "singly_linked_list.LinkedList"
    )
    assert len(var_0) == 1
    none_type_3 = linked_list_0.insert_tail(none_type_1)
    assert len(linked_list_0) == 2
    assert len(var_0) == 2
    bool_0 = False
    var_1 = linked_list_0.__getitem__(bool_0)
    assert (
        f"{type(var_1).__module__}.{type(var_1).__qualname__}"
        == "singly_linked_list.LinkedList"
    )
    assert len(var_1) == 2
    none_type_4 = var_1.reverse()
    var_2 = var_0.__iter__()
    str_0 = var_2.__repr__()
    none_type_5 = var_0.__setitem__(int_0, none_type_1)
    var_2.delete_tail()

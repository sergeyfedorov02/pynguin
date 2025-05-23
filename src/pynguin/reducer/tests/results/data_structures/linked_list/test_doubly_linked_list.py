# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import pytest
import doubly_linked_list as module_0


def test_case_0():
    doubly_linked_list_0 = module_0.DoublyLinkedList()
    assert len(doubly_linked_list_0) == 0
    var_0 = doubly_linked_list_0.insert_at_tail(doubly_linked_list_0)
    assert len(doubly_linked_list_0) == 1
    var_1 = doubly_linked_list_0.insert_at_tail(var_0)
    assert len(doubly_linked_list_0) == 2
    var_2 = doubly_linked_list_0.insert_at_head(var_0)
    assert len(doubly_linked_list_0) == 3


@pytest.mark.xfail(strict=True)
def test_case_1():
    doubly_linked_list_0 = module_0.DoublyLinkedList()
    assert len(doubly_linked_list_0) == 0
    doubly_linked_list_0.delete_head()


@pytest.mark.xfail(strict=True)
def test_case_2():
    float_0 = -1794.74707
    doubly_linked_list_0 = module_0.DoublyLinkedList()
    assert len(doubly_linked_list_0) == 0
    var_0 = doubly_linked_list_0.__str__()
    assert var_0 == ""
    var_0.insert_at_nth(float_0, float_0)


@pytest.mark.xfail(strict=True)
def test_case_3():
    doubly_linked_list_0 = module_0.DoublyLinkedList()
    assert len(doubly_linked_list_0) == 0
    doubly_linked_list_0.delete_tail()


def test_case_4():
    none_type_0 = None
    node_0 = module_0.Node(none_type_0)


@pytest.mark.xfail(strict=True)
def test_case_5():
    bool_0 = True
    node_0 = module_0.Node(bool_0)
    var_0 = node_0.__str__()
    doubly_linked_list_0 = module_0.DoublyLinkedList()
    assert len(doubly_linked_list_0) == 0
    none_type_0 = None
    var_1 = doubly_linked_list_0.insert_at_tail(none_type_0)
    assert len(doubly_linked_list_0) == 1
    var_2 = doubly_linked_list_0.delete_tail()
    assert len(doubly_linked_list_0) == 0
    var_2.insert_at_tail(none_type_0)


def test_case_6():
    doubly_linked_list_0 = module_0.DoublyLinkedList()
    assert len(doubly_linked_list_0) == 0


def test_case_7():
    doubly_linked_list_0 = module_0.DoublyLinkedList()
    assert len(doubly_linked_list_0) == 0
    var_0 = doubly_linked_list_0.insert_at_tail(doubly_linked_list_0)
    assert len(doubly_linked_list_0) == 1
    var_1 = doubly_linked_list_0.insert_at_tail(var_0)
    assert len(doubly_linked_list_0) == 2
    var_2 = doubly_linked_list_0.delete_tail()
    assert len(doubly_linked_list_0) == 1


def test_case_8():
    doubly_linked_list_0 = module_0.DoublyLinkedList()
    assert len(doubly_linked_list_0) == 0
    var_0 = doubly_linked_list_0.insert_at_tail(doubly_linked_list_0)
    assert len(doubly_linked_list_0) == 1
    var_1 = doubly_linked_list_0.insert_at_tail(var_0)
    assert len(doubly_linked_list_0) == 2
    str_0 = doubly_linked_list_0.delete(var_1)
    assert len(doubly_linked_list_0) == 1
    var_2 = doubly_linked_list_0.insert_at_head(var_0)
    assert len(doubly_linked_list_0) == 2
    var_3 = doubly_linked_list_0.is_empty()
    assert var_3 is False


def test_case_9():
    float_0 = 629.294004
    doubly_linked_list_0 = module_0.DoublyLinkedList()
    assert len(doubly_linked_list_0) == 0
    doubly_linked_list_1 = module_0.DoublyLinkedList()
    assert len(doubly_linked_list_1) == 0
    var_0 = doubly_linked_list_0.__len__()
    assert var_0 == 0
    var_1 = doubly_linked_list_0.insert_at_head(float_0)
    assert len(doubly_linked_list_0) == 1
    float_1 = 5758.0
    with pytest.raises(IndexError):
        doubly_linked_list_0.insert_at_nth(float_1, var_1)


def test_case_10():
    doubly_linked_list_0 = module_0.DoublyLinkedList()
    assert len(doubly_linked_list_0) == 0
    none_type_0 = None
    var_0 = doubly_linked_list_0.insert_at_tail(none_type_0)
    assert len(doubly_linked_list_0) == 1
    str_0 = doubly_linked_list_0.delete(none_type_0)
    assert len(doubly_linked_list_0) == 0
    bool_0 = True
    var_1 = doubly_linked_list_0.insert_at_tail(bool_0)


@pytest.mark.xfail(strict=True)
def test_case_11():
    doubly_linked_list_0 = module_0.DoublyLinkedList()
    assert len(doubly_linked_list_0) == 0
    var_0 = doubly_linked_list_0.insert_at_head(doubly_linked_list_0)
    assert len(doubly_linked_list_0) == 1
    doubly_linked_list_0.__str__()


def test_case_12():
    int_0 = -3556
    doubly_linked_list_0 = module_0.DoublyLinkedList()
    assert len(doubly_linked_list_0) == 0
    with pytest.raises(IndexError):
        doubly_linked_list_0.insert_at_nth(int_0, int_0)


def test_case_13():
    doubly_linked_list_0 = module_0.DoublyLinkedList()
    assert len(doubly_linked_list_0) == 0
    var_0 = doubly_linked_list_0.insert_at_tail(doubly_linked_list_0)
    assert len(doubly_linked_list_0) == 1
    var_1 = doubly_linked_list_0.insert_at_tail(var_0)
    assert len(doubly_linked_list_0) == 2
    str_0 = doubly_linked_list_0.delete(var_1)
    assert len(doubly_linked_list_0) == 1


def test_case_14():
    bool_0 = True
    doubly_linked_list_0 = module_0.DoublyLinkedList()
    assert len(doubly_linked_list_0) == 0
    var_0 = doubly_linked_list_0.insert_at_tail(doubly_linked_list_0)
    assert len(doubly_linked_list_0) == 1
    with pytest.raises(ValueError):
        doubly_linked_list_0.delete(bool_0)


def test_case_15():
    doubly_linked_list_0 = module_0.DoublyLinkedList()
    assert len(doubly_linked_list_0) == 0
    none_type_0 = None
    var_0 = doubly_linked_list_0.insert_at_tail(none_type_0)
    assert len(doubly_linked_list_0) == 1
    str_0 = doubly_linked_list_0.delete(none_type_0)
    assert len(doubly_linked_list_0) == 0
    bool_0 = True
    var_1 = doubly_linked_list_0.insert_at_tail(bool_0)
    var_2 = doubly_linked_list_0.insert_at_tail(none_type_0)
    assert len(doubly_linked_list_0) == 2
    var_3 = doubly_linked_list_0.insert_at_head(bool_0)
    assert len(doubly_linked_list_0) == 3
    var_4 = doubly_linked_list_0.delete_head()
    assert var_4 is True
    assert len(doubly_linked_list_0) == 2


def test_case_16():
    bool_0 = True
    doubly_linked_list_0 = module_0.DoublyLinkedList()
    assert len(doubly_linked_list_0) == 0
    var_0 = doubly_linked_list_0.insert_at_head(doubly_linked_list_0)
    assert len(doubly_linked_list_0) == 1
    var_1 = doubly_linked_list_0.insert_at_tail(doubly_linked_list_0)
    assert len(doubly_linked_list_0) == 2
    with pytest.raises(ValueError):
        doubly_linked_list_0.delete(bool_0)


def test_case_17():
    doubly_linked_list_0 = module_0.DoublyLinkedList()
    assert len(doubly_linked_list_0) == 0
    var_0 = doubly_linked_list_0.insert_at_tail(doubly_linked_list_0)
    assert len(doubly_linked_list_0) == 1
    var_1 = doubly_linked_list_0.insert_at_tail(var_0)
    assert len(doubly_linked_list_0) == 2
    var_2 = doubly_linked_list_0.insert_at_tail(var_0)
    assert len(doubly_linked_list_0) == 3
    str_0 = doubly_linked_list_0.delete(var_2)
    assert len(doubly_linked_list_0) == 2
    var_3 = doubly_linked_list_0.insert_at_head(var_0)
    assert len(doubly_linked_list_0) == 3
    var_4 = doubly_linked_list_0.is_empty()
    assert var_4 is False


@pytest.mark.xfail(strict=True)
def test_case_18():
    doubly_linked_list_0 = module_0.DoublyLinkedList()
    assert len(doubly_linked_list_0) == 0
    none_type_0 = None
    var_0 = doubly_linked_list_0.insert_at_tail(doubly_linked_list_0)
    assert len(doubly_linked_list_0) == 1
    var_1 = doubly_linked_list_0.insert_at_tail(none_type_0)
    assert len(doubly_linked_list_0) == 2
    str_0 = doubly_linked_list_0.delete(none_type_0)
    assert len(doubly_linked_list_0) == 1
    bool_0 = True
    var_2 = doubly_linked_list_0.insert_at_tail(bool_0)
    assert len(doubly_linked_list_0) == 2
    var_3 = doubly_linked_list_0.insert_at_tail(none_type_0)
    assert len(doubly_linked_list_0) == 3
    var_4 = doubly_linked_list_0.insert_at_nth(bool_0, var_1)
    assert len(doubly_linked_list_0) == 4
    var_4.insert_at_head(var_2)


@pytest.mark.xfail(strict=True)
def test_case_19():
    doubly_linked_list_0 = module_0.DoublyLinkedList()
    assert len(doubly_linked_list_0) == 0
    var_0 = doubly_linked_list_0.insert_at_tail(doubly_linked_list_0)
    assert len(doubly_linked_list_0) == 1
    var_1 = doubly_linked_list_0.insert_at_tail(var_0)
    assert len(doubly_linked_list_0) == 2
    str_0 = doubly_linked_list_0.delete(var_1)
    assert len(doubly_linked_list_0) == 1
    bool_0 = True
    var_2 = doubly_linked_list_0.insert_at_tail(bool_0)
    assert len(doubly_linked_list_0) == 2
    var_3 = doubly_linked_list_0.insert_at_tail(var_2)
    assert len(doubly_linked_list_0) == 3
    bool_1 = True
    var_4 = doubly_linked_list_0.delete_at_nth(bool_1)
    assert var_4 is True
    assert len(doubly_linked_list_0) == 2
    var_4.insert_at_head(var_0)

# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import pytest
import linked_queue as module_0

def test_case_0():
    linked_queue_0 = module_0.LinkedQueue()
    assert len(linked_queue_0) == 0
    none_type_0 = linked_queue_0.put(linked_queue_0)
    assert len(linked_queue_0) == 1
    var_0 = linked_queue_0.get()
    assert len(linked_queue_0) == 0
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'linked_queue.LinkedQueue'
    assert len(var_0) == 0

def test_case_2():
    linked_queue_0 = module_0.LinkedQueue()
    assert len(linked_queue_0) == 0
    str_0 = linked_queue_0.__str__()
    assert str_0 == ''

def test_case_4():
    linked_queue_0 = module_0.LinkedQueue()
    assert len(linked_queue_0) == 0
    with pytest.raises(IndexError):
        linked_queue_0.get()

def test_case_7():
    linked_queue_0 = module_0.LinkedQueue()
    assert len(linked_queue_0) == 0
    none_type_0 = linked_queue_0.put(linked_queue_0)
    assert len(linked_queue_0) == 1
    none_type_1 = linked_queue_0.put(linked_queue_0)
    assert len(linked_queue_0) == 2
    str_0 = linked_queue_0.get()
    assert len(linked_queue_0) == 1
    assert f'{type(str_0).__module__}.{type(str_0).__qualname__}' == 'linked_queue.LinkedQueue'
    assert len(str_0) == 1
# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import pytest
import treap as module_0

@pytest.mark.xfail(strict=True)
def test_case_1():
    str_3 = var_0.__str__()

@pytest.mark.xfail(strict=True)
def test_case_2():
    module_0.interact_treap(none_type_0, none_type_0)

@pytest.mark.xfail(strict=True)
def test_case_3():
    bool_0 = True
    list_0 = [bool_0, bool_0]
    module_0.insert(list_0, bool_0)

@pytest.mark.xfail(strict=True)
def test_case_5():
    var_0 = module_0.insert(none_type_0, bool_0)
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'treap.Node'
    assert var_0.value is False
    assert var_0.left is None
    assert var_0.right is None
    int_0 = 3704
    module_0.erase(int_0, int_0)

@pytest.mark.xfail(strict=True)
def test_case_6():
    none_type_0 = None
    set_0 = {none_type_0}
    var_0 = module_0.insert(none_type_0, set_0)
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'treap.Node'
    assert var_0.value == {None}
    assert var_0.left is None
    assert var_0.right is None
    str_0 = var_0.__repr__()
    str_1 = var_0.__str__()
    assert str_1 == '{None} '
    module_0.inorder(set_0)

def test_case_7():
    none_type_0 = None
    none_type_1 = module_0.inorder(none_type_0)
    str_0 = '[iID@`[Og/{aD*M?Gg'

@pytest.mark.xfail(strict=True)
def test_case_9():
    none_type_0 = None
    var_0 = module_0.insert(none_type_0, none_type_0)
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'treap.Node'
    assert var_0.value is None
    assert var_0.left is None
    assert var_0.right is None
    var_1 = module_0.insert(var_0, none_type_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'treap.Node'
    assert var_1.value is None
    assert var_1.left is None
    assert var_1.right is None
    module_0.erase(var_1, none_type_0)

@pytest.mark.xfail(strict=True)
def test_case_10():
    none_type_0 = None
    none_type_1 = module_0.inorder(none_type_0)
    none_type_2 = module_0.inorder(none_type_0)
    var_0 = module_0.merge(none_type_0, none_type_2)
    str_0 = var_0.__repr__()
    str_1 = var_0.__repr__()
    var_1 = module_0.insert(var_0, str_1)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'treap.Node'
    assert var_1.value == 'None'
    assert var_1.left is None
    assert var_1.right is None
    module_0.split(var_1, none_type_1)

@pytest.mark.xfail(strict=True)
def test_case_11():
    int_0 = 5715
    module_0.merge(int_0, int_0)

@pytest.mark.xfail(strict=True)
def test_case_12():
    assert var_2.right is None

@pytest.mark.xfail(strict=True)
def test_case_13():
    none_type_0 = None
    bool_0 = False
    var_0 = module_0.insert(none_type_0, bool_0)
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'treap.Node'
    assert var_0.value is False
    assert var_0.left is None
    assert var_0.right is None
    var_1 = module_0.insert(var_0, bool_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'treap.Node'
    assert var_1.value is False
    none_type_1 = None
    int_0 = 1142
    var_2 = module_0.insert(none_type_1, int_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'treap.Node'
    assert var_2.value == 1142
    assert var_2.left is None
    assert var_2.right is None
    none_type_2 = None
    var_4 = module_0.Node()
    str_0 = " [\\9^>(-D;6H'\x0c1YW$Y,"
    var_5 = module_0.interact_treap(none_type_2, str_0)
    str_1 = var_0.__repr__()
    var_6 = module_0.insert(var_2, bool_0)
    assert f'{type(var_2.left).__module__}.{type(var_2.left).__qualname__}' == 'treap.Node'
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'treap.Node'
    assert var_6.value == 1142
    assert f'{type(var_6.left).__module__}.{type(var_6.left).__qualname__}' == 'treap.Node'
    assert var_6.right is None
    str_3 = var_0.__str__()
    str_4 = var_1.__str__()
    assert str_4 == 'False False '
    str_5 = var_0.__str__()
    module_0.interact_treap(var_5, none_type_0)

@pytest.mark.xfail(strict=True)
def test_case_14():
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'treap.Node'
    assert var_0.value is False
    assert var_0.left is None
    assert var_0.right is None
    assert var_1.value is False
    assert f'{type(var_1.left).__module__}.{type(var_1.left).__qualname__}' == 'treap.Node'
    assert var_1.right is None
    bool_1 = False
    none_type_1 = None
    int_0 = 1142
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'treap.Node'
    assert var_2.value == 1142
    assert var_2.left is None
    assert var_2.right is None

@pytest.mark.xfail(strict=True)
def test_case_15():
    none_type_0 = None
    bool_0 = False
    var_0 = module_0.insert(none_type_0, bool_0)
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'treap.Node'
    assert var_0.value is False
    assert var_0.left is None
    assert var_0.right is None
    var_1 = module_0.insert(var_0, bool_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'treap.Node'
    assert var_1.value is False
    assert f'{type(var_1.left).__module__}.{type(var_1.left).__qualname__}' == 'treap.Node'
    assert var_1.right is None
    int_0 = 1142

@pytest.mark.xfail(strict=True)
def test_case_16():
    none_type_0 = None
    bool_0 = False
    var_0 = module_0.insert(none_type_0, bool_0)
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'treap.Node'
    assert var_0.value is False
    assert var_0.left is None
    assert var_0.right is None
    var_1 = module_0.insert(var_0, bool_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'treap.Node'
    assert var_1.value is False
    none_type_1 = None
    bool_1 = False
    var_4 = module_0.insert(var_0, bool_1)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'treap.Node'
    assert var_4.value is False
    assert f'{type(var_4.left).__module__}.{type(var_4.left).__qualname__}' == 'treap.Node'
    assert var_4.right is None
    str_2 = var_1.__repr__()
    str_3 = '-'
    module_0.interact_treap(none_type_1, str_3)
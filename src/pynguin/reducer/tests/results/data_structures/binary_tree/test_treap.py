# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import pytest
import treap as module_0


def test_case_0():
    none_type_0 = None
    int_0 = 975
    var_0 = module_0.insert(none_type_0, int_0)
    assert f"{type(var_0).__module__}.{type(var_0).__qualname__}" == "treap.Node"
    assert var_0.value == 975
    assert var_0.left is None
    assert var_0.right is None
    str_0 = var_0.__repr__()
    str_1 = var_0.__str__()
    assert str_1 == "975 "
    none_type_1 = module_0.inorder(none_type_0)
    str_2 = var_0.__repr__()
    str_3 = var_0.__repr__()


@pytest.mark.xfail(strict=True)
def test_case_1():
    none_type_0 = None
    bool_0 = False
    var_0 = module_0.insert(none_type_0, bool_0)
    assert f"{type(var_0).__module__}.{type(var_0).__qualname__}" == "treap.Node"
    assert var_0.value is False
    assert var_0.left is None
    assert var_0.right is None
    var_1 = module_0.insert(var_0, bool_0)
    assert f"{type(var_1).__module__}.{type(var_1).__qualname__}" == "treap.Node"
    assert var_1.value is False
    none_type_1 = None
    int_0 = 1142
    var_2 = module_0.insert(none_type_1, int_0)
    assert f"{type(var_2).__module__}.{type(var_2).__qualname__}" == "treap.Node"
    assert var_2.value == 1142
    assert var_2.left is None
    assert var_2.right is None
    var_3 = module_0.merge(none_type_1, none_type_1)
    set_0 = {var_3}
    var_4 = module_0.merge(none_type_1, set_0)
    none_type_2 = module_0.inorder(var_1)
    str_0 = " [\\9^>(-D;6H'\x0c1YW$Y,"
    var_5 = module_0.interact_treap(none_type_0, str_0)
    str_1 = var_0.__repr__()
    var_6 = module_0.insert(var_2, bool_0)
    assert f"{type(var_6).__module__}.{type(var_6).__qualname__}" == "treap.Node"
    str_2 = var_6.__repr__()
    str_3 = var_0.__str__()
    str_4 = var_1.__str__()
    assert str_4 == "False False "
    str_5 = var_1.__repr__()
    module_0.interact_treap(str_3, none_type_0)


@pytest.mark.xfail(strict=True)
def test_case_2():
    none_type_0 = None
    tuple_0 = module_0.split(none_type_0, none_type_0)
    module_0.interact_treap(none_type_0, none_type_0)


@pytest.mark.xfail(strict=True)
def test_case_3():
    bool_0 = True
    list_0 = [bool_0, bool_0]
    module_0.insert(list_0, bool_0)


@pytest.mark.xfail(strict=True)
def test_case_4():
    none_type_0 = None
    tuple_0 = (none_type_0, none_type_0)
    list_0 = [tuple_0]
    var_0 = module_0.merge(list_0, none_type_0)
    module_0.erase(var_0, none_type_0)


@pytest.mark.xfail(strict=True)
def test_case_5():
    none_type_0 = None
    bool_0 = False
    var_0 = module_0.insert(none_type_0, bool_0)
    assert f"{type(var_0).__module__}.{type(var_0).__qualname__}" == "treap.Node"
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
    assert f"{type(var_0).__module__}.{type(var_0).__qualname__}" == "treap.Node"
    assert var_0.value == {None}
    assert var_0.left is None
    assert var_0.right is None
    str_0 = var_0.__repr__()
    str_1 = var_0.__str__()
    assert str_1 == "{None} "
    module_0.inorder(set_0)


def test_case_7():
    none_type_0 = None
    none_type_1 = module_0.inorder(none_type_0)
    str_0 = "[iID@`[Og/{aD*M?Gg"
    var_0 = module_0.interact_treap(none_type_0, str_0)
    str_1 = var_0.__repr__()
    str_2 = "NQ\x0cH]/6=?"
    var_1 = module_0.interact_treap(none_type_0, str_2)
    tuple_0 = module_0.split(none_type_1, var_1)


@pytest.mark.xfail(strict=True)
def test_case_8():
    bool_0 = False
    bool_1 = False
    list_0 = [bool_0, bool_1]
    str_0 = "\x0bA{q_{!EZ"
    var_0 = module_0.interact_treap(list_0, str_0)
    tuple_0 = (var_0, var_0)
    module_0.interact_treap(tuple_0, var_0)


@pytest.mark.xfail(strict=True)
def test_case_9():
    none_type_0 = None
    var_0 = module_0.insert(none_type_0, none_type_0)
    assert f"{type(var_0).__module__}.{type(var_0).__qualname__}" == "treap.Node"
    assert var_0.value is None
    assert var_0.left is None
    assert var_0.right is None
    var_1 = module_0.insert(var_0, none_type_0)
    assert f"{type(var_1).__module__}.{type(var_1).__qualname__}" == "treap.Node"
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
    str_2 = var_0.__repr__()
    var_1 = module_0.insert(var_0, str_1)
    assert f"{type(var_1).__module__}.{type(var_1).__qualname__}" == "treap.Node"
    assert var_1.value == "None"
    assert var_1.left is None
    assert var_1.right is None
    module_0.split(var_1, none_type_1)


@pytest.mark.xfail(strict=True)
def test_case_11():
    int_0 = 5715
    module_0.merge(int_0, int_0)


@pytest.mark.xfail(strict=True)
def test_case_12():
    none_type_0 = None
    bool_0 = False
    tuple_0 = module_0.split(none_type_0, none_type_0)
    var_0 = module_0.insert(none_type_0, bool_0)
    assert f"{type(var_0).__module__}.{type(var_0).__qualname__}" == "treap.Node"
    assert var_0.value is False
    assert var_0.left is None
    assert var_0.right is None
    var_1 = module_0.insert(var_0, bool_0)
    assert f"{type(var_1).__module__}.{type(var_1).__qualname__}" == "treap.Node"
    assert var_1.value is False
    none_type_1 = None
    int_0 = 1142
    var_2 = module_0.insert(none_type_1, int_0)
    assert f"{type(var_2).__module__}.{type(var_2).__qualname__}" == "treap.Node"
    assert var_2.value == 1142
    assert var_2.left is None
    assert var_2.right is None
    var_3 = module_0.merge(none_type_1, none_type_1)
    set_0 = {var_3}
    var_4 = module_0.merge(none_type_1, set_0)
    str_0 = " [\\9^>(-D;6H'\x0c1YW$Y,"
    var_5 = module_0.interact_treap(var_3, str_0)
    str_1 = var_3.__str__()
    str_2 = var_5.__str__()
    node_0 = module_0.Node()
    module_0.interact_treap(none_type_0, bool_0)


@pytest.mark.xfail(strict=True)
def test_case_13():
    none_type_0 = None
    bool_0 = False
    var_0 = module_0.insert(none_type_0, bool_0)
    assert f"{type(var_0).__module__}.{type(var_0).__qualname__}" == "treap.Node"
    assert var_0.value is False
    assert var_0.left is None
    assert var_0.right is None
    var_1 = module_0.insert(var_0, bool_0)
    assert f"{type(var_1).__module__}.{type(var_1).__qualname__}" == "treap.Node"
    assert var_1.value is False
    none_type_1 = None
    int_0 = 1142
    var_2 = module_0.insert(none_type_1, int_0)
    assert f"{type(var_2).__module__}.{type(var_2).__qualname__}" == "treap.Node"
    assert var_2.value == 1142
    assert var_2.left is None
    assert var_2.right is None
    var_3 = module_0.merge(none_type_1, none_type_1)
    none_type_2 = None
    var_4 = module_0.Node()
    none_type_3 = module_0.insert(none_type_0, var_4)
    str_0 = " [\\9^>(-D;6H'\x0c1YW$Y,"
    var_5 = module_0.interact_treap(none_type_2, str_0)
    str_1 = var_0.__repr__()
    var_6 = module_0.insert(var_2, bool_0)
    assert (
        f"{type(var_2.left).__module__}.{type(var_2.left).__qualname__}" == "treap.Node"
    )
    assert f"{type(var_6).__module__}.{type(var_6).__qualname__}" == "treap.Node"
    assert var_6.value == 1142
    assert (
        f"{type(var_6.left).__module__}.{type(var_6.left).__qualname__}" == "treap.Node"
    )
    assert var_6.right is None
    str_2 = var_6.__repr__()
    str_3 = var_0.__str__()
    str_4 = var_1.__str__()
    assert str_4 == "False False "
    str_5 = var_0.__str__()
    str_6 = var_0.__repr__()
    module_0.interact_treap(var_5, none_type_0)


@pytest.mark.xfail(strict=True)
def test_case_14():
    none_type_0 = None
    bool_0 = False
    var_0 = module_0.insert(none_type_0, bool_0)
    assert f"{type(var_0).__module__}.{type(var_0).__qualname__}" == "treap.Node"
    assert var_0.value is False
    assert var_0.left is None
    assert var_0.right is None
    var_1 = module_0.insert(var_0, bool_0)
    assert f"{type(var_1).__module__}.{type(var_1).__qualname__}" == "treap.Node"
    assert var_1.value is False
    assert (
        f"{type(var_1.left).__module__}.{type(var_1.left).__qualname__}" == "treap.Node"
    )
    assert var_1.right is None
    bool_1 = False
    none_type_1 = None
    int_0 = 1142
    var_2 = module_0.insert(none_type_1, int_0)
    assert f"{type(var_2).__module__}.{type(var_2).__qualname__}" == "treap.Node"
    assert var_2.value == 1142
    assert var_2.left is None
    assert var_2.right is None
    var_3 = module_0.merge(none_type_1, none_type_1)
    none_type_2 = None
    set_0 = {var_3}
    var_4 = module_0.merge(none_type_1, set_0)
    str_0 = " [\\9^>(-D;6H'\x0c1YW$Y,"
    var_5 = module_0.interact_treap(none_type_2, str_0)
    str_1 = var_3.__str__()
    str_2 = var_5.__str__()
    node_0 = module_0.Node()
    module_0.interact_treap(none_type_2, bool_1)


@pytest.mark.xfail(strict=True)
def test_case_15():
    none_type_0 = None
    bool_0 = False
    var_0 = module_0.insert(none_type_0, bool_0)
    assert f"{type(var_0).__module__}.{type(var_0).__qualname__}" == "treap.Node"
    assert var_0.value is False
    assert var_0.left is None
    assert var_0.right is None
    var_1 = module_0.insert(var_0, bool_0)
    assert f"{type(var_1).__module__}.{type(var_1).__qualname__}" == "treap.Node"
    assert var_1.value is False
    assert (
        f"{type(var_1.left).__module__}.{type(var_1.left).__qualname__}" == "treap.Node"
    )
    assert var_1.right is None
    none_type_1 = None
    int_0 = 1142
    var_2 = module_0.insert(none_type_1, int_0)
    assert f"{type(var_2).__module__}.{type(var_2).__qualname__}" == "treap.Node"
    assert var_2.value == 1142
    assert var_2.left is None
    assert var_2.right is None
    var_3 = module_0.merge(none_type_1, none_type_1)
    none_type_2 = None
    set_0 = {var_3}
    var_4 = module_0.merge(none_type_1, set_0)
    none_type_3 = module_0.inorder(var_1)
    str_0 = " [\\9^>(-D;6H'\x0c1YW$Y,"
    var_5 = module_0.interact_treap(none_type_2, str_0)
    str_1 = var_0.__repr__()
    var_6 = module_0.merge(var_5, var_3)
    str_2 = var_6.__repr__()
    str_3 = var_0.__str__()
    assert str_3 == "False "
    str_4 = var_1.__str__()
    assert str_4 == "False False "
    str_5 = var_3.__repr__()
    str_6 = "5y\x0c+C*A"
    module_0.interact_treap(none_type_3, str_6)


@pytest.mark.xfail(strict=True)
def test_case_16():
    none_type_0 = None
    bool_0 = False
    var_0 = module_0.insert(none_type_0, bool_0)
    assert f"{type(var_0).__module__}.{type(var_0).__qualname__}" == "treap.Node"
    assert var_0.value is False
    assert var_0.left is None
    assert var_0.right is None
    var_1 = module_0.insert(var_0, bool_0)
    assert f"{type(var_1).__module__}.{type(var_1).__qualname__}" == "treap.Node"
    assert var_1.value is False
    none_type_1 = None
    var_2 = module_0.merge(none_type_0, var_1)
    assert f"{type(var_2).__module__}.{type(var_2).__qualname__}" == "treap.Node"
    assert var_2.value is False
    str_0 = var_1.__repr__()
    none_type_2 = module_0.inorder(none_type_0)
    var_3 = module_0.interact_treap(none_type_1, str_0)
    str_1 = var_2.__repr__()
    bool_1 = False
    var_4 = module_0.insert(var_0, bool_1)
    assert f"{type(var_4).__module__}.{type(var_4).__qualname__}" == "treap.Node"
    assert var_4.value is False
    assert (
        f"{type(var_4.left).__module__}.{type(var_4.left).__qualname__}" == "treap.Node"
    )
    assert var_4.right is None
    str_2 = var_1.__repr__()
    str_3 = "-"
    module_0.interact_treap(none_type_1, str_3)

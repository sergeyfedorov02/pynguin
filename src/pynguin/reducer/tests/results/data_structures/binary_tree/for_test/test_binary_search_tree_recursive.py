# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import pytest
import binary_search_tree_recursive as module_0
import inspect as module_1
import tokenize as module_2


def test_case_0():
    binary_search_tree_0 = module_0.BinarySearchTree()
    binary_search_tree_1 = module_0.BinarySearchTree()
    iterator_0 = binary_search_tree_1.inorder_traversal()
    int_0 = 1487
    none_type_0 = binary_search_tree_0.put(int_0)
    assert (
        f"{type(binary_search_tree_0.root).__module__}.{type(binary_search_tree_0.root).__qualname__}"
        == "binary_search_tree_recursive.Node"
    )
    with pytest.raises(Exception):
        binary_search_tree_1.get_max_label()


def test_case_1():
    binary_search_tree_0 = module_0.BinarySearchTree()
    bool_0 = binary_search_tree_0.exists(binary_search_tree_0)
    assert bool_0 is False


def test_case_2():
    binary_search_tree_0 = module_0.BinarySearchTree()
    int_0 = 841
    bool_0 = binary_search_tree_0.exists(int_0)
    assert bool_0 is False
    binary_search_tree_1 = module_0.BinarySearchTree()
    none_type_0 = binary_search_tree_1.empty()
    none_type_1 = binary_search_tree_0.empty()
    with pytest.raises(Exception):
        binary_search_tree_0.get_max_label()


@pytest.mark.xfail(strict=True)
def test_case_3():
    binary_search_tree_0 = module_0.BinarySearchTree()
    int_0 = -2049
    bool_0 = binary_search_tree_0.is_empty()
    none_type_0 = binary_search_tree_0.put(binary_search_tree_0)
    assert (
        f"{type(binary_search_tree_0.root).__module__}.{type(binary_search_tree_0.root).__qualname__}"
        == "binary_search_tree_recursive.Node"
    )
    node_0 = module_0.Node(bool_0, none_type_0)
    binary_search_tree_1 = module_0.BinarySearchTree()
    int_1 = binary_search_tree_0.get_min_label()
    assert (
        f"{type(int_1).__module__}.{type(int_1).__qualname__}"
        == "binary_search_tree_recursive.BinarySearchTree"
    )
    assert (
        f"{type(int_1.root).__module__}.{type(int_1.root).__qualname__}"
        == "binary_search_tree_recursive.Node"
    )
    int_2 = binary_search_tree_0.get_max_label()
    assert (
        f"{type(int_2).__module__}.{type(int_2).__qualname__}"
        == "binary_search_tree_recursive.BinarySearchTree"
    )
    assert (
        f"{type(int_2.root).__module__}.{type(int_2.root).__qualname__}"
        == "binary_search_tree_recursive.Node"
    )
    bool_1 = binary_search_tree_0.exists(int_2)
    assert bool_1 is False
    int_3 = binary_search_tree_0.get_min_label()
    binary_search_tree_0.put(int_0)


def test_case_4():
    binary_search_tree_0 = module_0.BinarySearchTree()
    int_0 = 2079
    bool_0 = binary_search_tree_0.exists(int_0)
    assert bool_0 is False
    binary_search_tree_1 = module_0.BinarySearchTree()
    none_type_0 = binary_search_tree_1.empty()
    with pytest.raises(Exception):
        binary_search_tree_1.get_min_label()


def test_case_5():
    binary_search_tree_0 = module_0.BinarySearchTree()


def test_case_6():
    none_type_0 = None
    node_0 = module_0.Node(none_type_0, none_type_0)


def test_case_7():
    binary_search_tree_0 = module_0.BinarySearchTree()
    int_0 = 841
    bool_0 = binary_search_tree_0.exists(int_0)
    assert bool_0 is False
    binary_search_tree_1 = module_0.BinarySearchTree()
    none_type_0 = binary_search_tree_0.empty()


def test_case_8():
    binary_search_tree_0 = module_0.BinarySearchTree()
    bool_0 = binary_search_tree_0.exists(binary_search_tree_0)
    assert bool_0 is False
    iterator_0 = binary_search_tree_0.inorder_traversal()


@pytest.mark.xfail(strict=True)
def test_case_9():
    binary_search_tree_0 = module_0.BinarySearchTree()
    binary_search_tree_1 = module_0.BinarySearchTree()
    iterator_0 = binary_search_tree_0.preorder_traversal()
    bool_0 = False
    binary_search_tree_0.remove(bool_0)


def test_case_10():
    int_0 = 1511
    binary_search_tree_0 = module_0.BinarySearchTree()
    binary_search_tree_1 = module_0.BinarySearchTree()
    iterator_0 = binary_search_tree_1.inorder_traversal()
    int_1 = 1487
    none_type_0 = binary_search_tree_0.put(int_0)
    assert (
        f"{type(binary_search_tree_0.root).__module__}.{type(binary_search_tree_0.root).__qualname__}"
        == "binary_search_tree_recursive.Node"
    )
    none_type_1 = binary_search_tree_0.put(int_1)
    with pytest.raises(Exception):
        binary_search_tree_1.get_max_label()


@pytest.mark.xfail(strict=True)
def test_case_11():
    binary_search_tree_0 = module_0.BinarySearchTree()
    none_type_0 = binary_search_tree_0.put(binary_search_tree_0)
    assert (
        f"{type(binary_search_tree_0.root).__module__}.{type(binary_search_tree_0.root).__qualname__}"
        == "binary_search_tree_recursive.Node"
    )
    binary_search_tree_1 = module_0.BinarySearchTree()
    bool_0 = True
    none_type_1 = binary_search_tree_1.put(bool_0)
    binary_search_tree_1.put(bool_0)


@pytest.mark.xfail(strict=True)
def test_case_12():
    binary_search_tree_0 = module_0.BinarySearchTree()
    bool_0 = True
    none_type_0 = binary_search_tree_0.put(bool_0)
    assert (
        f"{type(binary_search_tree_0.root).__module__}.{type(binary_search_tree_0.root).__qualname__}"
        == "binary_search_tree_recursive.Node"
    )
    iterator_0 = binary_search_tree_0.inorder_traversal()
    bool_1 = False
    none_type_1 = binary_search_tree_0.put(bool_1)
    int_0 = binary_search_tree_0.get_min_label()
    assert int_0 is False
    node_0 = binary_search_tree_0.search(bool_1)
    assert (
        f"{type(node_0).__module__}.{type(node_0).__qualname__}"
        == "binary_search_tree_recursive.Node"
    )
    assert node_0.label is False
    assert (
        f"{type(node_0.parent).__module__}.{type(node_0.parent).__qualname__}"
        == "binary_search_tree_recursive.Node"
    )
    assert node_0.left is None
    assert node_0.right is None
    iterator_1 = binary_search_tree_0.inorder_traversal()
    iterator_1.__or__(iterator_1)


@pytest.mark.xfail(strict=True)
def test_case_13():
    binary_search_tree_0 = module_0.BinarySearchTree()
    bool_0 = True
    int_0 = -1816
    none_type_0 = binary_search_tree_0.put(int_0)
    assert (
        f"{type(binary_search_tree_0.root).__module__}.{type(binary_search_tree_0.root).__qualname__}"
        == "binary_search_tree_recursive.Node"
    )
    none_type_1 = binary_search_tree_0.put(bool_0)
    none_type_2 = binary_search_tree_0.remove(int_0)
    int_1 = 2132
    none_type_3 = binary_search_tree_0.put(int_1)
    iterator_0 = binary_search_tree_0.preorder_traversal()
    int_2 = binary_search_tree_0.get_max_label()
    assert int_2 == 2132
    bool_1 = binary_search_tree_0.exists(iterator_0)
    assert bool_1 is False
    none_type_4 = None
    module_1.formatargspec(
        iterator_0,
        none_type_4,
        annotations=none_type_4,
        formatvarkw=int_2,
        formatreturns=bool_1,
    )


@pytest.mark.xfail(strict=True)
def test_case_14():
    binary_search_tree_0 = module_0.BinarySearchTree()
    int_0 = -1816
    none_type_0 = binary_search_tree_0.put(int_0)
    assert (
        f"{type(binary_search_tree_0.root).__module__}.{type(binary_search_tree_0.root).__qualname__}"
        == "binary_search_tree_recursive.Node"
    )
    int_1 = -1060
    none_type_1 = binary_search_tree_0.put(int_1)
    none_type_2 = binary_search_tree_0.remove(int_0)
    binary_search_tree_0.put(int_1)


def test_case_15():
    binary_search_tree_0 = module_0.BinarySearchTree()
    int_0 = -1816
    none_type_0 = binary_search_tree_0.put(int_0)
    assert (
        f"{type(binary_search_tree_0.root).__module__}.{type(binary_search_tree_0.root).__qualname__}"
        == "binary_search_tree_recursive.Node"
    )
    int_1 = -999
    none_type_1 = binary_search_tree_0.put(int_1)
    none_type_2 = binary_search_tree_0.remove(int_1)
    none_type_3 = binary_search_tree_0.put(int_1)
    iterator_0 = binary_search_tree_0.preorder_traversal()
    int_2 = binary_search_tree_0.get_min_label()
    assert int_2 == -1816
    int_3 = binary_search_tree_0.get_max_label()
    assert int_3 == -999
    bool_0 = binary_search_tree_0.exists(none_type_2)
    assert bool_0 is False
    none_type_4 = None
    var_0 = module_1.formatargspec(
        iterator_0,
        none_type_4,
        kwonlyargs=none_type_1,
        formatvarargs=none_type_1,
        formatvalue=none_type_1,
    )


@pytest.mark.xfail(strict=True)
def test_case_16():
    binary_search_tree_0 = module_0.BinarySearchTree()
    bool_0 = True
    none_type_0 = binary_search_tree_0.put(bool_0)
    assert (
        f"{type(binary_search_tree_0.root).__module__}.{type(binary_search_tree_0.root).__qualname__}"
        == "binary_search_tree_recursive.Node"
    )
    iterator_0 = binary_search_tree_0.inorder_traversal()
    bool_1 = False
    none_type_1 = binary_search_tree_0.put(bool_1)
    int_0 = -999
    bool_2 = binary_search_tree_0.exists(int_0)
    assert bool_2 is False
    none_type_2 = binary_search_tree_0.put(int_0)
    int_1 = binary_search_tree_0.get_min_label()
    assert int_1 == -999
    int_2 = binary_search_tree_0.get_max_label()
    assert int_2 is True
    bool_3 = binary_search_tree_0.exists(none_type_1)
    assert bool_3 is False
    int_3 = 2009
    binary_search_tree_0.search(int_3)


def test_case_17():
    binary_search_tree_0 = module_0.BinarySearchTree()
    bool_0 = True
    none_type_0 = binary_search_tree_0.put(bool_0)
    assert (
        f"{type(binary_search_tree_0.root).__module__}.{type(binary_search_tree_0.root).__qualname__}"
        == "binary_search_tree_recursive.Node"
    )
    none_type_1 = binary_search_tree_0.remove(bool_0)
    assert binary_search_tree_0.root is None
    bool_1 = True
    none_type_2 = binary_search_tree_0.put(bool_1)
    int_0 = binary_search_tree_0.get_max_label()
    assert int_0 is True
    bool_2 = binary_search_tree_0.is_empty()
    int_1 = binary_search_tree_0.get_min_label()
    assert int_1 is True
    bool_3 = binary_search_tree_0.exists(bool_2)
    assert bool_3 is False
    iterator_0 = binary_search_tree_0.inorder_traversal()


@pytest.mark.xfail(strict=True)
def test_case_18():
    binary_search_tree_0 = module_0.BinarySearchTree()
    int_0 = -1816
    none_type_0 = binary_search_tree_0.put(int_0)
    assert (
        f"{type(binary_search_tree_0.root).__module__}.{type(binary_search_tree_0.root).__qualname__}"
        == "binary_search_tree_recursive.Node"
    )
    int_1 = -999
    none_type_1 = binary_search_tree_0.put(int_1)
    int_2 = 2132
    none_type_2 = binary_search_tree_0.put(int_2)
    iterator_0 = binary_search_tree_0.preorder_traversal()
    int_3 = binary_search_tree_0.get_min_label()
    assert int_3 == -1816
    int_4 = binary_search_tree_0.get_max_label()
    assert int_4 == 2132
    bool_0 = binary_search_tree_0.exists(iterator_0)
    assert bool_0 is False
    module_1.formatargspec(
        iterator_0,
        none_type_2,
        annotations=none_type_2,
        formatvarkw=int_4,
        formatreturns=int_1,
    )


@pytest.mark.xfail(strict=True)
def test_case_19():
    binary_search_tree_0 = module_0.BinarySearchTree()
    none_type_0 = binary_search_tree_0.put(binary_search_tree_0)
    assert (
        f"{type(binary_search_tree_0.root).__module__}.{type(binary_search_tree_0.root).__qualname__}"
        == "binary_search_tree_recursive.Node"
    )
    binary_search_tree_1 = module_0.BinarySearchTree()
    binary_search_tree_2 = module_0.BinarySearchTree()
    bool_0 = False
    none_type_1 = binary_search_tree_1.put(bool_0)
    iterator_0 = binary_search_tree_0.inorder_traversal()
    iterator_1 = binary_search_tree_0.inorder_traversal()
    none_type_2 = binary_search_tree_1.remove(bool_0)
    assert binary_search_tree_1.root is None
    bool_1 = True
    none_type_3 = binary_search_tree_1.put(bool_1)
    int_0 = binary_search_tree_0.get_min_label()
    assert (
        f"{type(int_0).__module__}.{type(int_0).__qualname__}"
        == "binary_search_tree_recursive.BinarySearchTree"
    )
    assert (
        f"{type(int_0.root).__module__}.{type(int_0.root).__qualname__}"
        == "binary_search_tree_recursive.Node"
    )
    int_1 = binary_search_tree_1.get_max_label()
    assert int_1 is True
    int_2 = binary_search_tree_0.get_min_label()
    int_3 = binary_search_tree_0.get_max_label()
    module_2.untokenize(iterator_1)


@pytest.mark.xfail(strict=True)
def test_case_20():
    binary_search_tree_0 = module_0.BinarySearchTree()
    bool_0 = True
    none_type_0 = binary_search_tree_0.put(bool_0)
    assert (
        f"{type(binary_search_tree_0.root).__module__}.{type(binary_search_tree_0.root).__qualname__}"
        == "binary_search_tree_recursive.Node"
    )
    int_0 = -999
    none_type_1 = binary_search_tree_0.put(int_0)
    none_type_2 = binary_search_tree_0.remove(int_0)
    iterator_0 = binary_search_tree_0.preorder_traversal()
    int_1 = binary_search_tree_0.get_min_label()
    assert int_1 is True
    int_2 = binary_search_tree_0.get_max_label()
    assert int_2 is True
    none_type_3 = None
    module_1.formatargspec(
        iterator_0,
        none_type_3,
        annotations=none_type_3,
        formatvarkw=int_2,
        formatreturns=int_0,
    )


@pytest.mark.xfail(strict=True)
def test_case_21():
    binary_search_tree_0 = module_0.BinarySearchTree()
    bool_0 = True
    none_type_0 = binary_search_tree_0.put(bool_0)
    assert (
        f"{type(binary_search_tree_0.root).__module__}.{type(binary_search_tree_0.root).__qualname__}"
        == "binary_search_tree_recursive.Node"
    )
    int_0 = -999
    none_type_1 = binary_search_tree_0.put(int_0)
    none_type_2 = binary_search_tree_0.remove(bool_0)
    int_1 = 2107
    none_type_3 = binary_search_tree_0.put(int_1)
    iterator_0 = binary_search_tree_0.preorder_traversal()
    int_2 = binary_search_tree_0.get_min_label()
    assert int_2 == -999
    int_3 = binary_search_tree_0.get_max_label()
    assert int_3 == 2107
    bool_1 = binary_search_tree_0.exists(iterator_0)
    assert bool_1 is False
    bool_2 = binary_search_tree_0.is_empty()
    int_4 = 1723
    binary_search_tree_0.search(int_4)

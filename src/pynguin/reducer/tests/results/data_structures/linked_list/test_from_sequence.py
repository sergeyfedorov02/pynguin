# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import pytest
import from_sequence as module_0


def test_case_0():
    node_0 = module_0.Node()
    var_0 = node_0.__repr__()
    assert var_0 == "<None> ---> <END>"


@pytest.mark.xfail(strict=True)
def test_case_1():
    int_0 = 1785
    module_0.make_linked_list(int_0)


def test_case_2():
    none_type_0 = None
    with pytest.raises(Exception):
        module_0.make_linked_list(none_type_0)


def test_case_3():
    node_0 = module_0.Node()
    bytes_0 = b"^H(p\xd5\x8b\xb4"
    var_0 = module_0.make_linked_list(bytes_0)
    assert (
        f"{type(var_0).__module__}.{type(var_0).__qualname__}" == "from_sequence.Node"
    )
    assert var_0.data == 94
    assert (
        f"{type(var_0.next).__module__}.{type(var_0.next).__qualname__}"
        == "from_sequence.Node"
    )
    var_1 = module_0.make_linked_list(bytes_0)
    assert var_1.data == 94
    assert (
        f"{type(var_1.next).__module__}.{type(var_1.next).__qualname__}"
        == "from_sequence.Node"
    )
    var_2 = module_0.make_linked_list(bytes_0)
    assert var_2.data == 94
    assert (
        f"{type(var_2.next).__module__}.{type(var_2.next).__qualname__}"
        == "from_sequence.Node"
    )
    node_1 = module_0.Node(var_2)
    bytes_1 = b"z\x80\xeb\xda\x87(\x0e\xf5\x82\x14\xd9/\x11\x98\x11\xac\xdc<"
    var_3 = module_0.make_linked_list(bytes_1)
    assert var_3.data == 122
    assert (
        f"{type(var_3.next).__module__}.{type(var_3.next).__qualname__}"
        == "from_sequence.Node"
    )
    var_4 = var_3.__repr__()
    assert (
        var_4
        == "<122> ---> <128> ---> <235> ---> <218> ---> <135> ---> <40> ---> <14> ---> <245> ---> <130> ---> <20> ---> <217> ---> <47> ---> <17> ---> <152> ---> <17> ---> <172> ---> <220> ---> <60> ---> <END>"
    )
    node_2 = module_0.Node()
    node_3 = module_0.Node()
    node_4 = module_0.Node(node_1)
    var_5 = var_4.__repr__()
    assert (
        var_5
        == "'<122> ---> <128> ---> <235> ---> <218> ---> <135> ---> <40> ---> <14> ---> <245> ---> <130> ---> <20> ---> <217> ---> <47> ---> <17> ---> <152> ---> <17> ---> <172> ---> <220> ---> <60> ---> <END>'"
    )

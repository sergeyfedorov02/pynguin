# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import pytest
import page_rank as module_0


def test_case_0():
    list_0 = []
    var_0 = module_0.page_rank(list_0)


@pytest.mark.xfail(strict=True)
def test_case_1():
    bytes_0 = b"\xf1\x0e*Ww\xb7\x8a\xaaJ\xf5nF\x0e\xe2\x85U\xa2\x80\xc9"
    module_0.page_rank(bytes_0, bytes_0, bytes_0)


def test_case_2():
    list_0 = []
    var_0 = module_0.Node(list_0)


@pytest.mark.xfail(strict=True)
def test_case_3():
    list_0 = []
    none_type_0 = None
    node_0 = module_0.Node(list_0)
    var_0 = node_0.add_inbound(none_type_0)
    var_1 = module_0.page_rank(list_0)
    bool_0 = True
    node_1 = module_0.Node(bool_0)
    var_2 = node_0.add_inbound(node_0)
    bytes_0 = b"f\x0e\xa1\xff\xf5\xa5\x17]\x88"
    module_0.page_rank(bytes_0, bytes_0, none_type_0)


def test_case_4():
    int_0 = -3716
    tuple_0 = ()
    list_0 = [tuple_0]
    list_1 = [list_0, tuple_0]
    node_0 = module_0.Node(list_1)
    var_0 = node_0.add_outbound(int_0)


def test_case_5():
    bytes_0 = b"\xf1\x0e*Ww\xb7\x8a\xaaJ\xf5nF\x0e\xe2\x85{\xa2\x80\xc9"
    var_0 = module_0.Node(bytes_0)
    var_1 = var_0.__repr__()
    assert (
        var_1
        == "<node=b'\\xf1\\x0e*Ww\\xb7\\x8a\\xaaJ\\xf5nF\\x0e\\xe2\\x85{\\xa2\\x80\\xc9' inbound=[] outbound=[]>"
    )


@pytest.mark.xfail(strict=True)
def test_case_6():
    int_0 = -2246
    node_0 = module_0.Node(int_0)
    node_1 = module_0.Node(node_0)
    list_0 = [node_1, node_0]
    var_0 = module_0.page_rank(list_0)
    var_1 = node_1.__repr__()
    assert var_1 == "<node=<node=-2246 inbound=[] outbound=[]> inbound=[] outbound=[]>"
    var_2 = node_1.__repr__()
    assert var_2 == "<node=<node=-2246 inbound=[] outbound=[]> inbound=[] outbound=[]>"
    var_2.add_inbound(var_1)


@pytest.mark.xfail(strict=True)
def test_case_7():
    int_0 = -2246
    node_0 = module_0.Node(int_0)
    bool_0 = False
    var_0 = node_0.add_inbound(bool_0)
    node_1 = module_0.Node(bool_0)
    var_1 = module_0.Node(int_0)
    node_2 = module_0.Node(int_0)
    list_0 = [node_0, node_1]
    module_0.page_rank(list_0)

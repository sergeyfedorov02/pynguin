# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import pytest
import radix_tree as module_0

@pytest.mark.xfail(strict=True)
def test_case_3():
    str_1 = 'aS}XDl}pvSm:vCA,'
    radix_node_0 = module_0.RadixNode(str_1)
    radix_node_0.print_tree(radix_node_0)

@pytest.mark.xfail(strict=True)
def test_case_8():
    str_0 = 'f*\rR:/<'
    radix_node_0 = module_0.RadixNode()
    assert radix_node_0.prefix == ''
    str_1 = 'foY@P15\\I9%X?Aac'
    list_0 = [str_0, str_1]
    none_type_0 = radix_node_0.insert_many(list_0)
    assert len(radix_node_0.nodes) == 1
    bool_0 = radix_node_0.find(str_0)
    assert bool_0 is True
    bool_1 = radix_node_0.delete(str_0)
    assert bool_1 is True
    bool_2 = radix_node_0.find(str_0)
    assert bool_2 is False

def test_case_10():
    str_1 = 'aS}XDl}pvSm:vCA,'
    radix_node_2 = module_0.RadixNode(str_1)
    none_type_1 = radix_node_2.insert(str_1)
    assert radix_node_2.is_leaf is True

@pytest.mark.xfail(strict=True)
def test_case_11():
    str_0 = '\\\x0b*jb|yi~pT-aBs4'
    radix_node_0 = module_0.RadixNode()
    assert radix_node_0.prefix == ''
    none_type_0 = radix_node_0.insert(str_0)
    assert len(radix_node_0.nodes) == 1
    str_1 = '(##)2NY5>NLw\t2J8'
    radix_node_1 = module_0.RadixNode()
    assert radix_node_1.prefix == ''
    tuple_0 = radix_node_1.match(str_1)
    str_2 = 'D*'
    list_0 = [str_2]
    bool_0 = radix_node_0.delete(list_0)
    assert bool_0 is False

@pytest.mark.xfail(strict=True)
def test_case_13():
    str_0 = 'f*\rR:/<'
    radix_node_0 = module_0.RadixNode()
    assert radix_node_0.prefix == ''
    str_1 = 'foY@P15\\I9%X?Aac'
    none_type_0 = radix_node_0.insert_many(str_0)
    assert len(radix_node_0.nodes) == 7
    list_0 = [str_0, str_1]
    none_type_1 = radix_node_0.insert_many(list_0)
    bool_0 = radix_node_0.delete(str_0)
    assert bool_0 is True
    radix_node_0.insert_many(str_0)

@pytest.mark.xfail(strict=True)
def test_case_14():
    str_0 = 'f*\rR:/<'
    radix_node_0 = module_0.RadixNode()
    assert radix_node_0.prefix == ''
    str_1 = 'foY@P15\\I9%X?Aac'
    list_0 = [str_1]
    none_type_0 = radix_node_0.insert_many(list_0)
    assert len(radix_node_0.nodes) == 1
    bool_0 = radix_node_0.delete(str_0)
    assert bool_0 is False
    none_type_1 = radix_node_0.print_tree()
    none_type_2 = radix_node_0.insert_many(str_0)
    assert len(radix_node_0.nodes) == 7
    bool_1 = radix_node_0.find(str_0)
    assert bool_1 is False
    radix_node_2.insert(bool_2)
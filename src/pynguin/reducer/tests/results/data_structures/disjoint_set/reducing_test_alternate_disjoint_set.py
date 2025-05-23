# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import pytest
import alternate_disjoint_set as module_0

def test_case_3():
    bool_0 = True
    list_0 = [bool_0, bool_0, bool_0, bool_0, bool_0]
    disjoint_set_0 = module_0.DisjointSet(list_0)
    assert f'{type(disjoint_set_0).__module__}.{type(disjoint_set_0).__qualname__}' == 'alternate_disjoint_set.DisjointSet'
    assert disjoint_set_0.set_counts == [True, True, True, True, True]
    assert disjoint_set_0.max_set is True
    assert disjoint_set_0.ranks == [1, 1, 1, 1, 1]
    assert disjoint_set_0.parents == [0, 1, 2, 3, 4]
    bool_1 = False
    bool_2 = disjoint_set_0.merge(bool_0, bool_1)
    assert bool_2 is True
    assert disjoint_set_0.set_counts == [2, 0, True, True, True]
    assert disjoint_set_0.max_set == 2
    assert disjoint_set_0.ranks == [2, 1, 1, 1, 1]
    assert disjoint_set_0.parents == [0, False, 2, 3, 4]
    bool_3 = disjoint_set_0.merge(bool_2, bool_2)
    assert bool_3 is False
    int_0 = -3
    bool_4 = disjoint_set_0.merge(int_0, bool_3)
    assert bool_4 is True
    assert disjoint_set_0.set_counts == [3, 0, 0, True, True]
    assert disjoint_set_0.max_set == 3
    assert disjoint_set_0.parents == [0, False, False, 3, 4]

@pytest.mark.xfail(strict=True)
def test_case_4():
    bool_0 = False
    list_0 = [bool_0, bool_0, bool_0, bool_0]
    disjoint_set_0 = module_0.DisjointSet(list_0)
    assert f'{type(disjoint_set_0).__module__}.{type(disjoint_set_0).__qualname__}' == 'alternate_disjoint_set.DisjointSet'
    assert disjoint_set_0.set_counts == [False, False, False, False]
    assert disjoint_set_0.max_set is False
    assert disjoint_set_0.ranks == [1, 1, 1, 1]
    assert disjoint_set_0.parents == [0, 1, 2, 3]
    int_0 = 2
    none_type_0 = None
    bool_1 = True
    bool_2 = disjoint_set_0.merge(bool_1, bool_0)
    assert bool_2 is True
    assert disjoint_set_0.ranks == [2, 1, 1, 1]
    assert disjoint_set_0.parents == [0, False, 2, 3]
    bool_3 = disjoint_set_0.merge(bool_0, int_0)
    assert bool_3 is True
    assert disjoint_set_0.parents == [0, False, False, 3]
    module_0.DisjointSet(none_type_0)
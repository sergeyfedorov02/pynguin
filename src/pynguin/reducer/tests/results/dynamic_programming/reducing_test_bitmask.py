# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import pytest
import collections as module_0
import bitmask as module_1

@pytest.mark.xfail(strict=True)
def test_case_0():
    defaultdict_0 = module_0.defaultdict()
    module_1.AssignmentUsingBitmask(defaultdict_0, defaultdict_0)

@pytest.mark.xfail(strict=True)
def test_case_2():
    bool_0 = True
    str_0 = 'V<^[Y$'
    assignment_using_bitmask_0 = module_1.AssignmentUsingBitmask(str_0, bool_0)
    assert assignment_using_bitmask_0.dp == [[-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1]]
    assert assignment_using_bitmask_0.final_mask == 63
    var_0 = assignment_using_bitmask_0.count_no_of_ways(str_0)
    assert var_0 == 0
    assert assignment_using_bitmask_0.dp == [[-1, 0], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1]]
    assert len(assignment_using_bitmask_0.task) == 6
    var_1 = assignment_using_bitmask_0.count_no_of_ways(str_0)
    assert var_1 == 0
    module_1.AssignmentUsingBitmask(assignment_using_bitmask_0, var_1)

@pytest.mark.xfail(strict=True)
def test_case_3():
    bool_0 = True
    str_0 = ''
    assignment_using_bitmask_0 = module_1.AssignmentUsingBitmask(str_0, bool_0)
    assert assignment_using_bitmask_0.dp == [[-1, -1]]
    assert assignment_using_bitmask_0.final_mask == 0
    var_0 = assignment_using_bitmask_0.count_no_of_ways(str_0)
    assert var_0 == 1
    module_0.defaultdict(*var_0)
# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import pytest
import breadth_first_search as module_0

@pytest.mark.xfail(strict=True)
def test_case_5():
    graph_0 = module_0.Graph()
    none_type_0 = None
    none_type_1 = graph_0.print_graph()
    bool_0 = False
    none_type_3 = graph_0.add_edge(bool_0, bool_0)
    assert graph_0.vertices == {False: [False]}
# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import pytest
import depth_first_search_2 as module_0

@pytest.mark.xfail(strict=True)
def test_case_0():
    graph_0 = module_0.Graph()
    int_0 = 993
    none_type_0 = graph_0.add_edge(graph_0, int_0)
    assert len(graph_0.vertex) == 1
    graph_0.dfs()

def test_case_5():
    graph_0 = module_0.Graph()
    graph_1 = module_0.Graph()
    none_type_0 = graph_1.add_edge(graph_0, graph_0)
    assert len(graph_1.vertex) == 1
    none_type_3 = graph_1.add_edge(graph_0, graph_1)

@pytest.mark.xfail(strict=True)
def test_case_6():
    bool_0 = False
    graph_0 = module_0.Graph()
# Test cases automatically generated by Pynguin (https://www.pynguin.eu).
# Please check them before you use them.
import pytest
import breadth_first_search as module_0


def test_case_0():
    graph_0 = module_0.Graph()
    graph_1 = module_0.Graph()
    bool_0 = True
    none_type_0 = graph_1.print_graph()
    none_type_1 = graph_1.print_graph()
    none_type_2 = graph_1.add_edge(bool_0, bool_0)
    assert graph_1.vertices == {True: [True]}
    none_type_3 = graph_0.add_edge(bool_0, bool_0)
    graph_2 = module_0.Graph()
    none_type_4 = graph_2.add_edge(none_type_3, bool_0)
    none_type_5 = graph_1.print_graph()
    none_type_6 = graph_2.add_edge(none_type_5, graph_2)
    graph_3 = module_0.Graph()
    none_type_7 = graph_1.print_graph()
    bool_1 = False
    none_type_8 = graph_2.print_graph()
    bool_2 = False
    none_type_9 = graph_0.add_edge(bool_1, bool_2)


def test_case_1():
    graph_0 = module_0.Graph()
    none_type_0 = graph_0.print_graph()
    none_type_1 = graph_0.print_graph()


@pytest.mark.xfail(strict=True)
def test_case_2():
    none_type_0 = None
    bool_0 = False
    graph_0 = module_0.Graph()
    none_type_1 = graph_0.add_edge(bool_0, bool_0)
    assert graph_0.vertices == {False: [False]}
    graph_1 = module_0.Graph()
    graph_1.bfs(none_type_0)


@pytest.mark.xfail(strict=True)
def test_case_3():
    graph_0 = module_0.Graph()
    graph_0.bfs(graph_0)


@pytest.mark.xfail(strict=True)
def test_case_4():
    bool_0 = True
    graph_0 = module_0.Graph()
    none_type_0 = graph_0.add_edge(bool_0, bool_0)
    assert graph_0.vertices == {True: [True]}
    int_0 = -508
    none_type_1 = graph_0.print_graph()
    none_type_2 = graph_0.add_edge(bool_0, int_0)
    graph_1 = module_0.Graph()
    none_type_3 = graph_1.add_edge(bool_0, none_type_0)
    bool_1 = True
    graph_1.bfs(bool_1)


@pytest.mark.xfail(strict=True)
def test_case_5():
    graph_0 = module_0.Graph()
    none_type_0 = None
    none_type_1 = graph_0.print_graph()
    bool_0 = False
    none_type_2 = graph_0.print_graph()
    graph_1 = module_0.Graph()
    none_type_3 = graph_0.add_edge(bool_0, bool_0)
    assert graph_0.vertices == {False: [False]}
    bool_1 = False
    set_0 = graph_0.bfs(bool_1)
    graph_2 = module_0.Graph()
    none_type_4 = graph_0.add_edge(none_type_3, none_type_1)
    int_0 = -2914
    int_1 = 2316
    bytes_0 = b'\x96\x19VMR"\xf9(\xbb\xf2o\xca'
    none_type_5 = graph_0.add_edge(int_1, bytes_0)
    none_type_6 = graph_0.add_edge(bool_0, bool_0)
    none_type_7 = graph_0.add_edge(none_type_6, none_type_0)
    none_type_8 = graph_0.add_edge(none_type_4, int_0)
    none_type_9 = graph_0.add_edge(int_0, int_0)
    set_1 = graph_0.bfs(none_type_0)
    none_type_10 = graph_0.print_graph()
    none_type_11 = graph_0.print_graph()
    none_type_12 = graph_0.print_graph()
    graph_3 = module_0.Graph()
    none_type_13 = graph_3.print_graph()
    none_type_14 = graph_0.print_graph()
    none_type_15 = graph_3.add_edge(int_0, set_1)
    assert graph_3.vertices == {-2914: [{-2914, None}]}
    graph_4 = module_0.Graph()
    int_2 = -99
    none_type_16 = graph_3.print_graph()
    graph_2.bfs(int_2)

import unittest
from jgf import JgfEdge, JgfNode, _Guard, JgfGraph, JgfMultiGraph, Jgf # , JgfDirectedHyperEdge, JgfUndirectedHyperEdge,

import unittest

class TestGuard(unittest.TestCase):
    
    # describe('#non empty string parameter')
    def test_non_empty_string_throws_on_invalid_input(self):
        """should throw error on anything else than a non-empty string"""
        invalid_inputs = [
            12,         # Number
            None,       # Null
            {},         # Empty Dict
            ""          # Empty String
        ]
        
        for invalid in invalid_inputs:
            with self.assertRaises(ValueError):
                _Guard.assert_non_empty_string_parameter('bla', invalid)

    def test_non_empty_string_accepts_valid_input(self):
        """should not throw error on an non-empty string"""
        _Guard.assert_non_empty_string_parameter('bla', 'hello')
        _Guard.assert_non_empty_string_parameter('bla', 'ok')

    # describe('#valid metadata')
    def test_valid_metadata_throws_on_invalid_input(self):
        """should throw error on invalid metadata"""
        # Note: Guard logic enforces non-empty dictionary
        invalid_inputs = [
            'bla',      # String
            None,       # Null
            [],         # List
            {}          # Empty Dict
        ]
        
        for invalid in invalid_inputs:
            with self.assertRaises(ValueError):
                _Guard.assert_valid_metadata(invalid)

    def test_valid_metadata_accepts_valid_input(self):
        """should not throw error valid metadata"""
        _Guard.assert_valid_metadata({'some': 'data'})
        _Guard.assert_valid_metadata({'some': 'data', 'more': 'stuff'})

    # describe('#valid metadata or null')
    def test_valid_metadata_or_null_throws_on_invalid_input(self):
        """should throw error on invalid metadata"""
        invalid_inputs = [
            'bla',
            [],
            {}          # Still invalid because if it exists, it must be non-empty
        ]

        for invalid in invalid_inputs:
            with self.assertRaises(ValueError):
                _Guard.assert_valid_metadata_or_null(invalid)

    def test_valid_metadata_or_null_accepts_valid_input(self):
        """should not throw error valid metadata or null"""
        _Guard.assert_valid_metadata_or_null(None)
        _Guard.assert_valid_metadata_or_null({'some': 'data'})
        _Guard.assert_valid_metadata_or_null({'some': 'data', 'more': 'stuff'})

    # describe('#valid directed')
    def test_valid_directed_throws_on_invalid_input(self):
        """should throw error on invalid directed"""
        invalid_inputs = [
            'bla',
            None,
            2,
            []
        ]

        for invalid in invalid_inputs:
            with self.assertRaises(ValueError):
                _Guard.assert_valid_directed(invalid)

    def test_valid_directed_accepts_valid_input(self):
        """should not throw error valid directed"""
        _Guard.assert_valid_directed(True)
        _Guard.assert_valid_directed(False)

    # ---------------------------------------------------------
    # NEW TESTS FOR V2 SPEC (Hyperedges)
    # ---------------------------------------------------------

    # # describe('#list of strings (Hyperedges)')
    # def test_list_of_strings_throws_on_invalid_input(self):
    #     """should throw error if input is not a list of strings or is empty"""
    #     invalid_inputs = [
    #         "not a list",       # String
    #         123,                # Integer
    #         None,               # None
    #         [1, 2, 3],          # List of Integers
    #         ["a", 1],           # Mixed List
    #         []                  # Empty List (Hyperedges must contain nodes)
    #     ]

    #     for invalid in invalid_inputs:
    #         with self.assertRaises(ValueError):
    #             Guard.assert_list_of_strings('hyperedge_nodes', invalid)

    # def test_list_of_strings_accepts_valid_input(self):
    #     """should accept a non-empty list of strings"""
    #     Guard.assert_list_of_strings('hyperedge_nodes', ['node1', 'node2'])
    #     Guard.assert_list_of_strings('hyperedge_nodes', ['single_node'])

class TestEdge(unittest.TestCase):
    
    # describe('#constructor')
    def test_constructor_sets_passed_parameters_to_properties(self):
        """should set passed parameters to properties"""
        edge = JgfEdge(
            source='earth', 
            target='moon', 
            relation='has-satellite', 
        )

        self.assertEqual(edge.source, 'earth')
        self.assertEqual(edge.target, 'moon')
        self.assertEqual(edge.relation, 'has-satellite')

    def test_constructor_allows_omitting_optional_parameters(self):
        """should allow omitting optional parameters"""
        edge = JgfEdge('earth', 'moon')

        self.assertEqual(edge.source, 'earth')
        self.assertEqual(edge.target, 'moon')
        self.assertIsNone(edge.relation)


    # describe('#mutators')
    def test_mutators_throw_error_on_setting_invalid_ids(self):
        """should throw error on setting invalid source id or target id"""
        edge = JgfEdge('earth', 'moon')

        # Source validation
        with self.assertRaises(ValueError):
            edge.source = None
        with self.assertRaises(ValueError):
            edge.source = 2  # Not a string
        with self.assertRaises(ValueError):
            edge.source = [] # Not a string (empty list)
        with self.assertRaises(ValueError):
             edge.source = "" # Empty string

        # Target validation
        with self.assertRaises(ValueError):
            edge.target = None
        with self.assertRaises(ValueError):
            edge.target = 2
        with self.assertRaises(ValueError):
            edge.target = []
        with self.assertRaises(ValueError):
            edge.target = ""

    def test_mutators_set_and_get_valid_source(self):
        """should set and get valid source"""
        edge = JgfEdge('earth', 'moon')
        edge.source = 'orbit'
        self.assertEqual(edge.source, 'orbit')

    def test_mutators_set_and_get_valid_target(self):
        """should set and get valid target"""
        edge = JgfEdge('earth', 'moon')
        edge.target = 'orbit'
        self.assertEqual(edge.target, 'orbit')

    def test_mutators_throw_error_on_setting_invalid_metadata(self):
        """should throw error on setting invalid metadata"""
        edge = JgfGraph('earth', 'moon')

        with self.assertRaises(ValueError):
            edge.metadata = 2
        with self.assertRaises(ValueError):
            edge.metadata = []
        with self.assertRaises(ValueError):
            edge.metadata = {} # Empty dict is invalid per Guard logic

    def test_mutators_set_and_get_valid_metadata(self):
        """should set and get valid metadata"""
        edge = JgfEdge('earth', 'moon')

        edge.metadata = {'bla': 'bli'}
        self.assertEqual(edge.metadata, {'bla': 'bli'})

        edge.metadata = None
        self.assertIsNone(edge.metadata)

    def test_mutators_throw_error_on_setting_invalid_directed(self):
        """should throw error on setting invalid directed"""
        edge = JgfGraph('thing')

        with self.assertRaises(ValueError):
            edge.directed = 2
        with self.assertRaises(ValueError):
            edge.directed = []
        with self.assertRaises(ValueError):
            edge.directed = {}
        with self.assertRaises(ValueError):
            edge.directed = None

    # describe('#is equal to')
    def test_is_equal_to_knows_equal_edges_are_equal(self):
        """should know equal edges are equal"""
        # Note: Ensure JgfEdge class has the is_equal_to method implemented
        edge = JgfEdge('earth', 'moon')
        equal_edge = JgfEdge('earth', 'moon')

        self.assertTrue(edge.__eq__(equal_edge))

        edge = JgfEdge('earth', 'moon', 'is-satellite')
        equal_edge = JgfEdge('earth', 'moon', 'is-satellite')

        self.assertTrue(edge.__eq__(equal_edge))

        # Different relation
        edge = JgfEdge('earth', 'moon', 'is-satellite')
        equal_edge = JgfEdge('earth', 'moon', 'attracts')

        self.assertTrue(edge.__eq__(equal_edge) == False)

    def test_is_equal_to_knows_different_edges_are_not_equal(self):
        """should know different edges are not equal"""
        edge = JgfEdge('earth', 'moon')
        different_edge = JgfEdge('earth', 'sun')

        self.assertFalse(edge.__eq__(different_edge))

        edge = JgfEdge('earth', 'moon', 'is-satellite')
        different_edge = JgfEdge('earth', 'moon', 'attracts')

        # compare_relation=True, so they should differ
        self.assertFalse(edge.__eq__(different_edge))

class TestNode(unittest.TestCase):
    
    # describe('#constructor')
    def test_constructor_sets_passed_parameters_to_properties(self):
        """should set passed parameters to properties"""
        node = JgfNode('first-knot', 'First Knot', {'foo': 'bar'})

        self.assertEqual(node.id, 'first-knot')
        self.assertEqual(node.label, 'First Knot')
        self.assertEqual(node.metadata, {'foo': 'bar'})

    def test_constructor_only_sets_valid_objects_as_metadata(self):
        """should only set objects passed as metadata to metadata property"""
        invalid_metadata_list = [
            'string-metadata', 
            2, 
            ['bla'], 
            {} 
        ]

        for invalid_metadata in invalid_metadata_list:
            with self.assertRaises(ValueError):
                JgfNode('id', 'label', invalid_metadata)

    # describe('#mutators')
    def test_mutators_set_and_get_metadata(self):
        """should set and get metadata"""
        node = JgfNode('id', 'label')
        node.metadata = {'some': 'thing'}
        self.assertEqual(node.metadata, {'some': 'thing'})

    def test_mutators_only_set_valid_objects_as_metadata(self):
        """should only set objects passed as metadata to metadata property"""
        node = JgfNode('id', 'label')
        
        invalid_metadata_list = [
            'string-metadata', 
            2, 
            ['bla'], 
            {} 
        ]

        for invalid_metadata in invalid_metadata_list:
            with self.assertRaises(ValueError):
                node.metadata = invalid_metadata

from unittest.mock import patch

class TestGraph(unittest.TestCase):

    # describe('#add node(s)')
    def test_add_simple_node(self):
        """should add a simple node"""
        graph = JgfGraph()
        node_id = 'lebron-james#2254'
        node_label = 'LeBron James'

        graph.add_node(JgfNode(node_id, node_label))

        # V2 Change: Nodes are stored in a Dictionary {id: Node}
        
        self.assertEqual(len(graph.nodes), 1)
        self.assertEqual(graph.nodes[node_id].id, node_id)
        self.assertEqual(graph.nodes[node_id].label, node_label)

    def test_add_node_with_metadata(self):
        """should add a node to a graph, with meta data"""
        graph = JgfGraph()
        node_id = 'kevin-durant#4497'
        node_label = 'Kevin Durant'
        metadata = {
            'type': 'NBAPlayer',
            'position': 'Power Forward',
            'shirt': 35
        }

        graph.add_node(JgfNode(node_id, node_label, metadata))

        self.assertEqual(len(graph.nodes), 1)
        self.assertEqual(graph.nodes[node_id].metadata['position'], 'Power Forward')
        self.assertEqual(graph.nodes[node_id].metadata['shirt'], 35)

    def test_add_node_throws_on_duplicate(self):
        """should throw error when adding a node that already exists"""
        graph = JgfGraph()
        node_id = 'kevin-durant#4497'
        node_label = 'Kevin Durant'

        graph.add_node(JgfNode(node_id, node_label))

        with self.assertRaisesRegex(ValueError, 'A node already exists'):
            graph.add_node(JgfNode(node_id, node_label))

    def test_add_nodes_throws_on_duplicate(self):
        """should throw error when adding nodes that already exist"""
        graph = JgfGraph()
        node_id = 'kevin-durant#4497'
        node_label = 'Kevin Durant'

        graph.add_node(JgfNode(node_id, node_label))

        more_nodes = [
            JgfNode(node_id, node_label),
            JgfNode('kyrie-irving#9876', 'Kyrie Irving'),
        ]

        with self.assertRaisesRegex(ValueError, 'A node already exists'):
            graph.add_nodes(more_nodes)

    def test_add_multiple_nodes_at_once(self):
        """should add multiple nodes at once"""
        graph = JgfGraph()
        more_nodes = [
            JgfNode(id1:='kevin-durant#4497', 'Kevin Durant'),
            JgfNode(id2:='kyrie-irving#9876', 'Kyrie Irving'),
        ]

        graph.add_nodes(more_nodes)

        self.assertEqual(graph.nodes[id1].id, 'kevin-durant#4497')
        self.assertEqual(graph.nodes[id1].label, 'Kevin Durant')
        self.assertEqual(graph.nodes[id2].id, 'kyrie-irving#9876')
        self.assertEqual(graph.nodes[id2].label, 'Kyrie Irving')

    # describe('#removeNode')
    def test_remove_node(self):
        """should remove a node"""
        graph = JgfGraph()
        node_id = 'kevin-durant#4497'
        node_label = 'Kevin Durant'

        graph.add_node(n:=JgfNode(node_id, node_label))
        
        graph.remove_node(n)
        
        self.assertEqual(len(graph.nodes), 0, 'After remove_node there should be zero nodes')

    def test_remove_non_existent_node_throws(self):
        """should throw error when removing a non existant node"""
        graph = JgfGraph()
        
        with self.assertRaisesRegex(ValueError, 'A node does not exist'):
            graph.remove_node('some dummy id')

    # describe('#getNode')
    def test_get_node(self):
        """should lookup a node by id"""
        graph = JgfGraph()
        node_id = 'kevin-durant#4497'
        node_label = 'Kevin Durant'

        graph.add_node(JgfNode(node_id, node_label))

        node = graph.get_node_by_id(node_id)
        self.assertIsNotNone(node)
        self.assertEqual(node.id, node_id)

    def test_get_non_existent_node_throws(self):
        """should throw error when looking up a non existant node"""
        graph = JgfGraph()
        with self.assertRaisesRegex(ValueError, 'A node does not exist'):
            graph.get_node_by_id('some dummy id')

    # describe('#addEdge')
    def test_add_simple_edge(self):
        """should add a simple edge to a graph"""
        graph = JgfGraph()
        node1_id = 'lebron-james#2254'
        node2_id = 'la-lakers#1610616839'
        relation = 'Plays for'

        graph.add_node(JgfNode(node1_id, 'LeBron James'))
        graph.add_node(JgfNode(node2_id, 'Los Angeles Lakers'))

        self.assertEqual(len(graph.nodes), 2)

        graph.add_edge(JgfEdge(node1_id, node2_id, relation))

        self.assertEqual(len(graph.edges), 1)
        self.assertEqual(graph.edges[0].relation, relation)

    def test_add_edge_throws_if_nodes_do_not_exist(self):
        """should throw error if source or target nodes do not exist"""
        graph = JgfGraph()
        node1_id = 'lebron-james#234'
        node2_id = 'la-lakers#12345'

        graph.add_node(JgfNode(node1_id, 'LeBron James'))
        graph.add_node(JgfNode(node2_id, 'Los Angeles Lakers'))

        # Check source invalid
        with self.assertRaises(ValueError):
            graph.add_edge(JgfEdge(node1_id + '-nonsense', node2_id, 'Plays for'))
        
        # Check target invalid
        with self.assertRaises(ValueError):
            graph.add_edge(JgfEdge(node1_id, node2_id + '-nonsense', 'Plays for'))
            
        # Check both invalid
        with self.assertRaises(ValueError):
            graph.add_edge(JgfEdge(node1_id + '-nonsense', node2_id + '-nonsense', 'Plays for'))

    def test_add_edge_throws_if_mandatory_param_missing(self):
        """should throw error if mandatory parameter is missing"""
        graph = JgfGraph()
        
        # In Python, missing arguments to a constructor raise TypeError.
        # However, passing None explicitly (as in JS tests often implies) raises ValueError via Guards.
        with self.assertRaises(TypeError):
            graph.add_edge() # Missing arguments for add_edge? No, add_edge takes 1 arg (edge).
                             # But constructing the JgfEdge needs args.
        
        # The JS test was `graph.addEdge('source')` which implies passing bad data type.
        # Python's type hinting and runtime checks will catch this differently.
        # We will assume we are testing the JgfEdge constructor validation here mainly.
        
        with self.assertRaises(TypeError):
             # JgfEdge constructor requires source and target
             graph.add_edge(JgfEdge()) 

    # describe('#addEdges')
    def test_add_edges_does_not_call_add_edge_if_empty(self):
        """should not call addEdge if no edges are passed"""
        graph = JgfGraph()
        
        # Spy on the add_edge method
        with patch.object(graph, 'add_edge', wraps=graph.add_edge) as mock_add_edge:
            graph.add_edges([])
            mock_add_edge.assert_not_called()

    def test_add_edges_calls_add_edge_for_each(self):
        """should call addEdge for each edge"""
        graph = JgfGraph()
        
        # Create dummy nodes so add_edge validation passes
        graph.add_node(JgfNode('firstSource', 'l'))
        graph.add_node(JgfNode('targetOne', 'l'))
        graph.add_node(JgfNode('secondSource', 'l'))
        graph.add_node(JgfNode('targetTwo', 'l'))

        edge_one = JgfEdge('firstSource', 'targetOne', 'targetOne',)
        edge_two = JgfEdge('secondSource', 'targetTwo', 'secondRelation',)

        with patch.object(graph, 'add_edge', wraps=graph.add_edge) as mock_add_edge:
            graph.add_edges([edge_one, edge_two])
            
            self.assertEqual(mock_add_edge.call_count, 2)
            
            # Verify calls args. In Python mock, call_args_list is a list of calls.
            # call[0] are positional args, call[1] are kwargs.
            self.assertEqual(mock_add_edge.call_args_list[0][0][0], edge_one)
            self.assertEqual(mock_add_edge.call_args_list[1][0][0], edge_two)

    # describe('#removeEdge')
    def test_remove_edge(self):
        """should remove a graph edge"""
        graph = JgfGraph()
        node1_id = 'n1'
        node2_id = 'n2'
        relation = 'Plays for'

        graph.add_node(JgfNode(node1_id, 'L'))
        graph.add_node(JgfNode(node2_id, 'L'))
        graph.add_edge(JgfEdge(node1_id, node2_id, relation))

        graph.remove_edge(JgfEdge(node1_id, node2_id, relation))
        self.assertEqual(len(graph.edges), 0)

    def test_remove_edge_specific_relation(self):
        """should only remove the edge specified by relation parameter"""
        graph = JgfGraph()
        node1_id = 'n1'
        node2_id = 'n2'
        rel_play = 'Plays for'
        rel_pay = 'Gets his salary paid by'

        graph.add_node(JgfNode(node1_id, 'L'))
        graph.add_node(JgfNode(node2_id, 'L'))
        graph.add_edge(JgfEdge(node1_id, node2_id, rel_play))
        graph.add_edge(JgfEdge(node1_id, node2_id, rel_pay))
        
        self.assertEqual(len(graph.edges), 2)

        # Remove 'Plays for'
        graph.remove_edge(JgfEdge(node1_id, node2_id, rel_play))
        self.assertEqual(len(graph.edges), 1)
        self.assertEqual(graph.edges[0].relation, rel_pay)

        # Try to remove edge without specifying relation (None).
        # Since remove_edge uses strict equality (compare_relation=True),
        # And the remaining edge HAS a relation (rel_pay), 
        # JgfEdge(n1, n2, None) != JgfEdge(n1, n2, rel_pay).
        # So nothing should be removed.
        graph.remove_edge(JgfEdge(node1_id, node2_id))
        self.assertEqual(len(graph.edges), 1)

        # Remove 'Gets his salary paid by' explicitly
        graph.remove_edge(JgfEdge(node1_id, node2_id, rel_pay))
        self.assertEqual(len(graph.edges), 0)

    # describe('#getEdgesByNodes')
    def test_get_edges_by_nodes(self):
        """should lookup edges"""
        graph = JgfGraph()
        node1_id = 'n1'
        node2_id = 'n2'
        relation = 'Plays for'

        graph.add_node(JgfNode(node1_id, 'L'))
        graph.add_node(JgfNode(node2_id, 'L'))
        
        expected_edge = JgfEdge(node1_id, node2_id, relation)
        graph.add_edge(expected_edge)

        # We must reconstruct the edge because object identity might differ, 
        # but content equality should match.
        edges = graph.get_edges_by_nodes(node1_id, node2_id, relation)
        
        self.assertEqual(len(edges), 1)
        # Using custom equality check or asserting attributes
        self.assertTrue(edges[0].__eq__(expected_edge))

    def test_get_edges_by_nodes_throws_if_nodes_invalid(self):
        """should throw error if source or target node does not exist"""
        graph = JgfGraph()
        node1_id = 'n1'
        node2_id = 'n2'
        relation = 'Plays for'

        graph.add_node(JgfNode(node1_id, 'L'))
        graph.add_node(JgfNode(node2_id, 'L'))
        graph.add_edge(JgfEdge(node1_id, node2_id, relation))

        with self.assertRaises(ValueError):
            graph.get_edges_by_nodes('n1-nonsense', node2_id, relation)
        
        with self.assertRaises(ValueError):
            graph.get_edges_by_nodes(node1_id, 'n2-nonsense', relation)

    # describe('#mutators')
    def test_mutators_metadata(self):
        """should be able to set and get metadata"""
        graph = JgfGraph()
        self.assertIsNone(graph.metadata)

        metadata = {'bla': 'some-setting-metadata'}
        graph.metadata = metadata
        self.assertEqual(graph.metadata, metadata)

    # describe('#graphDimensions')
    def test_graph_dimensions_empty(self):
        """should return zero dimensions for an empty graph"""
        graph = JgfGraph()
        dims = graph.graph_dimensions
        self.assertEqual(dims['nodes'], 0)
        self.assertEqual(dims['edges'], 0)

    def test_graph_dimensions_non_empty(self):
        """should return valid dimensions for a non-empty graph"""
        graph = JgfGraph()
        graph.add_node(JgfNode('node1', 'A'))
        graph.add_node(JgfNode('node2', 'B'))
        graph.add_edge(JgfEdge('node1', 'node2', 'C'))

        dims = graph.graph_dimensions
        self.assertEqual(dims['nodes'], 2)
        self.assertEqual(dims['edges'], 1)

class TestMultiGraph(unittest.TestCase):

    # describe('#createMultiGraph')
    def test_create_empty_multi_graph(self):
        """should create empty multi graph"""
        # Python requires type and label args. 
        # Passing empty strings to simulate default creation.
        multi_graph = JgfMultiGraph(type='', label='')
        
        self.assertIsNotNone(multi_graph)

        # In Python, we verify the list is empty rather than checking for Null/None
        # because the constructor initializes self._graphs = []
        self.assertEqual(multi_graph.graphs, [])

    # describe('#addGraph')
    def test_add_graph(self):
        """should add a graph to the container"""
        multi_graph = JgfMultiGraph(type='', label='')

        # JgfGraph() is valid because its constructor has default arguments
        multi_graph.add_graph(JgfGraph())
        self.assertEqual(len(multi_graph.graphs), 1)

        multi_graph.add_graph(JgfGraph())
        self.assertEqual(len(multi_graph.graphs), 2)

    # describe('#mutators')
    def test_mutators_metadata(self):
        """should set and get metadata"""
        multi_graph = JgfMultiGraph(type='', label='')
        metadata = {'wegot': 'crass'}

        self.assertIsNone(multi_graph.metadata)

        multi_graph.metadata = metadata
        self.assertEqual(multi_graph.metadata, metadata)

class TestJsonDecorator(unittest.TestCase):
    maxDiff = None

    # describe('#to json')
    def test_to_json_throws_error_for_non_supported_objects(self):
        """should throw error for non supported objects"""
        # Note: We must instantiate valid objects (pass required args) 
        # to ensure the error comes from the decorator check, not the constructor.
        
        with self.assertRaises(TypeError):
            Jgf.to_json(JgfNode('id', 'label'))
            
        with self.assertRaises(TypeError):
            Jgf.to_json(JgfEdge('s', 't'))
            
        with self.assertRaises(TypeError):
            Jgf.to_json(Jgf())
            
        with self.assertRaises(TypeError):
            Jgf.to_json(object())
            
        with self.assertRaises(TypeError):
            Jgf.to_json(None)
            
        with self.assertRaises(TypeError):
            Jgf.to_json(2)
            
        with self.assertRaises(TypeError):
            Jgf.to_json('hello')

        Jgf.to_json(JgfGraph(), validate=True)
        # For MultiGraph, we need to pass mandatory constructor args if strict typing is enforced,
        # but empty strings are fine.
        Jgf.to_json(JgfMultiGraph(type='', label=''), validate=True)

    def test_transform_single_graph_to_json(self):
        """should transform single graph to json"""
        graph = JgfGraph('someType', 'someLabel', True, {'bla': 'some-meta-data'})

        graph.add_node(JgfNode('firstNodeId', 'blubb-label', {'bla': 'whoopp'}))
        graph.add_node(JgfNode('secondNodeId', 'bla-label', {'bli': 'whaaat'}))

        graph.add_edge(JgfEdge('firstNodeId', 'secondNodeId', 'is-test-edge'))

        expected_json = {
            'graph': {
                'type': 'someType',
                'label': 'someLabel',
                'directed': True,
                'metadata': {'bla': 'some-meta-data'},
                'nodes': {
                    'firstNodeId': {'label': 'blubb-label', 'metadata': {'bla': 'whoopp'}},
                    'secondNodeId': {'label': 'bla-label', 'metadata': {'bli': 'whaaat'}},
                },
                'edges': [
                    {
                        'source': 'firstNodeId',
                        'target': 'secondNodeId',
                        'relation': 'is-test-edge'
                    }
                ]
            }
        }

        self.assertEqual(Jgf.to_json(graph, validate=True), expected_json)

    # def test_transform_hypergraph_to_json(self):
    #     """should transform graph with hyperedges to json (V2 Specific)"""
    #     graph = JgfGraph('hyper', 'graph')
    #     graph.add_node(JgfNode('n1'))
    #     graph.add_node(JgfNode('n2'))
    #     graph.add_node(JgfNode('n3'))

    #     # Directed Hyperedge (n1 -> n2, n3)
    #     dhe = JgfDirectedHyperEdge(source=['n1'], target=['n2', 'n3'], relation='broadcast')
    #     graph.add_hyperedge(dhe)

    #     # Undirected Hyperedge (n1, n2, n3 connected together)
    #     uhe = JgfUndirectedHyperEdge(nodes=['n1', 'n2', 'n3'], relation='team')
    #     graph.add_hyperedge(uhe)

    #     json_out = JgfJsonDecorator.to_json(graph, validate=True)
        
    #     self.assertTrue('hyperedges' in json_out['graph'])
    #     self.assertEqual(len(json_out['graph']['hyperedges']), 2)
        
    #     # Verify structure of first hyperedge (Directed)
    #     he1 = json_out['graph']['hyperedges'][0]
    #     self.assertEqual(he1['relation'], 'broadcast')
    #     self.assertEqual(he1['source'], ['n1'])
    #     self.assertEqual(he1['target'], ['n2', 'n3'])

    #     # Verify structure of second hyperedge (Undirected)
    #     he2 = json_out['graph']['hyperedges'][1]
    #     self.assertEqual(he2['relation'], 'team')
    #     self.assertEqual(he2['nodes'], ['n1', 'n2', 'n3'])

    def test_transform_multigraph_to_json(self):
        """should transform multigraph to json"""
        multigraph = JgfMultiGraph('weird-multigraph', 'This is weird', {'weirdness': 100})

        # Graph 1
        graph1 = JgfGraph('someType', 'someLabel', True, {'bla': 'some-meta-data'})
        graph1.add_node(JgfNode('firstNodeId', 'blubb-label', {'bla': 'whoopp'}))
        graph1.add_node(JgfNode('secondNodeId', 'bla-label', {'bli': 'whaaat'}))
        graph1.add_edge(JgfEdge('firstNodeId', 'secondNodeId', 'is-test-edge'))
        
        multigraph.add_graph(graph1)

        # Graph 2
        graph2 = JgfGraph('otherType', 'otherLabel', False, {'ble': 'some-blumeta-data'})
        graph2.add_node(JgfNode('other-firstNodeId', 'effe-label', {'ufe': 'schnad'}))
        graph2.add_node(JgfNode('other-secondNodeId', 'uffe-label', {'bame': 'bral'}))
        graph2.add_edge(JgfEdge('other-firstNodeId', 'other-secondNodeId', 'is-ither-test-edge',))
        
        multigraph.add_graph(graph2)

        expected_json = {
            'type': 'weird-multigraph',
            'label': 'This is weird',
            'metadata': {'weirdness': 100},
            'graphs': [
                {
                    'type': "someType",
                    'label': "someLabel",
                    'directed': True,
                    'metadata': {'bla': "some-meta-data"},
                    'nodes': {
                        "firstNodeId": {'label': "blubb-label", 'metadata': {'bla': "whoopp"}},
                        "secondNodeId": {'label': "bla-label", 'metadata': {'bli': "whaaat"}},
                    },
                    'edges': [{
                        'source': "firstNodeId",
                        'target': "secondNodeId",
                        'relation': "is-test-edge",
                    }],
                },
                {
                    'type': "otherType",
                    'label': "otherLabel",
                    'directed': False,
                    'metadata': {'ble': "some-blumeta-data"},
                    'nodes': {
                        "other-firstNodeId": {'label': "effe-label", 'metadata': {'ufe': "schnad"}},
                        "other-secondNodeId": {'label': "uffe-label", 'metadata': {'bame': "bral"}},
                    },
                    'edges': [{
                        'source': "other-firstNodeId",
                        'target': "other-secondNodeId",
                        'relation': "is-ither-test-edge",
                    }],
                }
            ]
        }

        self.assertEqual(Jgf.to_json(multigraph, validate=True), expected_json)

    # describe('#from json')
    def test_transform_single_graph_json_to_jgf_graph(self):
        """should transform single graph json to JgfGraph"""
        json_data = {
            'graph': {
                'type': 'someType',
                'label': 'someLabel',
                'directed': True,
                'metadata': {'bla': 'some-meta-data'},
                'nodes': {
                    (id1:='firstNodeId'): {'label': 'blubb-label', 'metadata': {'bla': 'whoopp'}},
                    (id2:='secondNodeId'): {'label': 'bla-label', 'metadata': {'bli': 'whaaat'}},
                },
                'edges': [
                    {
                        'source': 'firstNodeId',
                        'target': 'secondNodeId',
                        'relation': 'is-test-edge',
                    }
                ]
            }
        }

        graph = Jgf.from_json(json_data, validate=True)

        self.assertIsInstance(graph, JgfGraph)
        self.assertEqual(graph.type, 'someType')
        self.assertEqual(graph.label, 'someLabel')
        self.assertEqual(graph.directed, True)
        self.assertEqual(graph.metadata, {'bla': "some-meta-data"})
        
        # Check Nodes Map size
        self.assertEqual(len(graph.nodes), 2)
        self.assertEqual(len(graph.edges), 1)

        # Check Nodes
        self.assertIsInstance(graph.nodes[id1], JgfNode)
        self.assertEqual(graph.nodes[id1].id, 'firstNodeId')
        self.assertEqual(graph.nodes[id1].label, 'blubb-label')
        self.assertEqual(graph.nodes[id1].metadata, {'bla': 'whoopp'})

        self.assertIsInstance(graph.nodes[id2], JgfNode)
        self.assertEqual(graph.nodes[id2].id, 'secondNodeId')
        self.assertEqual(graph.nodes[id2].label, 'bla-label')
        self.assertEqual(graph.nodes[id2].metadata, {'bli': 'whaaat'})

        # Check Edges
        self.assertIsInstance(graph.edges[0], JgfEdge)
        self.assertEqual(graph.edges[0].source, 'firstNodeId')
        self.assertEqual(graph.edges[0].target, 'secondNodeId')
        self.assertEqual(graph.edges[0].relation, 'is-test-edge')

    def test_transform_json_of_multigraph_to_jgf_multigraph(self):
        """should transform json of a multigraph to JgfMultiGraph"""
        json_data = {
            'type': 'weird-multigraph',
            'label': 'This is weird',
            'metadata': {'weirdness': 100},
            'graphs': [
                {
                    'type': "someType",
                    'label': "someLabel",
                    'directed': True,
                    'metadata': {'bla': "some-meta-data"},
                    'nodes': 
                        {
                            (id1:="firstNodeId"): {'label': "blubb-label", 'metadata': {'bla': "whoopp"}},
                            (id2:="secondNodeId"): {'label': "bla-label", 'metadata': {'bli': "whaaat"}},
                        },
                    'edges': [{
                        'source': "firstNodeId",
                        'target': "secondNodeId",
                        'relation': "is-test-edge",
                    }],
                },
                {
                    'type': "otherType",
                    'label': "otherLabel",
                    'directed': False,
                    'metadata': {'ble': "some-blumeta-data"},
                    'nodes': {
                        (g2id1:="other-firstNodeId"): {'label': "effe-label", 'metadata': {'ufe': "schnad"}},
                        (g2id2:="other-secondNodeId"): {'label': "uffe-label", 'metadata': {'bame': "bral"}},
                    },
                    'edges': [{
                        'source': "other-firstNodeId",
                        'target': "other-secondNodeId",
                        'relation': "is-ither-test-edge",
                    }],
                }
            ]
        }

        multigraph = Jgf.from_json(json_data, validate=True)

        self.assertIsInstance(multigraph, JgfMultiGraph)
        self.assertEqual(len(multigraph.graphs), 2)
        self.assertEqual(multigraph.type, 'weird-multigraph')
        self.assertEqual(multigraph.label, 'This is weird')
        self.assertEqual(multigraph.metadata, {'weirdness': 100})

        # Check First Graph
        graph1 = multigraph.graphs[0]
        self.assertIsInstance(graph1, JgfGraph)
        self.assertEqual(graph1.type, 'someType')
        self.assertEqual(graph1.label, 'someLabel')
        self.assertEqual(graph1.metadata, {'bla': "some-meta-data"})
        self.assertEqual(graph1.directed, True)
        self.assertEqual(len(graph1.nodes), 2)
        self.assertEqual(len(graph1.edges), 1)

        self.assertIsInstance(graph1.nodes[id1], JgfNode)
        self.assertEqual(graph1.nodes[id1].id, 'firstNodeId')
        self.assertEqual(graph1.nodes[id1].label, 'blubb-label')
        self.assertEqual(graph1.nodes[id1].metadata, {'bla': 'whoopp'})

        self.assertIsInstance(graph1.edges[0], JgfEdge)
        self.assertEqual(graph1.edges[0].source, 'firstNodeId')
        self.assertEqual(graph1.edges[0].target, 'secondNodeId')
        self.assertEqual(graph1.edges[0].relation, 'is-test-edge')

        # Check Second Graph
        graph2 = multigraph.graphs[1]
        self.assertIsInstance(graph2, JgfGraph)
        self.assertEqual(graph2.type, 'otherType')
        self.assertEqual(graph2.label, 'otherLabel')
        self.assertEqual(graph2.directed, False)
        self.assertEqual(graph2.metadata, {'ble': "some-blumeta-data"})
        self.assertEqual(len(graph2.nodes), 2)
        self.assertEqual(len(graph2.edges), 1)

        self.assertIsInstance(graph2.nodes[g2id1], JgfNode)
        self.assertEqual(graph2.nodes[g2id1].id, 'other-firstNodeId')
        self.assertEqual(graph2.nodes[g2id1].label, 'effe-label')
        self.assertEqual(graph2.nodes[g2id1].metadata, {'ufe': "schnad"})

        self.assertIsInstance(graph2.edges[0], JgfEdge)
        self.assertEqual(graph2.edges[0].source, 'other-firstNodeId')
        self.assertEqual(graph2.edges[0].target, 'other-secondNodeId')
        self.assertEqual(graph2.edges[0].relation, 'is-ither-test-edge')

    def test_from_json_throws_error_for_invalid_json(self):
        """should throw error for json that does not have a graph or graphs property"""
        invalid_val = [
            {}, 
            {'graphiti': 'graphiti'}, 
            {'more': 'nonsense'}
        ]

        for invalid in invalid_val:
            with self.assertRaises(ValueError):
                Jgf.from_json(invalid)
        
        invalid_typ = [
             'string-metadata', 
            2, 
            ['bla'], 
        ]

        for invalid in invalid_typ:
            with self.assertRaises(TypeError):
                Jgf.from_json(invalid)

    def test_graph_generation_deep(self):
        # this basically test it doesn't hang
        obj = make_chain_graph_deep(500, 100)
        assert (data:=Jgf.to_json(obj, validate=True))
        assert Jgf.from_json(data, validate=True)


    def test_to_from_json_worst_case(self):
        import time

        sizes = [10, 25, 50, 100, 200, 500, 1000]
        num_fields = 20

        ecode_times = []
        decode_times = []

        for n in sizes:
            root = make_chain_graph_deep(n, num_fields)

            start = time.perf_counter()
            encoded = Jgf.to_json(root, validate=True)
            ecode_times.append(time.perf_counter() - start)

            start = time.perf_counter()
            Jgf.from_json(encoded, validate=True)
            decode_times.append(time.perf_counter() - start)

        for n, t in zip(sizes, ecode_times):
            print(f"N={n:<4} F={num_fields:<3} encode={t:.6f}s")
        ratios = [ecode_times[i+1] / ecode_times[i] for i in range(len(ecode_times) - 1)]
        # Allow noise, but fail on quadratic explosion
        assert max(ratios) < 7, 'encode time appears super-linear; type propagation may be degrading worse than expected.'

        for n, t in zip(sizes, decode_times):
            print(f"N={n:<4} F={num_fields:<3} decode={t:.6f}s")
        ratios = [decode_times[i+1] / decode_times[i] for i in range(len(decode_times) - 1)]
        # Allow noise, but fail on quadratic explosion
        assert max(ratios) < 7, 'decode time appears super-linear; type propagation may be degrading worse than expected.'


def make_chain_graph_deep(size: int, num_fields: int) -> JgfGraph:
    """
    Create a worst-case chain graph using JgfGraph with:
      - `size` nodes
      - `num_fields` edges per node pair
    Each edge represents a 'field' pointing to the *next* node in the chain.

    This stresses:
      - JGF edge list parsing
      - Node lookup performance (by ID)
      - Handling of multiple edges between the same two nodes (multigraph behavior)
    """

    # 1. Initialize the Graph
    graph = JgfGraph(
        type="chain-graph-deep", 
        label=f"Deep Chain (size={size}, fields={num_fields})"
    )

    # 2. Create and add all nodes
    # We use simple string integers for IDs: "0", "1", "2", ...
    for i in range(size):
        graph.add_node(JgfNode(id=str(i), label=f"Node {i}"))

    # 3. Build chain connections
    # For every node i (except the last one), create 'num_fields' edges to node i+1
    for i in range(size - 1):
        source_id = str(i)
        target_id = str(i + 1)
        
        for j in range(num_fields):
            # We map the concept of "child_j" field to the Edge Relation
            edge = JgfEdge(
                source=source_id,
                target=target_id,
                relation=f"child_{j}",
            )
            graph.add_edge(edge)

    return graph

if __name__ == '__main__':
    unittest.main()
import unittest
from datetime import datetime

from jgf import JgfNode, JgfEdge, JgfGraph, Jgf
from gson import  Encoder, Decoder

class TestGJSON(unittest.TestCase):

    def setUp(self):
        self.encoder = Encoder()
        self.decoder = Decoder()

    def test_primitives(self):
        """Test strict equality for primitive types."""
        test_cases = [
            1, 
            1.5, 
            "Hello World", 
            True, 
            False, 
            None
        ]
        
        for original in test_cases:
            with self.subTest(val=original):
                graph = self.encoder.encode(original)
                decoded = self.decoder.decode(graph)
                self.assertEqual(original, decoded)
                self.assertEqual(type(original), type(decoded))

    def test_simple_containers(self):
        """Test basic lists and dictionaries."""
        # List
        original_list = [1, 2, 3]
        graph_list = self.encoder.encode(original_list)
        decoded_list = self.decoder.decode(graph_list)
        self.assertEqual(original_list, decoded_list)

        # Dict
        original_dict = {"a": 1, "b": 2}
        graph_dict = self.encoder.encode(original_dict)
        decoded_dict = self.decoder.decode(graph_dict)
        self.assertEqual(original_dict, decoded_dict)

        original_dict = {None: None, "b": None, 3: None}
        graph_dict = self.encoder.encode(original_dict)
        decoded_dict = self.decoder.decode(graph_dict)
        self.assertEqual(repr(original_dict), repr(decoded_dict))

    def test_nested_structures(self):
        """Test a mix of lists and dicts."""
        data = {
            "users": [
                {"id": 1, "name": "Alice"},
                {"id": 2, "name": "Bob"}
            ],
            "meta": {"page": 1}
        }
        graph = self.encoder.encode(data)
        decoded = self.decoder.decode(graph)
        self.assertEqual(data, decoded)

    def test_object_identity(self):
        """
        Ensure that if an object appears twice in the input.
        """
        shared_obj = {"id": "shared"}
        # 'shared_obj' appears twice
        data = [shared_obj, shared_obj]
        
        graph = self.encoder.encode(data)
        decoded = self.decoder.decode(graph)
        
        # Value equality
        self.assertEqual(decoded, [{"id": "shared"}, {"id": "shared"}])
        
        # Identity equality (The core requirement)
        self.assertIs(decoded[0], decoded[1], "Decoder created two separate objects instead of preserving identity")

    def test_cyclical_reference_self(self):
        """Test a list that contains itself (Infinite recursion if not handled)."""
        data = []
        data.append(data)  # Cycle: data -> data
        
        graph = self.encoder.encode(data)
        decoded = self.decoder.decode(graph)
        
        self.assertIsInstance(decoded, list)
        self.assertEqual(len(decoded), 1)
        # The item inside the list should be the list itself
        self.assertIs(decoded[0], decoded)

    def test_cyclical_reference_indirect(self):
        """Test A -> B -> A cycle."""
        a = {}
        b = {}
        a["next"] = b
        b["next"] = a
        
        graph = self.encoder.encode(a)
        decoded_a = self.decoder.decode(graph)
        
        decoded_b = decoded_a["next"]
        
        self.assertIsInstance(decoded_a, dict)
        self.assertIs(decoded_b["next"], decoded_a)

    def test_custom_type_registry(self):
        """Test adding support for a custom type (datetime) via registry."""
        
        # 1. Define Custom Handler for Encoder
        def encode_dt(obj, node_id, encoder):
            meta = {"type": "datetime", "value": obj.timestamp()}
            encoder.graph.add_node(JgfNode(id=node_id, label=str(obj), metadata=meta))

        # 2. Define Custom Handlers for Decoder
        def create_dt(node):
            return datetime.fromtimestamp(node.metadata["value"])

        # 3. Register them
        self.encoder.register(datetime, encode_dt)

        # datetime is an immutable leaf object, nothing to fill via edges, so we can use the regular register()
        self.decoder.register("datetime", create_dt)

        # 4. Test
        now = datetime.now()
        graph = self.encoder.encode(now)
        decoded = self.decoder.decode(graph)
        
        self.assertEqual(now, decoded)
        self.assertIsInstance(decoded, datetime)

    def test_json_interop(self):
        """Verify the JGF Utils can actually serialize the graph produced."""
        data = [1, 2, 3]
        graph = self.encoder.encode(data)
        
        # Convert to raw JSON dict
        json_output = Jgf.to_json(graph)
        
        self.assertIsInstance(json_output, dict)
        self.assertIn("graph", json_output)
        self.assertTrue(len(json_output["graph"]["nodes"]) > 0)
        
        # Convert back
        graph_restored = Jgf.from_json(json_output)
        decoded = self.decoder.decode(graph_restored)
        
        self.assertEqual(data, decoded)
    
    def test_cyclic_list_and_tuple(self):
        """
        A cyclic structure composed of lists and tuples.
        """
        # 1. Construct the cycle
        l = []
        t = (l, 100)
        l.append(t)

        # 2. Encode
        graph = self.encoder.encode(l)
        
        # 3. Decode
        decoded_l = self.decoder.decode(graph)

        # 4. Assertions
        self.assertIsInstance(decoded_l, list)
        self.assertEqual(len(decoded_l), 1)
        
        # Extract the tuple inside the list
        decoded_t = decoded_l[0]
        self.assertIsInstance(decoded_t, tuple)
        self.assertEqual(len(decoded_t), 2)
        
        # Check values
        self.assertEqual(decoded_t[1], 100)
        
        # CYCLE CHECK: The first element of the tuple must be the list itself
        self.assertIs(decoded_t[0], decoded_l, "Cycle broken: Tuple does not point back to the parent list object")

    def test_dict_tuple_keys_and_values(self):
        """
        A dict that has tuples of primitives as keys and other tuples as values.
        """
        # 1. Construct data
        # Key: (1, 2), Value: (3, 4)
        original_data = {
            (1, 2): (3, 4),
            ("a", "b"): (True, False)
        }

        # 2. Encode
        graph = self.encoder.encode(original_data)

        # 3. Decode
        decoded_data = self.decoder.decode(graph)

        # 4. Assertions
        self.assertIsInstance(decoded_data, dict)
        self.assertEqual(len(decoded_data), 2)
        
        # Check specific key lookup (requires hashing to work correctly)
        self.assertIn((1, 2), decoded_data)
        self.assertEqual(decoded_data[(1, 2)], (3, 4))
        self.assertEqual(decoded_data[("a", "b")], (True, False))

    def test_dict_nested_tuple_keys(self):
        """
        A dict that has tuples of tuples and primitives as keys.
        """
        # 1. Construct data
        complex_key = ((10, 20), "end")
        original_data = {
            complex_key: "Success"
        }

        # 2. Encode
        graph = self.encoder.encode(original_data)

        # 3. Decode
        decoded_data = self.decoder.decode(graph)

        # 4. Assertions
        # Verify structure
        self.assertIn(complex_key, decoded_data)
        self.assertEqual(decoded_data[complex_key], "Success")

        # Verify deep types
        reconstructed_key = list(decoded_data.keys())[0]
        self.assertIsInstance(reconstructed_key, tuple)          # Outer
        self.assertIsInstance(reconstructed_key[0], tuple)       # Inner
        self.assertEqual(reconstructed_key[0][1], 20)

# Increase recursion limit for deep nesting tests
import sys, time
sys.setrecursionlimit(20000)

class TestPerformance(unittest.TestCase):

    def generate_structure(self, depth: int, width: int, current_depth: int = 0):
        """
        Generates a nested list structure.
        Total nodes approx = width * depth
        """
        if current_depth >= depth:
            return "leaf"
        
        # Create a list with 'width' items
        # The last item is the recursive step to ensure nesting
        level = [f"data_{current_depth}_{i}" for i in range(width - 1)]
        level.append(self.generate_structure(depth, width, current_depth + 1))
        
        return level

    def count_nodes(self, graph) -> int:
        return len(graph.nodes)

    def measure_phase(self, name: str, func, *args):
        start = time.perf_counter()
        result = func(*args)
        end = time.perf_counter()
        return result, end - start

    def test_linearity(self):
        print(f"\n{'='*60}")
        print(f"{'Performance Linearity Test':^60}")
        print(f"{'='*60}")
        print(f"{'Nodes':<10} | {'Enc Time':<10} | {'Enc/Node':<10} | {'Dec Time':<10} | {'Dec/Node':<10}")
        print("-" * 60)

        encoder = Encoder()
        decoder = Decoder()
        
        # We will double the size (depth) in each step to see if time doubles (Linear)
        # or quadruples (Quadratic)
        base_width = 90
        depths = [500, 1000, 2000, 4000] 
        
        results = []

        for depth in depths:
            # 1. Generate Input
            data = self.generate_structure(depth=depth, width=base_width)
            
            # 2. Measure Encoding
            graph, t_enc = self.measure_phase("Encode", encoder.encode, data)
            n_nodes = self.count_nodes(graph)
            
            # 3. Measure Decoding
            decoded, t_dec = self.measure_phase("Decode", decoder.decode, graph)
            
            # 4. Calculate stats
            enc_per_node = t_enc / n_nodes
            dec_per_node = t_dec / n_nodes
            
            print(f"{n_nodes:<10} | {t_enc:.4f}s    | {enc_per_node:.2e}   | {t_dec:.4f}s    | {dec_per_node:.2e}")
            
            results.append({
                "n": n_nodes,
                "enc_per_node": enc_per_node,
                "dec_per_node": dec_per_node
            })

        print("-" * 60)

        # ASSERTIONS
        # We verify that the time-per-node does not grow significantly.
        # If it was O(N^2), the time-per-node for 4000 would be ~8x higher than for 500.
        # In O(N), it should be roughly the same (allowing for GC noise).
        
        first_run = results[0]
        last_run = results[-1]
        
        # We allow a small margin of growth (e.g. 2x) for overhead/memory pressure, 
        # but certainly not the linear growth factor (last_n / first_n = 8x) that O(N^2) would imply.
        
        growth_factor_enc = last_run["enc_per_node"] / first_run["enc_per_node"]
        growth_factor_dec = last_run["dec_per_node"] / first_run["dec_per_node"]
        
        print(f"Growth Factor (Enc): {growth_factor_enc:.2f}x")
        print(f"Growth Factor (Dec): {growth_factor_dec:.2f}x")

        # Threshold: If time-per-node grew by more than 2x while input grew 8x, 
        # something is likely inefficient (non-linear).
        self.assertLess(growth_factor_enc, 2, "Encoding time complexity appears non-linear")
        self.assertLess(growth_factor_dec, 2, "Decoding time complexity appears non-linear")


class TestUnsupportedAndInvalid(unittest.TestCase):
    def setUp(self):
        self.encoder = Encoder()
        self.decoder = Decoder()
    
    def test_unsupported_object_crash(self):
        """
        Scenario: Encoding or Decoding an unregistered type.
        Expected: The system should strictly RAISE an error (ValueError or TypeError) 
        instead of guessing or falling back to string representation.
        """
        class UnknownAlien:
            pass
        
        # 1. Test Encoding Crash
        # The encoder encountering an unknown Python type should fail immediately.
        with self.assertRaises(ValueError):
            assert Jgf.to_json(self.encoder.encode(UnknownAlien())) == {}

        # 2. Test Decoding Crash
        # The decoder encountering a node with an unknown "type" metadata should fail.
        graph = JgfGraph()
        graph.add_node(JgfNode(
            id="n1", 
            label="?", 
            metadata={"type": "UnknownAlien", "value": "check"}
        ))
        
        with self.assertRaises(ValueError):
            self.decoder.decode(graph)

    def test_decode_missing_type_metadata(self):
        """
        Scenario: Decoding a graph where a node is corrupted and missing 'type' metadata.
        Expected: Should default to extracting 'value' or None, but not crash.
        """
        graph = JgfGraph()
        # Node missing "type" in metadata
        graph.add_node(JgfNode(id="n1", label="broken", metadata={"some_other_key": 1}))
        
        with self.assertRaises(ValueError):
            self.decoder.decode(graph)

    def test_impossible_recursive_tuple(self):
        """
        Scenario: A graph describes a tuple containing ITSELF: t = (t,)
        Context: This is impossible in Python (tuples are immutable, so they cannot reference 
        themselves during creation).
        Expected: RecursionError. The 'Materialize on Demand' strategy will loop infinitely 
        trying to build the child before the parent.
        """
        graph = JgfGraph()
        graph.add_node(JgfNode(id="t1", label="tuple", metadata={"type": "tuple"}))
        # Edge pointing back to itself
        graph.add_edge(JgfEdge(source="t1", target="t1", relation="list/item", metadata={"index": 0}))

        with self.assertRaises(RecursionError):
            self.decoder.decode(graph)

    def test_unhashable_tuple_as_dict_key(self):
        """
        Scenario: Decoding a dictionary where a Key is a Tuple containing a List.
        Context: In Python, `d = { ([],): 1 }` raises TypeError because lists are unhashable.
        Expected: The Decoder should successfully build the structure but raise TypeError 
        when finally attempting to insert the key into the dict.
        """
        # Build Graph: Dict -> Key(Tuple) -> Item(List)
        graph = JgfGraph()
        
        # Nodes
        graph.add_node(JgfNode(id="dict1", label="dict", metadata={"type": "dict"}))
        graph.add_node(JgfNode(id="tuple1", label="tuple", metadata={"type": "tuple"}))
        graph.add_node(JgfNode(id="list1", label="list", metadata={"type": "list"}))
        
        # Edges
        # Tuple contains List
        graph.add_edge(JgfEdge(source="tuple1", target="list1", relation="list/item", metadata={"index": 0}))
        # Dict uses Tuple as Key
        graph.add_edge(JgfEdge(source="dict1", target="tuple1", relation="dict/key", metadata={"index": 0}))
        # Dict uses "val" as Value
        val_node = JgfNode(id="v1", label="val", metadata={"type": "str", "value": "val"})
        graph.add_node(val_node)
        graph.add_edge(JgfEdge(source="dict1", target="v1", relation="dict/value", metadata={"index": 0}))

        # The decode call should crash when executing `obj[key_obj] = ...`
        with self.assertRaises(TypeError):
            self.decoder.decode(graph)

    def test_corrupted_graph_missing_edge_target(self):
        """
        Scenario: An edge points to a node ID that does not exist in the nodes list.
        Expected: ValueError during decode traversal.
        """
        graph = JgfGraph()
        graph.add_node(JgfNode(id="list1", label="list", metadata={"type": "list"}))
        graph.add_node(JgfNode(id="ghost_node", label="list", metadata={"type": "list"}))
        graph.add_edge(JgfEdge(source="list1", target="ghost_node", relation="list/item", metadata={"index": 0}))
        graph.remove_node("ghost_node")  # Simulate corruption by removing the target node

        with self.assertRaises(ValueError):
            self.decoder.decode(graph)

if __name__ == '__main__':
    unittest.main()

if __name__ == '__main__':
    unittest.main()
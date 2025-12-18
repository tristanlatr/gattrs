"""
Serialization for Python Objects with Cycle Support.
"""
# TODOS:
# - Do not store the index of dict keys/values, no I can;t because the index match each key/value pair together. 
#   We could instead use a DictItem node that hase two edges, one to key and one to value...
# - [DONE] Do not store the type of primitive types in metadata, rely simply json value.
# - [DONE] Optimize the leaf primitive types so that several nodes refers to the same primitive value
# - [DONE] Use several stegries to generete IDs based on provided Encoder argument, a simple counter would do probbaly.
    

from __future__ import annotations

import uuid
from typing import Any, Dict, Iterator, List, Callable, Type, Tuple
from collections import defaultdict

from jgf import Jgf, JgfNode, JgfEdge, JgfGraph

def _is_primitive(obj: Any) -> bool:
    """Check if the object is an immutable primitive type."""
    return isinstance(obj, (str, int, float, bool, type(None))) 

def _gen_uuid() -> Iterator[str]:
    """Generate unique random identifier."""
    while True:
        yield str(uuid.uuid4())

def _gen_inc() -> Iterator[str]:
    """Generate incrementing identifier based on char(), using more digits when needed."""
    i = 1
    # The limit is the max unicode code point (0x10ffff) + 1 for the zero-th index.
    base = 0x110000 
    
    while True:
        # Convert integer 'i' to a "base-1114112" string
        temp_i = i
        digits = []
        
        while True:
            temp_i, remainder = divmod(temp_i, base)
            digits.append(chr(remainder))
            if temp_i == 0:
                break
        
        # digits are collected in reverse order (least significant first)
        yield "".join(reversed(digits))
        i += 1

class Encoder:
    """
    Encodes Python objects into a JGF Graph with support for cyclical references.
    """
    visited: Dict[int, str]  # Map id(obj) -> node_id
    graph: JgfGraph
    
    def __init__(self, id_generator: Callable[[], Iterator[str]] = _gen_inc):
        
        # Registry: Type -> Encoder Function
        # Signature: (obj, node_id) -> None
        self._registry: Dict[Type, Callable[[Any, str, Encoder], None]] = {
            str: self._encode_primitive,
            int: self._encode_primitive,
            float: self._encode_primitive,
            bool: self._encode_primitive,
            type(None): self._encode_primitive,
            list: self._encode_list,
            tuple: self._encode_tuple,  # Built-in Tuple support
            dict: self._encode_dict,
        }

        self._id_generator = id_generator()

    def register(self, type_class: Type, handler: Callable[[Any, str, Encoder], None]):
        """Register a custom encoder for a specific type."""
        self._registry[type_class] = handler

    def encode(self, obj: Any) -> JgfGraph:
        """Main entry point."""
        self.visited = {}
        self.graph = JgfGraph(type="gson")
        
        # Start recursion
        self._process_node(obj)
        
        return self.graph

    def _process_node(self, obj: Any) -> str:
        """
        Recursive function to handle object identity and dispatch encoding.
        Returns the node_id.
        """
        obj_id = id(obj) if not _is_primitive(obj) else hash(obj)
        
        # 1. Check Identity (Cycle detection)
        if obj_id in self.visited:
            return self.visited[obj_id]

        # 2. Generate ID and mark visited
        node_id = next(self._id_generator)
        self.visited[obj_id] = node_id

        # 3. Dispatch to specific handler
        obj_type = type(obj)
        handler = self._registry.get(obj_type)
        
        if not handler:
            raise ValueError(f"No encoder registered for type: {obj_type.__name__}")
            # Fallback for subclasses or unknown types (treat as primitive string)
            # handler = self._encode_primitive

        handler(obj, node_id, self)
        return node_id

    # --- Built-in Handlers ---

    @staticmethod
    def _encode_primitive(obj: Any, node_id: str, encoder: Encoder):
        meta = {"value": obj}
        encoder.graph.add_node(JgfNode(id=node_id, metadata=meta))

    @staticmethod
    def _encode_list(obj: List, node_id: str, encoder: Encoder):
        encoder.graph.add_node(JgfNode(id=node_id, metadata={"type": "list"}))
        encoder._encode_sequence_items(obj, node_id, encoder)

    @staticmethod
    def _encode_tuple(obj: Tuple, node_id: str, encoder: Encoder):
        encoder.graph.add_node(JgfNode(id=node_id, metadata={"type": "tuple"}))
        encoder._encode_sequence_items(obj, node_id, encoder)    

    @staticmethod
    def _encode_sequence_items(obj: List | Tuple, node_id: str, encoder: Encoder):
        """Helper to link items for both lists and tuples."""
        for i, item in enumerate(obj):
            target_id = encoder._process_node(item)
            edge = JgfEdge(
                source=node_id,
                target=target_id,
                relation="list/item",
                metadata={"index": i}
            )
            encoder.graph.add_edge(edge)

    @staticmethod
    def _encode_dict(obj: Dict, node_id: str, encoder: Encoder):
        encoder.graph.add_node(JgfNode(id=node_id, metadata={"type": "dict"}))
        
        # Enumerate to give edges a stable index pairs
        for i, (key, value) in enumerate(obj.items()):
            key_id = encoder._process_node(key)
            val_id = encoder._process_node(value)
            
            # Edge to Key
            encoder.graph.add_edge(JgfEdge(
                source=node_id, target=key_id, 
                relation="dict/key", metadata={"index": i}
            ))
            
            # Edge to Value
            encoder.graph.add_edge(JgfEdge(
                source=node_id, target=val_id, 
                relation="dict/value", metadata={"index": i}
            ))

def _has_value(node: JgfNode) -> bool:
    """Helper to check if node metadata has a 'value' field."""
    return "value" in (node.metadata or {})

def _type_of(node: JgfNode) -> str | None:
    """Helper to extract type from node metadata, 
    return None if the type should be the same as the value."""
    try:
        return (node.metadata or {})["type"]
    except KeyError:
        if _has_value(node):
            return None
        raise ValueError(f"Node {node.id} missing 'type' metadata.")    

def _value_of(node: JgfNode) -> Any:
    """Helper to extract primitive value from node metadata."""
    try:
        return (node.metadata or {})["value"]
    except KeyError:
        raise ValueError(f"Node {node.id} missing 'value' metadata.")

def _index_of(edge: JgfEdge) -> int:
    """Helper to extract index from edge metadata."""
    try:
        return (edge.metadata or {})["index"]
    except KeyError:
        raise ValueError(f"Edge from {edge.source} to {edge.target} missing 'index' metadata.")

class Decoder:
    """
    Decodes a JGF Graph back into Python objects.
    Optimized to strictly avoid O(N*E) complexity.
    
    Supports two categories of types:
    1. Mutable (Shell Strategy): Created empty in Pass 1, filled in Pass 2. 
       (e.g., list, dict)
    2. Immutable (Materialize Strategy): Created Just-In-Time via recursion. 
       (e.g., tuple, frozenset). This allows them to be hashable dict keys.
    """

    object_map: Dict[str, Any]
    edges_by_source: Dict[str, List[JgfEdge]]
    
    def __init__(self):
        
        # Registry for Mutable Types (Shell Strategy)
        # Type Name -> (Shell Creator, Content Filler)
        self._mutable_registry: Dict[str | None, Tuple[Callable[[JgfNode], Any], Callable[[Any, List[JgfEdge], Decoder], None] | None]] = {
            "list": (lambda n: [], self._fill_list),
            "dict": (lambda n: {}, self._fill_dict),

            # Primitives are technically immutable but leaf nodes, so we map them directly
            None: (lambda n: _value_of(n), None),
            # "int": (lambda n: _value_of(n), None),
            # "float": (lambda n: _value_of(n), None),
            # "str": (lambda n: _value_of(n), None),
            # "bool": (lambda n: _value_of(n), None),
            # "NoneType": (lambda n: None, None),
        }

        # Registry for Immutable Containers Types (Materialize Strategy)
        # Type Name -> Materializer Function
        # Signature: (node_id, edges, decoder_instance) -> immutable_obj
        self._immutable_registry: Dict[str, Callable[[str, List[JgfEdge], Decoder], Any]] = {
            "tuple": self._materialize_tuple
        }

    def register(self, type_name: str, creator: Callable[[JgfNode], Any], 
                         filler: Callable[[Any, List[JgfEdge], Decoder], None] | None = None):
        """
        Register a custom type (e.g datetime).

        For container types (e.g. a deque), both creator and filler should be provided.
        For leaf types, only creator is needed and filler can be None.
        """
        self._mutable_registry[type_name] = (creator, filler)

    def register_immutable(self, type_name: str, materializer: Callable[[str, List[JgfEdge], Decoder], Any]):
        """
        Register a custom immutable container type (e.g. frozenset).
        """
        self._immutable_registry[type_name] = materializer

    def decode(self, graph: JgfGraph) -> Any:
        self.graph = graph
        self.object_map = {}
        self.edges_by_source = defaultdict(list)

        # Optimization: Pre-index edges by source for O(1) lookup
        for edge in graph.edges:
            self.edges_by_source[edge.source].append(edge)

        if not graph.nodes:
            return None
        
        # Determine Root Node (assumed to be the first added node)
        # This might need to be reconsidered!
        root_id = list(graph.nodes.keys())[0]

        # Pass 1: Instantiate Mutable Shells ONLY
        # We consciously skip Immutable container types here.
        for node_id, node in graph.nodes.items():
            type_name = _type_of(node)
            
            if type_name in self._mutable_registry:
                creator, _ = self._mutable_registry[type_name]
                self.object_map[node_id] = creator(node)

        # Pass 2: Fill Mutable Contents
        # Immutable types are not filled here; they are created on-demand when referenced.
        for node_id, node in graph.nodes.items():
            self._fill_mutable_content(node_id, node, graph)

        # Finally, return the root. 
        # If the root is immutable (e.g. the whole data is just one tuple), 
        # it won't be in object_map yet, so we request it via _get_or_create.
        return self._get_or_create(root_id, graph)

    def _get_or_create(self, node_id: str, graph: JgfGraph) -> Any:
        """
        The core accessor. 
        - If object exists (Mutable Shell or Primitive), returns it.
        - If missing, assumes it is Immutable and attempts to materialize it recursively.
        """
        if node_id in self.object_map:
            return self.object_map[node_id]
        try:
            node = graph.nodes[node_id]
        except KeyError:
            raise ValueError(f"Node {node_id} not found in graph.")
        type_name = _type_of(node)

        if type_name in self._immutable_registry:
            materializer = self._immutable_registry[type_name]
            edges = self.edges_by_source[node_id]
            
            # Materialize
            obj = materializer(node_id, edges, self)
            
            # Cache it (Memoization)
            self.object_map[node_id] = obj
            return obj
        
        # Should not happen if graph is valid
        raise ValueError(f"Node {node_id} (type: {type_name}) not found in map and no immutable handler registered.")

    def _fill_mutable_content(self, node_id: str, node: JgfNode, graph: JgfGraph):
        type_name = _type_of(node)
        
        if type_name not in self._mutable_registry:
            return # an immutable container type

        _, filler = self._mutable_registry[type_name]
        if filler is None:
            return  # No filler defined (e.g., for leaf objects)
            # we could check if type_name is None upthere but whatever this does.

        obj = self.object_map[node_id]
        edges = self.edges_by_source[node_id]
        
        filler(obj, edges, self)

    # --- Built-in Fillers (Mutable) ---

    @staticmethod
    def _fill_list(obj: List, edges: List[JgfEdge], decoder: Decoder):
        items = [e for e in edges if e.relation == "list/item"]
        items.sort(key=lambda e: _index_of(e))
        
        for edge in items:
            # Use _get_or_create to handle both mutable references and on-demand immutables
            child_obj = decoder._get_or_create(edge.target, decoder.graph)
            obj.append(child_obj)

    @staticmethod
    def _fill_dict(obj: Dict, edges: List[JgfEdge], decoder: Decoder):
        # We need to pair keys and values by their index
        keys = {}
        values = {}

        for edge in edges:
            idx = _index_of(edge)
            # This call allows Tuples to be materialized and returned as hashable keys
            target_obj = decoder._get_or_create(edge.target, decoder.graph)
            
            if edge.relation == "dict/key":
                keys[idx] = target_obj
            elif edge.relation == "dict/value":
                values[idx] = target_obj

        for idx, key_obj in keys.items():
            if idx in values:
                obj[key_obj] = values[idx]

    # --- Built-in Materializers (Immutable) ---

    @staticmethod
    def _materialize_tuple(node_id: str, edges: List[JgfEdge], decoder: Decoder) -> tuple:
        items = [e for e in edges if e.relation == "list/item"]
        items.sort(key=lambda e: _index_of(e))
        
        # Recursively resolve contents
        resolved_items = [
            decoder._get_or_create(edge.target, decoder.graph) 
            for edge in items
        ]
        
        return tuple(resolved_items)

if __name__ == "__main__":
    import sys, json
    from pprint import pprint
    encoder = Encoder()
    g = encoder.encode(json.load(sys.stdin))
    dat = Jgf.to_json(g, validate=True)
    pprint(dat)
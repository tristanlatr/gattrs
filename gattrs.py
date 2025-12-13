from __future__ import annotations

from typing import Any, Dict, List, Tuple, Optional, get_origin, get_args, get_type_hints
import attrs
import json
import uuid
import pathlib
import array
import enum
from collections import defaultdict, OrderedDict, deque
from types import MappingProxyType, SimpleNamespace
from decimal import Decimal
from fractions import Fraction
from datetime import datetime, timedelta


def _is_attrs_instance(obj: Any) -> bool:
    return hasattr(obj.__class__, "__attrs_attrs__")


def _format_expected(t):
    from typing import get_origin, get_args, Union
    import types as _types
    UnionType = getattr(_types, "UnionType", None)
    origin = get_origin(t)
    if origin is None:
        return getattr(t, "__name__", str(t))
    args = get_args(t)
    if origin is list:
        return f"list[{_format_expected(args[0])}]"
    if origin is dict:
        return f"dict[{_format_expected(args[0])}, {_format_expected(args[1])}]"
    if origin is tuple:
        return f"tuple[{_format_expected(args[0])}]"
    if origin is set:
        return f"set[{_format_expected(args[0])}]"
    if origin is frozenset:
        return f"frozenset[{_format_expected(args[0])}]"
    if origin is Union or (UnionType is not None and origin is UnionType):  # type: ignore[name-defined]
        opts = [_format_expected(a) for a in args if a is not type(None)]
        return " or ".join(opts)
    return str(t)


def _validate_and_deserialize(val: Any, ftype: Any, fname: str) -> Any:
    """Validate metadata `val` against `ftype` and return converted python value or raise ValueError."""
    origin = get_origin(ftype)
    args = get_args(ftype)
    # Union
    # Union handling
    from typing import Union as TypingUnion
    import types as _types
    UnionType = getattr(_types, "UnionType", None)
    if get_origin(ftype) is TypingUnion or (UnionType is not None and isinstance(ftype, UnionType)):
        for a in get_args(ftype):
            if a is type(None) and val is None:
                return None
            try:
                return _validate_and_deserialize(val, a, fname)
            except ValueError:
                continue
        raise ValueError(f"attribute '{fname}'. expected {_format_expected(ftype)}")

    if origin is None and isinstance(ftype, type) and issubclass(ftype, enum.Enum):
        # Enum: convert underlying value to enum instance where possible
        try:
            return ftype(val)
        except Exception:
            return val
    if origin is None and ftype in (int, float, str, bool):
        if not isinstance(val, ftype):
            raise ValueError(f"attribute '{fname}'. expected {_format_expected(ftype)}")
        return val
    if origin is None and ftype is Decimal:
        try:
            return Decimal(val)
        except Exception:
            raise ValueError(f"attribute '{fname}'. expected Decimal")
    if origin is None and ftype is Fraction:
        try:
            return Fraction(val["numerator"], val["denominator"])
        except Exception:
            raise ValueError(f"attribute '{fname}'. expected Fraction")
    if origin is None and ftype is uuid.UUID:
        try:
            return uuid.UUID(val)
        except Exception:
            raise ValueError(f"attribute '{fname}'. expected UUID")
    if origin is None and ftype in (pathlib.PurePath, pathlib.Path):
        return pathlib.PurePath(val)
    if origin is None and ftype is datetime:
         # expecting ISO format
         if not isinstance(val, str):
             raise ValueError(f"attribute '{fname}'. expected datetime string")
         return datetime.fromisoformat(val)
    if origin is None and ftype is timedelta:
        if not isinstance(val, (int, float)):
             raise ValueError(f"attribute '{fname}'. expected milliseconds for timedelta")
        return timedelta(milliseconds=int(val))
    if origin is None and ftype in (bytes, bytearray, memoryview):
        if not isinstance(val, list) or not all(isinstance(x, int) for x in val):
            raise ValueError(f"attribute '{fname}'. expected bytes-like list of ints")
        if ftype is bytes:
            return bytes(val)
        if ftype is bytearray:
            return bytearray(val)
        return memoryview(bytes(val))
    # complex
    if origin is None and ftype is complex:
        if not isinstance(val, dict) or "real" not in val or "imag" not in val:
            raise ValueError(f"attribute '{fname}'. expected complex object")
        return complex(val.get("real"), val.get("imag"))
    # arrays, range
    if origin is None and ftype is array.array:
        if not isinstance(val, list):
            raise ValueError(f"attribute '{fname}'. expected list for array")
        return array.array("i", val)
    if origin is None and ftype is range:
        if not isinstance(val, list):
            raise ValueError(f"attribute '{fname}'. expected list for range")
        return range(val[0], val[-1] + 1) if val else range(0)
    if origin is None and ftype is defaultdict:
        if not isinstance(val, dict):
            raise ValueError(f"attribute '{fname}'. expected dict for defaultdict")
        return defaultdict(int, val)
    if origin is None and ftype is OrderedDict:
        if not isinstance(val, dict):
            raise ValueError(f"attribute '{fname}'. expected dict for OrderedDict")
        return OrderedDict(val)
    if origin is None and ftype is MappingProxyType:
        if not isinstance(val, dict):
            raise ValueError(f"attribute '{fname}'. expected dict for MappingProxyType")
        return MappingProxyType(dict(val))
    if origin is None and ftype is deque:
        if not isinstance(val, list):
            raise ValueError(f"attribute '{fname}'. expected list for deque")
        return deque(val)
    if origin is None and ftype is SimpleNamespace:
        if not isinstance(val, dict):
            raise ValueError(f"attribute '{fname}'. expected object for SimpleNamespace")
        return SimpleNamespace(**val)
    # Simple containers
    if origin is list:
        if not isinstance(val, list):
            raise ValueError(f"attribute '{fname}'. expected list")
        inner = args[0] if args else Any
        return [_validate_and_deserialize(v, inner, fname) for v in val]
    if origin is tuple:
        if not isinstance(val, list):
            raise ValueError(f"attribute '{fname}'. expected list to build tuple")
        inner = args[0] if args else Any
        return tuple(_validate_and_deserialize(v, inner, fname) for v in val)
    if origin is set or origin is frozenset:
        if not isinstance(val, list):
            raise ValueError(f"attribute '{fname}'. expected list to build set")
        inner = args[0] if args else Any
        s = set(_validate_and_deserialize(v, inner, fname) for v in val)
        return frozenset(s) if origin is frozenset else s
    if origin is dict:
        if not isinstance(val, dict):
            raise ValueError(f"attribute '{fname}'. expected dict")
        ktype, vtype = args if args else (Any, Any)
        d = {}
        for k, v in val.items():
            if ktype is not Any and not isinstance(k, ktype):
                raise ValueError(f"attribute '{fname}'. expected dict key type {_format_expected(ktype)}")
            d[k] = _validate_and_deserialize(v, vtype, fname)
        return d
    # default: pass-through
    return val


def _serialize_meta_value(val: Any) -> Any:
    """Serialize a non-node metadata value into JSON-friendly primitives.
    This mirrors decode's expectations for types that are not node references.
    """
    # attrs node should not be serialized here
    if _is_attrs_instance(val):
        raise TypeError("serialize_meta_value should not be called on node instances")

    # builtin conversions
    if isinstance(val, enum.Enum):
        return val.value
    if isinstance(val, (bytes, bytearray, memoryview)):
        return list(bytes(val))
    if isinstance(val, complex):
        return {"real": val.real, "imag": val.imag}
    if isinstance(val, Decimal):
        return str(val)
    if isinstance(val, Fraction):
        return {"numerator": val.numerator, "denominator": val.denominator}
    if isinstance(val, uuid.UUID):
        return str(val)
    if isinstance(val, (pathlib.PurePath, pathlib.Path)):
        return str(val)
    if isinstance(val, datetime):
        return val.isoformat()
    if isinstance(val, timedelta):
        return int(val.total_seconds() * 1000)
    if isinstance(val, array.array):
        return list(val)
    if isinstance(val, range):
        return list(val)
    if isinstance(val, (defaultdict, OrderedDict, MappingProxyType)):
        return dict(val)
    if isinstance(val, deque):
        return list(val)
    if isinstance(val, SimpleNamespace):
        return vars(val)

    # containers - recursively convert elements
    if isinstance(val, dict):
        return {k: _serialize_meta_value(v) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return [_serialize_meta_value(v) for v in val]
    if isinstance(val, (set, frozenset)):
        # convert set to deterministic list sorted by string repr of serialized elements
        ser = [_serialize_meta_value(v) for v in val]
        return sorted(ser, key=lambda x: str(x))

    # default: pass-through primitives
    return val


class Encoder:
    def __init__(self) -> None:
        self.obj_to_id: Dict[int, str] = {}
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.hyperedges: List[Dict[str, Any]] = []
        self._next_id = 1
        self._processed: set[int] = set()

    def _new_id(self) -> str:
        nid = str(self._next_id)
        self._next_id += 1
        return nid

    def register_node(self, obj: Any) -> str:
        oid = id(obj)
        if oid in self.obj_to_id:
            return self.obj_to_id[oid]
        nid = self._new_id()
        self.obj_to_id[oid] = nid
        # placeholder metadata, filled later
        self.nodes[nid] = {"metadata": {}}
        return nid

    def encode(self, root: Any) -> Dict[str, Any]:
        if not _is_attrs_instance(root):
            raise TypeError("encode() expects an attrs instance as root")

        # traverse
        stack = [root]
        self.register_node(root)

        while stack:
            obj = stack.pop(0)
            # avoid reprocessing objects already handled (prevents cycles)
            if id(obj) in self._processed:
                continue
            src_id = self.register_node(obj)
            meta: Dict[str, Any] = {}

            for field in attrs.fields(obj.__class__):
                name = field.name
                val = getattr(obj, name)

                # don't emit None values into metadata
                if val is None:
                    continue

                # node reference
                if _is_attrs_instance(val):
                    tgt_id = self.register_node(val)
                    self.hyperedges.append({"source": [src_id], "target": [tgt_id], "relation": name})
                    if id(val) not in self._processed:
                        stack.append(val)
                    continue

                # dict
                if isinstance(val, dict):
                    # if any value is an attrs node -> edges
                    if any(_is_attrs_instance(v) for v in val.values()):
                        for k, v in val.items():
                            if _is_attrs_instance(v):
                                tgt = self.register_node(v)
                                edge = {"source": [src_id], "target": [tgt], "relation": name, "metadata": {"key": k}}
                                self.hyperedges.append(edge)
                                if id(v) not in self._processed:
                                    stack.append(v)
                        continue
                    # else serialize dict
                    # serialize nested metadata values
                    meta[name] = {k: _serialize_meta_value(v) for k, v in val.items()}
                    continue

                # list/tuple
                if isinstance(val, (list, tuple)):
                    if any(_is_attrs_instance(e) for e in val):
                        for idx, e in enumerate(val):
                            if _is_attrs_instance(e):
                                tgt = self.register_node(e)
                                edge = {"source": [src_id], "target": [tgt], "relation": name, "metadata": {"index": idx}}
                                self.hyperedges.append(edge)
                                if id(e) not in self._processed:
                                    stack.append(e)
                        continue
                    meta[name] = [_serialize_meta_value(e) for e in val]
                    continue

                # set/frozenset
                if isinstance(val, (set, frozenset)):
                    if any(_is_attrs_instance(e) for e in val):
                        for e in val:
                            if _is_attrs_instance(e):
                                tgt = self.register_node(e)
                                edge = {"source": [src_id], "target": [tgt], "relation": name}
                                self.hyperedges.append(edge)
                                if id(e) not in self._processed:
                                    stack.append(e)
                        continue
                    meta[name] = [_serialize_meta_value(e) for e in sorted(val, key=lambda x: str(_serialize_meta_value(x)))]
                    continue

                # special types
                if isinstance(val, (bytes, bytearray, memoryview)):
                    b = bytes(val)
                    meta[name] = list(b)
                    continue
                if isinstance(val, complex):
                    meta[name] = {"real": val.real, "imag": val.imag}
                    continue
                if isinstance(val, Decimal):
                    meta[name] = str(val)
                    continue
                if isinstance(val, Fraction):
                    meta[name] = {"numerator": val.numerator, "denominator": val.denominator}
                    continue
                if isinstance(val, uuid.UUID):
                    meta[name] = str(val)
                    continue
                if isinstance(val, (pathlib.PurePath, pathlib.Path)):
                    meta[name] = str(val)
                    continue
                if isinstance(val, datetime):
                    meta[name] = val.isoformat()
                    continue
                if isinstance(val, timedelta):
                    meta[name] = int(val.total_seconds() * 1000)
                    continue
                if isinstance(val, array.array):
                    meta[name] = list(val)
                    continue
                if isinstance(val, range):
                    meta[name] = list(val)
                    continue
                if isinstance(val, (defaultdict, OrderedDict, MappingProxyType)):
                    meta[name] = dict(val)
                    continue
                if isinstance(val, deque):
                    meta[name] = list(val)
                    continue
                if isinstance(val, SimpleNamespace):
                    meta[name] = vars(val)
                    continue

                # enums and flags
                if isinstance(val, enum.Enum):
                    meta[name] = val.value
                    continue

                # fallback: primitives and others
                meta[name] = _serialize_meta_value(val)

            self.nodes[src_id]["metadata"] = meta
            # mark processed to avoid reprocessing during cycles
            self._processed.add(id(obj))

        graph: Dict[str, Any] = {"graph": {"nodes": self.nodes}}
        if self.hyperedges:
            graph["graph"]["hyperedges"] = self.hyperedges
        return graph


class Decoder:
    def __init__(self) -> None:
        self.node_objs: Dict[str, Any] = {}
        self.node_types: Dict[str, Any] = {}

    def _load_input(self, data: Any) -> Dict[str, Any]:
        if hasattr(data, "read"):
            text = data.read()
            return json.loads(text)
        if isinstance(data, str):
            return json.loads(data)
        if isinstance(data, dict):
            return data
        raise TypeError("Unsupported input to decode()")

    def decode(self, data: Any, objtype: type) -> Any:
        payload = self._load_input(data)
        if "graph" not in payload:
            raise ValueError("missing 'graph' in payload")
        graph = payload["graph"]
        nodes = graph.get("nodes", {})
        hyperedges = graph.get("hyperedges", [])

        # root is node '1'
        if "1" not in nodes:
            raise ValueError("no node '1' in graph")

        # assign root type
        self.node_types["1"] = objtype

        # propagate types from hyperedges
        # do iterative propagation starting from known types
        changed = True
        while changed:
            changed = False
            for edge in hyperedges:
                src = edge["source"][0]
                tgt = edge["target"][0]
                rel = edge["relation"]
                if src in self.node_types and tgt not in self.node_types:
                    src_type = self.node_types[src]
                    # inspect annotation of rel on src_type
                    ann = None
                    try:
                        for f in attrs.fields(src_type):
                            if f.name == rel:
                                ann = f.type
                                break
                        # Resolve string (forward-ref) annotations
                        if isinstance(ann, str):
                            try:
                                hints = get_type_hints(src_type)
                                ann = hints.get(rel, ann)
                            except Exception as e:
                                # fall back to the raw annotation if resolution fails
                                import warnings
                                warnings.warn(str(e))
                    except Exception:
                        ann = None
                    if ann is not None:
                        # resolve collection element type
                        origin = get_origin(ann)
                        args = get_args(ann)
                        elem_type = None
                        if origin is dict and len(args) == 2:
                            elem_type = args[1]
                        elif origin in (list, tuple, set, frozenset) and args:
                            elem_type = args[0]
                        else:
                            # direct type or Union
                            if get_origin(ann) is None:
                                elem_type = ann
                            else:
                                # Union[...] -> pick first non-None
                                for a in args:
                                    if a is not type(None):
                                        elem_type = a
                                        break
                        if isinstance(elem_type, type):
                            self.node_types[tgt] = elem_type
                            changed = True

        # instantiate nodes with metadata
        for nid, node in nodes.items():
            nmeta = node.get("metadata", {})
            ntype = self.node_types.get(nid)
            if ntype is None:
                # fallback: create plain object with metadata as SimpleNamespace
                self.node_objs[nid] = SimpleNamespace(**nmeta)
                continue

            # validate unknown keys
            fields = {f.name: f.type for f in attrs.fields(ntype)}
            # resolve forward-ref / string annotations via get_type_hints
            try:
                hints = get_type_hints(ntype)
                for k, v in list(fields.items()):
                    if isinstance(v, str):
                        fields[k] = hints.get(k, v)
            except Exception as e:
                import warnings
                warnings.warn(str(e))
            for k in nmeta.keys():
                if k not in fields:
                    raise ValueError(f"unknown key '{k}'")

            # prepare constructor kwargs for non-node fields (primitives and collections)
            kwargs = {}
            for fname, ftype in fields.items():
                if fname in nmeta:
                    val = nmeta[fname]
                    # validate and convert
                    kwargs[fname] = _validate_and_deserialize(val, ftype, fname)

                else:
                    # missing metadata: assign None; for containers, use empty
                    origin = get_origin(ftype)
                    if origin is list:
                        kwargs[fname] = []
                    elif origin is dict:
                        kwargs[fname] = {}
                    elif origin is set:
                        kwargs[fname] = set()
                    elif origin is frozenset:
                        kwargs[fname] = frozenset()
                    else:
                        kwargs[fname] = None

            # construct instance
            obj = ntype(**kwargs)
            self.node_objs[nid] = obj

        # Now apply hyperedges to build relations
        # Helper to set container entries
        tuple_build: Dict[Tuple[str, str], Dict[int, Any]] = {}
        for edge in hyperedges:
            src = edge["source"][0]
            tgt = edge["target"][0]
            rel = edge["relation"]
            meta = edge.get("metadata", {})
            src_obj = self.node_objs.get(src)
            tgt_obj = self.node_objs.get(tgt)
            if src_obj is None or tgt_obj is None:
                continue
            # determine field type
            ann = None
            try:
                for f in attrs.fields(type(src_obj)):
                    if f.name == rel:
                        ann = f.type
                        break
            except Exception:
                ann = None
            # resolve forward-ref string annotations for field types
            if isinstance(ann, str):
                try:
                    hints = get_type_hints(type(src_obj))
                    ann = hints.get(rel, ann)
                except Exception as e:
                    import warnings
                    warnings.warn(str(e))

            origin = get_origin(ann)
            if origin is dict:
                key = meta.get("key")
                d = getattr(src_obj, rel)
                if d is None:
                    d = {}
                    setattr(src_obj, rel, d)
                d[key] = tgt_obj
            elif origin in (list,):
                idx = meta.get("index")
                lst = getattr(src_obj, rel)
                if lst is None:
                    lst = []
                    setattr(src_obj, rel, lst)
                # ensure list size
                if idx is None:
                    lst.append(tgt_obj)
                else:
                    # expand
                    while len(lst) <= idx:
                        lst.append(None)
                    lst[idx] = tgt_obj
            elif origin in (tuple,):
                idx = meta.get("index")
                key = (src, rel)
                if key not in tuple_build:
                    tuple_build[key] = {}
                tuple_build[key][idx] = tgt_obj
            elif origin in (set, frozenset):
                st = getattr(src_obj, rel)
                if st is None:
                    st = set()
                    setattr(src_obj, rel, st)
                # if field was a frozenset, convert to mutable set for building
                if isinstance(st, frozenset):
                    st = set(st)
                    setattr(src_obj, rel, st)
                st.add(tgt_obj)
            else:
                # single value
                setattr(src_obj, rel, tgt_obj)

        # finalize tuples and frozensets
        for (src, rel), items in tuple_build.items():
            lst = [v for k, v in sorted(items.items())]
            setattr(self.node_objs[src], rel, tuple(lst))

        # finalize frozensets if any - convert sets to frozenset for annotated fields
        for nid, obj in self.node_objs.items():
            if not _is_attrs_instance(obj):
                continue
            for f in attrs.fields(type(obj)):
                ann = f.type
                if get_origin(ann) is frozenset:
                    val = getattr(obj, f.name)
                    if val is None:
                        setattr(obj, f.name, frozenset())
                    elif isinstance(val, set):
                        setattr(obj, f.name, frozenset(val))

        # return root
        return self.node_objs["1"]


# Public API
def encode(obj: Any) -> Dict[str, Any]:
    enc = Encoder()
    return enc.encode(obj)


def decode(data: Any, objtype: type) -> Any:
    dec = Decoder()
    return dec.decode(data, objtype)

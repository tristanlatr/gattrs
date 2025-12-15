from __future__ import annotations

from datetime import datetime, timedelta, timezone
import enum
import json
import io
import decimal
from decimal import Decimal
from fractions import Fraction
import uuid
import pathlib
import array
import time
from collections import defaultdict, OrderedDict, deque
from types import MappingProxyType, SimpleNamespace
from typing import TypedDict

import pytest

import attrs, gattrs


@attrs.define
class PersonEncode:
    name: str
    friend: PersonEncode | None = None


def test_encode_decode_simple_roundtrip():
    a = PersonEncode("Alice")
    b = PersonEncode("Bob", friend=a)

    encoded = gattrs.encode(b)
    expected = {
        "graph": {
            "nodes": {
                "1": {"metadata": {"name": "Bob"}},
                "2": {"metadata": {"name": "Alice"}}
            },
            "hyperedges": [
                {"source": ["1"], "target": ["2"], "relation": "friend"}
            ]
        }
    }
    assert encoded == expected

    decoded = gattrs.decode(encoded, PersonEncode)

    assert isinstance(decoded, PersonEncode)
    assert decoded.name == "Bob"
    assert isinstance(decoded.friend, PersonEncode)
    assert decoded.friend.name == "Alice"


class ColorCollections(enum.Enum):
    RED = "red"
    BLUE = "blue"

@attrs.define
class ContainerCollections:
    colors: set[ColorCollections]
    items: frozenset[int]
    mapping: dict[str, int]
    tup: tuple[int, ...]


def test_collections_and_enum_roundtrip():
    c = ContainerCollections({ColorCollections.RED}, frozenset({1, 2}), {"a": 1}, (1, 2))

    encoded = gattrs.encode(c)
    expected = {
        "graph": {
            "nodes": {
                "1": {
                    "metadata": {
                        "colors": ["red"],
                        "items": [1, 2],
                        "mapping": {"a": 1},
                        "tup": [1, 2]
                    }
                }
            }
        }
    }
    assert encoded == expected

    decoded = gattrs.decode(encoded, ContainerCollections)

    assert isinstance(decoded, ContainerCollections)
    assert decoded.colors == c.colors
    assert decoded.items == c.items
    assert decoded.mapping == c.mapping
    assert decoded.tup == c.tup


@attrs.define
class EventDatetime:
    ts: datetime
    dur: timedelta


def test_datetime_and_timedelta_support():
    now = datetime.now(timezone.utc).replace(microsecond=0)
    e = EventDatetime(now, timedelta(minutes=1))

    encoded = gattrs.encode(e)
    expected = {
        "graph": {
            "nodes": {
                "1": {"metadata": {"ts": now.isoformat(), "dur": 60000}}
            }
        }
    }
    assert encoded == expected

    decoded = gattrs.decode(encoded, EventDatetime)

    assert decoded.ts == e.ts
    assert decoded.dur == e.dur


@attrs.define(frozen=True)
class FrozenImmutable:
    x: int


def test_frozen_class_support():
    f = FrozenImmutable(1)
    encoded = gattrs.encode(f)
    expected = {"graph": {"nodes": {"1": {"metadata": {"x": 1}}}}}
    assert encoded == expected

    decoded = gattrs.decode(encoded, FrozenImmutable)
    assert isinstance(decoded, FrozenImmutable)
    assert decoded.x == f.x


@attrs.define
class NodeCycle:
    name: str
    next: NodeCycle | None = None


def test_graph_cycles_are_preserved():
    a = NodeCycle("a")
    b = NodeCycle("b", a)
    a.next = b

    encoded = gattrs.encode(a)
    expected = {
        "graph": {
            "nodes": {
                "1": {"metadata": {"name": "a"}},
                "2": {"metadata": {"name": "b"}}
            },
            "hyperedges": [
                {"source": ["1"], "target": ["2"], "relation": "next"},
                {"source": ["2"], "target": ["1"], "relation": "next"}
            ]
        }
    }
    assert encoded == expected

    decoded = gattrs.decode(encoded, NodeCycle)

    # a -> b -> a (cycle) should be preserved after decode
    assert decoded.next.next is decoded

@attrs.define
class TStrict:
    a: int
    b: str

def test_strict_unknown_keys_and_error_messages():
    # unknown key 'c' should cause a strict parsing error
    bad = {
        "graph": {
            "nodes": {
                "1": {"metadata": {"a": 1, "b": "x", "c": 3}}
            }
        }
    }
    with pytest.raises(ValueError, match=r"unknown key 'c'"):
        gattrs.decode(bad, TStrict)

@attrs.define
class UInvalid:
    x: int
    y: list[int]


def test_invalid_type_reports_location_expected_actual():
    bad = {
        "graph": {
            "nodes": {"1": {"metadata": {"x": "not-an-int", "y": [1, 2, 3]}}}
        }
    }
    with pytest.raises(ValueError, match=r"attribute 'x'.*expected.*int"):
        gattrs.decode(bad, UInvalid)


@attrs.define
class VUnion:
    u: int | str


def test_union_annotation_validation_requires_py310_style():
    # wrong type (list) should raise a validation error for union type
    bad = {
        "graph": {
            "nodes": {"1": {"metadata": {"u": [1, 2, 3]}}}
        }
    }
    with pytest.raises(ValueError, match=r"attribute 'u'.*expected.*int or str"):
        gattrs.decode(bad, VUnion)

@attrs.define
class SBind:
    n: int
    s: str


def test_bind_from_file_stream_and_preparsed_dict(tmp_path):
    s = SBind(1, "ok")
    encoded = gattrs.encode(s)

    expected = {"graph": {"nodes": {"1": {"metadata": {"n": 1, "s": "ok"}}}}}
    assert encoded == expected

    # pre-parsed dict must work
    decoded_from_dict = gattrs.decode(encoded, SBind)
    assert isinstance(decoded_from_dict, SBind)
    assert decoded_from_dict.n == 1

    # write JSON to a file and try passing the file object if supported
    p = tmp_path / "g.json"
    p.write_text(json.dumps(encoded))

    # Try file-like object
    with p.open("r", encoding="utf8") as fh:
        decoded_from_fileobj = gattrs.decode(fh, SBind)
    assert isinstance(decoded_from_fileobj, SBind)

    # Try stream (StringIO)
    stream = io.StringIO(json.dumps(encoded))
    decoded_from_stream = gattrs.decode(stream, SBind)
    assert isinstance(decoded_from_stream, SBind)


class EMany(enum.Enum):
    A = "a"

@attrs.define
class ComplexMany:
    tags: set[str]
    flags: frozenset[str]
    mapping: dict[str, tuple[int, ...]]
    enums: list[EMany]


def test_support_many_builtin_types_and_complex_structures():
    c = ComplexMany({"t"}, frozenset({"f"}), {"k": (1, 2, 3)}, [EMany.A])
    encoded = gattrs.encode(c)
    expected = {
        "graph": {
            "nodes": {
                "1": {
                    "metadata": {
                        "tags": ["t"],
                        "flags": ["f"],
                        "mapping": {"k": [1, 2, 3]},
                        "enums": ["a"]
                    }
                }
            }
        }
    }
    assert encoded == expected

    decoded = gattrs.decode(encoded, ComplexMany)
    assert isinstance(decoded, ComplexMany)
    assert decoded.mapping["k"] == (1, 2, 3)


@attrs.define
class PSchema:
    a: int
    b: str




@attrs.define(frozen=True)
class ItemNode:
    name: str


@attrs.define
class NodeCollections:
    mapping_str: dict[str, ItemNode]
    mapping_int: dict[int, ItemNode]
    lst: list[ItemNode]
    tup: tuple[ItemNode, ...]
    st: set[ItemNode]
    fst: frozenset[ItemNode]


def test_node_collection_types_refer_to_nodes():
    # Create several node instances
    i1 = ItemNode("a")
    i2 = ItemNode("b")
    i3 = ItemNode("c")

    c = NodeCollections(
        {"alpha": i1},
        {2: i2},
        [i1, i2, i3],
        (i2, i3),
        {i1},
        frozenset({i3}),
    )

    encoded = gattrs.encode(c)

    expected = {
        "graph": {
            "nodes": {
                "1": {"metadata": {}},
                "2": {"metadata": {"name": "a"}},
                "3": {"metadata": {"name": "b"}},
                "4": {"metadata": {"name": "c"}},
            },
            "hyperedges": [
                {"source": ["1"], "target": ["2"], "relation": "mapping_str", "metadata": {"key": "alpha"}},
                {"source": ["1"], "target": ["3"], "relation": "mapping_int", "metadata": {"key": 2}},
                {"source": ["1"], "target": ["2"], "relation": "lst", "metadata": {"index": 0}},
                {"source": ["1"], "target": ["3"], "relation": "lst", "metadata": {"index": 1}},
                {"source": ["1"], "target": ["4"], "relation": "lst", "metadata": {"index": 2}},
                {"source": ["1"], "target": ["3"], "relation": "tup", "metadata": {"index": 0}},
                {"source": ["1"], "target": ["4"], "relation": "tup", "metadata": {"index": 1}},
                {"source": ["1"], "target": ["2"], "relation": "st"},
                {"source": ["1"], "target": ["4"], "relation": "fst"},
            ],
        }
    }

    assert encoded == expected

    decoded = gattrs.decode(encoded, NodeCollections)
    assert isinstance(decoded, NodeCollections)
    assert isinstance(decoded.lst[0], ItemNode)
    assert decoded.lst[1].name == "b"
    assert decoded.tup[1].name == "c"


@attrs.define
class BuiltinMany:
    b: bytes
    ba: bytearray
    mv: memoryview
    comp: complex
    dec: Decimal
    frac: Fraction
    uid: uuid.UUID
    p: pathlib.PurePath
    arr: array.array
    rng: range
    dflt: defaultdict
    od: OrderedDict
    mp: MappingProxyType
    dq: deque
    ns: SimpleNamespace
    tdict: dict
    flag: enum.IntFlag


class TD_TypedDict(TypedDict):
    a: int


class MFlags(enum.IntFlag):
    A = 1
    B = 2


def test_other_builtin_types_and_mapping_variants_roundtrip():
    # Prepare values
    b = b"hi"
    ba = bytearray(b"xy")
    mv = memoryview(b"z")
    comp = 1 + 2j
    dec = Decimal("1.23")
    frac = Fraction(3, 4)
    uid = uuid.UUID("11111111-1111-1111-1111-111111111111")
    p = pathlib.PurePath("a/b")
    arr = array.array("i", [1, 2])
    rng = range(1, 4)
    dflt = defaultdict(int, {"a": 1})
    od = OrderedDict([("a", 1), ("b", 2)])
    mp = MappingProxyType({"a": 1, "b": 2})
    dq = deque([1, 2, 3])
    ns = SimpleNamespace(x=5)
    tdict = {"a": 1}
    flag = MFlags.A | MFlags.B

    obj = BuiltinMany(b, ba, mv, comp, dec, frac, uid, p, arr, rng, dflt, od, mp, dq, ns, tdict, flag)

    encoded = gattrs.encode(obj)

    expected = {
        "graph": {
            "nodes": {
                "1": {
                    "metadata": {
                        "b": [104, 105],
                        "ba": [120, 121],
                        "mv": [122],
                        "comp": {"real": 1, "imag": 2},
                        "dec": "1.23",
                        "frac": {"numerator": 3, "denominator": 4},
                        "uid": "11111111-1111-1111-1111-111111111111",
                        "p": "a/b",
                        "arr": [1, 2],
                        "rng": [1, 2, 3],
                        "dflt": {"a": 1},
                        "od": {"a": 1, "b": 2},
                        "mp": {"a": 1, "b": 2},
                        "dq": [1, 2, 3],
                        "ns": {"x": 5},
                        "tdict": {"a": 1},
                        "flag": 3
                    }
                }
            }
        }
    }

    assert encoded == expected

    decoded = gattrs.decode(encoded, BuiltinMany)
    assert isinstance(decoded, BuiltinMany)
    assert decoded.b == b
    assert decoded.ba == ba
    assert bytes(decoded.mv) == bytes(mv)
    assert decoded.comp == comp
    assert decoded.dec == dec
    assert decoded.frac == frac
    assert decoded.uid == uid
    assert decoded.p == p
    assert list(decoded.arr) == list(arr)
    assert list(decoded.rng) == list(rng)
    assert dict(decoded.dflt) == dict(dflt)
    assert list(decoded.od.items()) == list(od.items())
    assert dict(decoded.mp) == dict(mp)
    assert list(decoded.dq) == list(dq)
    assert isinstance(decoded.ns, SimpleNamespace)
    assert decoded.ns.x == ns.x
    assert decoded.tdict == tdict
    assert decoded.flag == flag

@attrs.define
class HeterogenTuples():
    tup1: tuple[int, str, float, HeterogenTuples]
    tup2: tuple[int | HeterogenTuples | None, ...]
    
def test_tuples_heterogenous_types():
    a = HeterogenTuples((), ())
    b = HeterogenTuples((1, "x", 2.0, a), (a, None, 3,))

    encoded = gattrs.encode(b)
    assert encoded == {'graph': {'hyperedges': [{'metadata': {'index': 3},
                             'relation': 'tup1',
                             'source': ['1'],
                             'target': ['2']},
                            {'metadata': {'index': 0},
                             'relation': 'tup2',
                             'source': ['1'],
                             'target': ['2']}],
                        'nodes': {'1': {'metadata': {'tup1': [1, 'x', 2.0, None],
                                                    'tup2': [None, None, 3]}},
                                '2': {'metadata': {'tup1': [], 'tup2': []}}}}}

    decoded = gattrs.decode(encoded, HeterogenTuples)

    assert isinstance(decoded, HeterogenTuples)
    assert decoded == b
    assert decoded.tup1[0] == 1
    assert decoded.tup1[3] is decoded.tup2[0]


@attrs.define
class ChainNode:
    next: "ChainNode | None" = None

def make_chain_graph_simple(size: int) -> ChainNode:
    """
    Create a linear chain of ChainNode objects of length `size`.
    This produces the worst-case decode behavior for type propagation.
    """
    nodes = [ChainNode() for _ in range(size)]
    for i in range(size - 1):
        nodes[i].next = nodes[i + 1]
    return nodes[0]

def test_graph_generation_simple():
    # test graph generation
    assert gattrs.encode(make_chain_graph_simple(10)) == {'graph': {'hyperedges': [{'relation': 'next',
                             'source': ['1'],
                             'target': ['2']},
                            {'relation': 'next',
                             'source': ['2'],
                             'target': ['3']},
                            {'relation': 'next',
                             'source': ['3'],
                             'target': ['4']},
                            {'relation': 'next',
                             'source': ['4'],
                             'target': ['5']},
                            {'relation': 'next',
                             'source': ['5'],
                             'target': ['6']},
                            {'relation': 'next',
                             'source': ['6'],
                             'target': ['7']},
                            {'relation': 'next',
                             'source': ['7'],
                             'target': ['8']},
                            {'relation': 'next',
                             'source': ['8'],
                             'target': ['9']},
                            {'relation': 'next',
                             'source': ['9'],
                             'target': ['10']}],
             'nodes': {'1': {'metadata': {}},
                       '10': {'metadata': {}},
                       '2': {'metadata': {}},
                       '3': {'metadata': {}},
                       '4': {'metadata': {}},
                       '5': {'metadata': {}},
                       '6': {'metadata': {}},
                       '7': {'metadata': {}},
                       '8': {'metadata': {}},
                       '9': {'metadata': {}}}}}

def timed_decode(encoded, root_type):
    start = time.perf_counter()
    gattrs.decode(encoded, root_type)
    return time.perf_counter() - start

# -------------------------
# Worst-case decode test
# -------------------------

def test_decode_worst_case_type_propagation_is_approximately_linear():
    """
    This test constructs a worst-case chain graph for decode() where
    type propagation advances one node per iteration.

    We assert decode time grows approximately linearly with graph size.
    """

    sizes = [10, 20, 40, 80, 160, 320, 640, 1000, 1500, 2000, 10000]
    timings = []

    for size in sizes:
        root = make_chain_graph_simple(size)
        encoded = gattrs.encode(root)

        t = timed_decode(encoded, ChainNode)
        timings.append(t)

    # Print timings for debugging / inspection
    for size, t in zip(sizes, timings):
        print(f"size={size:<5} decode_time={t:.6f}s")

    # -------------------------
    # Linear growth assertion
    # -------------------------
    #
    # We compare ratios of time increases to size increases.
    # For approximately linear behavior, time/size should be bounded.
    #

    time_per_node = [t / s for t, s in zip(timings, sizes)]

    # Allow generous slack for CI variance
    max_ratio = max(time_per_node)
    min_ratio = min(time_per_node)

    # Assert growth is not super-linear (quadratic would explode here)
    assert max_ratio < min_ratio * 2, (
        "Decode time appears super-linear; "
        "type propagation may be degrading worse than expected."
    )

def make_chain_graph_deep(size: int, num_fields: int):
    """
    Create a worst-case chain graph with:
      - `size` nodes
      - `num_fields` attrs fields per node
    Each field points to the *next* node in the chain.

    This stresses:
      - decode field iteration (F)
      - hyperedge processing (E ≈ N * F)
      - type propagation convergence
    """

    # -------------------------
    # Dynamically generate attrs class
    # -------------------------

    field_defs = {
        f"child_{i}": attrs.field(default=None)
        for i in range(num_fields)
    }

    ChainNode = attrs.make_class(
        "ChainNode",
        field_defs,
        slots=False,
    )

    # -------------------------
    # Build chain instances
    # -------------------------

    nodes = [ChainNode() for _ in range(size)]

    for i in range(size - 1):
        for j in range(num_fields):
            setattr(nodes[i], f"child_{j}", nodes[i + 1])

    return nodes[0], ChainNode

def test_graph_generation_deep():
    assert gattrs.encode(make_chain_graph_deep(3, 2)[0]) == {
        'graph': {'hyperedges': [{'relation': 'child_0',
                             'source': ['1'],
                             'target': ['2']},
                            {'relation': 'child_1',
                             'source': ['1'],
                             'target': ['2']},
                            {'relation': 'child_0',
                             'source': ['2'],
                             'target': ['3']},
                            {'relation': 'child_1',
                             'source': ['2'],
                             'target': ['3']}],
             'nodes': {'1': {'metadata': {}},
                       '2': {'metadata': {}},
                       '3': {'metadata': {}}}}}
    

    assert gattrs.encode(make_chain_graph_deep(5, 5)[0]) == {'graph': {'hyperedges': [{'relation': 'child_0',
                             'source': ['1'],
                             'target': ['2']},
                            {'relation': 'child_1',
                             'source': ['1'],
                             'target': ['2']},
                            {'relation': 'child_2',
                             'source': ['1'],
                             'target': ['2']},
                            {'relation': 'child_3',
                             'source': ['1'],
                             'target': ['2']},
                            {'relation': 'child_4',
                             'source': ['1'],
                             'target': ['2']},
                            {'relation': 'child_0',
                             'source': ['2'],
                             'target': ['3']},
                            {'relation': 'child_1',
                             'source': ['2'],
                             'target': ['3']},
                            {'relation': 'child_2',
                             'source': ['2'],
                             'target': ['3']},
                            {'relation': 'child_3',
                             'source': ['2'],
                             'target': ['3']},
                            {'relation': 'child_4',
                             'source': ['2'],
                             'target': ['3']},
                            {'relation': 'child_0',
                             'source': ['3'],
                             'target': ['4']},
                            {'relation': 'child_1',
                             'source': ['3'],
                             'target': ['4']},
                            {'relation': 'child_2',
                             'source': ['3'],
                             'target': ['4']},
                            {'relation': 'child_3',
                             'source': ['3'],
                             'target': ['4']},
                            {'relation': 'child_4',
                             'source': ['3'],
                             'target': ['4']},
                            {'relation': 'child_0',
                             'source': ['4'],
                             'target': ['5']},
                            {'relation': 'child_1',
                             'source': ['4'],
                             'target': ['5']},
                            {'relation': 'child_2',
                             'source': ['4'],
                             'target': ['5']},
                            {'relation': 'child_3',
                             'source': ['4'],
                             'target': ['5']},
                            {'relation': 'child_4',
                             'source': ['4'],
                             'target': ['5']}],
             'nodes': {'1': {'metadata': {}},
                       '2': {'metadata': {}},
                       '3': {'metadata': {}},
                       '4': {'metadata': {}},
                       '5': {'metadata': {}}}}}
    
    obj, objtype = make_chain_graph_deep(11, 9)
    assert obj
    assert gattrs.encode(obj) == {}

def test_encode_decode_worst_case_many_fields():
    import time
    import gattrs

    sizes = [10, 25, 50, 100, 200]
    num_fields = 20

    ecode_times = []
    decode_times = []

    for n in sizes:
        root, NodeType = make_chain_graph_deep(n, num_fields)
        start = time.perf_counter()
        encoded = gattrs.encode(root)
        ecode_times.append(time.perf_counter() - start)

        start = time.perf_counter()
        gattrs.decode(encoded, NodeType)
        decode_times.append(time.perf_counter() - start)

    for n, t in zip(sizes, ecode_times):
        print(f"N={n:<4} F={num_fields:<3} decode={t:.6f}s")
    ratios = [ecode_times[i+1] / ecode_times[i] for i in range(len(ecode_times) - 1)]
    # Allow noise, but fail on quadratic explosion
    assert max(ratios) < 6, 'decode time appears super-linear; type propagation may be degrading worse than expected.'

    for n, t in zip(sizes, decode_times):
        print(f"N={n:<4} F={num_fields:<3} decode={t:.6f}s")
    ratios = [decode_times[i+1] / decode_times[i] for i in range(len(decode_times) - 1)]
    # Allow noise, but fail on quadratic explosion
    assert max(ratios) < 6, 'decode time appears super-linear; type propagation may be degrading worse than expected.'
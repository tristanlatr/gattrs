# gattrs

A layer on top of cattrs to support [de]serializing graph-like structures

The goal of this project is to provide functions `gattrs.encode(obj)` and `gattrs.decode(data, objtype)` which turns a stucture composed of attrs classes into a plain dictionary.

Vertexes are attrs classes.
Edges are attributes of classes.

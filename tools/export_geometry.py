"""Run with klayout -b -r; paths are passed via environment variables."""
import json
import os
import pya
layout = pya.Layout()
layout.read(os.environ['ALIGN_GDS'])
polygons = []
for layer in layout.layer_indexes():
    iterator = layout.top_cell().begin_shapes_rec(layer)
    while not iterator.at_end():
        shape = iterator.shape()
        if shape.is_polygon() or shape.is_box():
            polygon = shape.polygon if shape.is_polygon() else pya.Polygon(shape.box)
            polygon = polygon.transformed(iterator.trans())
            polygons.append([[p.x * layout.dbu, p.y * layout.dbu]
                             for p in polygon.each_point_hull()])
        iterator.next()
json.dump(polygons, open(os.environ['ALIGN_GEOMETRY'], 'w'))

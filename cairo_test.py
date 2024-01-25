import cairo
import colorsys
import math

drawing = [
    ((1, 0), 0),
    ((2, 0), 0),
    ((2, 1), 0),
    ((3, 0), 0),
    ((3, 1), 0),
    ((3, 2), 2),
    ((4, 0), 1),
    ((4, 1), 0),
    ((4, 2), 2),
    ((5, 0), 1),
    ((5, 1), 2),
    ((5, 2), 2),
    ((6, 0), 1),
    ((6, 1), 1),
    ((6, 2), 2),
    ((7, 0), 1),
    ((7, 1), 1),
    ((7, 2), 3),
    ((8, 0), 3),
    ((8, 1), 3),
    ((8, 2), 3),
    ((9, 1), 3),
    ((9, 2), 3),
]

COLORS = [colorsys.hls_to_rgb(hue / 360, 0.5, 0.7) for hue in [0, 60, 150, 270]]

def points_up(x, y):
    """Returns whether the triangle at (x, y) is up-pointing."""
    return (x + y) % 2 == 0


def vertices(xy):
    """Returns the three floating point coordinates of the vertices of the
    triangle. Intended for use with drawing packages.
    """
    x, y = xy

    side = 100  # width of a trianglar cell in pixels

    half_side = 0.5 * side
    center = half_side * x
    left = center - half_side
    right = center + half_side

    side_root3h = half_side * math.sqrt(3)
    top = side_root3h * (y - 1)
    bottom = side_root3h * y

    return (
        [(center, top), (right, bottom), (left, bottom)]
        if points_up(x, y)
        else [(left, top), (right, top), (center, bottom)]
    )


def draw_piece(cells):
    min_x = math.inf
    max_x = -math.inf
    min_y = math.inf
    max_y = -math.inf

    for c in cells:
        for v in vertices(c[0]):
            min_x = min(min_x, v[0])
            max_x = max(max_x, v[0])
            min_y = min(min_y, v[1])
            max_y = max(max_y, v[1])

    width = int(math.ceil(1 + max_x - min_x))
    height = int(math.ceil(1 + max_y - min_y))
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
    ctx = cairo.Context(surface)

    ctx.set_source_rgba(0.0, 0.0, 0.0, 0.0) # transparent black
    ctx.rectangle(0, 0, width, height)
    ctx.fill()

    for c in cells:
        edges = [((v[0] - min_x), (v[1] - min_y)) for v in vertices(c[0])]
        for i in range(2):
            ctx.move_to(*edges[0])
            for e in edges[1:]:
                ctx.line_to(*e)
            ctx.close_path()
            if i==1:
                ctx.set_source_rgb(1, 1, 1)  # white
                ctx.stroke()
            else:
                ctx.set_source_rgb(*COLORS[c[1]])
                ctx.fill()

    surface.write_to_png("example.png")


draw_piece(drawing)

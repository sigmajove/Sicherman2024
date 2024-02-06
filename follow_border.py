import cairo
import colorsys
import math
import tqdm
from matplotlib import colors

# Here are the pieces of the puzzle as expressed by the angle at
# each vertex. An is a value in range(1, 6) expressed in units of 60 degrees.
PIECE0 = [2, 1, 5, 1, 3, 1, 3, 2]

# For all but the first piece, we reverse each list to represent flipping the
# piece.
VARIATIONS = [
    [p, [x for x in reversed(p)]]
    for p in [
        [1, 3, 2, 1, 4, 2, 1, 4],
        [2, 1, 4, 1, 3, 1, 3],
        [2, 2, 1, 5, 2, 1, 3, 2],
    ]
]

COLORS = [
    colorsys.hls_to_rgb(hue / 360.0, 0.5, 0.7) for hue in [0, 60, 150, 270]
]


class Cursor:
    """A class that can be used to find the vertices of a polygon
    specified by the angle at each vertex.

    A direction is an integer in range(6), specifying an angle at 60
    degree intervals. Zero means horizontally to the right. Increasing the value
    by 1 rotates the direction 60 degrees clockwise.
    """

    def __init__(self, x, y, direction):
        """x, y, and direction specify the position and rotation of
        the figure whose vertices are computed.
        (x, y) should be the coordinates desired for the first vertex.
        direction should be the direction the edge enters the first vertex.
        """
        self.x = x
        self.y = y
        self.direction = direction

    def advance(self, angle):
        """advance should be called for each vertex of the figure,
        passing in the angle of that vertex. Angle should be in range(1, 6),
        with a unit of 60 degrees.
        Between calls, (x, y) are the coordinates of a vertex. After n calls,
        x, y, and direction will have the values originally passed to
        the constructor.
        """

        new_direction = (self.direction + 3 - angle) % len(Cursor.WHEEL)
        dx, dy = Cursor.WHEEL[new_direction]
        self.x += dx
        self.y += dy
        self.direction = new_direction
        return self.vertex()

    def vertex(self):
        return (self.x, self.y)


# WHEEL[direction] is the coordinates of the point adjacent to (0, 0)
# in the given direction.
Cursor.WHEEL = [(2, 0), (1, 1), (-1, 1), (-2, 0), (-1, -1), (1, -1)]


# The height of an equilateral triangle with base 1.
ROOT3H = 0.5 * math.sqrt(3)


def show_piece(p0):
    c = Cursor(0, 0, 0)
    for v in p0:
        print(*cursor.advance(v))


def circular_slice(data, start, end):
    """Return the slice starting at start and ending at end, returning
    neither endpoint. However, the buffer is considered to be circular.
    if end has wrapped around, assemble the two pieces to implement
    wrapping.
    """

    # Circular buffers have a classic problem that completely empty and
    # completely full are not distinguishable. In this case, we assume
    # the buffer is never completely full.
    return (
        data[start + 1 : end]
        if start <= end
        else data[start + 1 :] + data[:end]
    )


def write_surfaces(name, surfaces):
    border = 20
    agg_width = max(s.get_width() for s in surfaces) + 2 * border
    agg_height = sum(s.get_height() for s in surfaces) + border * (
        len(surfaces) + 1
    )
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, agg_width, agg_height)
    ctx = cairo.Context(surface)
    y_val = border
    for s in surfaces:
        ctx.set_source_surface(s, border, y_val)
        ctx.paint()
        y_val += s.get_height() + border
    surface.write_to_png(name)


class Solver:
    def __init__(self):
        # An answer consists of a an [(x, y), flip, direction] for
        # each of the three pieces.
        self._answer = 3 * [None]
        self._progress = tqdm.tqdm(unit=" candidates")
        self._surfaces = []
        self._pos_id = 0
        self._num_solutions = 0

    def _try_connect(self, p0, i, var_id, flip_id, j):
        p1 = VARIATIONS[var_id][flip_id]
        start = p0[i] + p1[j]
        if start >= 6:
            return None
        ii = i
        jj = j
        while True:
            ii = (ii + 1) % len(p0)
            jj = (jj - 1) % len(p1)
            finish = p0[ii] + p1[jj]
            if finish > 6:
                return None
            elif finish == 6:
                continue
            else:
                break

        splice = (
            circular_slice(p0, ii, i)
            + [start]
            +  circular_slice(p1, j, jj)
            + [finish]
        )

        # Test if the border crosses itself.
        # This means there is an overlap or hole.
        c = Cursor(0, 0, 0)
        if len(set(c.advance(a) for a in splice)) != len(splice):
          return None

#       # Determine the position of p1
#       c = Cursor(0, 0, 0)
#       for v in p0[0:i]:
#           c.advance(v)
#       c.advance(start)
#       for v in p1[j + 1 :]:
#           c.advance(v)
#       self._pos_id += 1

#       self._answer[var_id] = [c.vertex(), flip_id, c.direction, self._pos_id]

        return splice

    def _connect_all(self, p0, var_id):
        # p0 is an aggregate piece, expressed as a list of angles.
        # var_id is the index one of the VARIATIONS
        # Generate all legal ways of connecting the two pieces.
        var = VARIATIONS[var_id]
        for flip_id in range(len(var)):
            for i in range(len(p0)):
                for j in range(len(var[flip_id])):
                    t = self._try_connect(p0, i, var_id, flip_id, j)
                    if t:
                        yield t

    def solve(self):
        # Try connecting each of the three variations with PIECE0,
        # deferring the other two variations to _level_two.
        self._level_one(PIECE0, 0, 1, 2)
        self._level_one(PIECE0, 1, 0, 2)
        self._level_one(PIECE0, 2, 0, 1)
        self._progress.close()
        print(f"Found {self._num_solutions} solutions")
        # self._display_answer()
        write_surfaces("outlines.png", self._surfaces)

    def _level_one(self, p0, var_id, x0, x1):
        for p1 in self._connect_all(p0, var_id):
            self._level_two(p1, x0, x1)
            self._level_two(p1, x1, x0)
        self._answer[var_id] = None

    def _level_two(self, p0, var_id, x0):
        for p1 in self._connect_all(p0, var_id):
            self._level_three(p1, x0)
        self._answer[var_id] = None

    def _level_three(self, p0, var_id):
        for p1 in self._connect_all(p0, var_id):
            self._progress.update(1)
            # p1 is the boundary of a way of connecting
            # all four pieces.  Test if p1 is convex
            if all(a <= 3 for a in p1):
                # We have found a solution
                self._num_solutions += 1
                

                self._surfaces.append(
                    make_surface([[generate_piece(p1, (0, 0), 0), "salmon"]])
                )
        self._answer[var_id] = None

#   def _display_answer(self):
#       surfaces = []
#       for solution in self._solutions:
#           result = [[generate_piece(PIECE0, (0, 0), 0), COLORS[0]]]
#           for i, a in enumerate(solution):
#               xy, flip_id, direction, debug_id = a
#               print(xy, flip_id, direction, debug_id)
#               result.append(
#                   [
#                       generate_piece(VARIATIONS[i][flip_id], xy, direction),
#                       COLORS[i + 1],
#                   ]
#               )
#           surfaces.append(make_surface(result))
#       write_surfaces("answer.png", surfaces)


def triscale(x, y):
    return [0.5 * x, ROOT3H * y]


def draw_pieces(name, pieces):
    # Use a recording surface to avoid having to calculate the bounds
    # of the image.  The background will be transparent.
    recording = cairo.RecordingSurface(cairo.FORMAT_ARGB32, None)
    ctx = cairo.Context(recording)
    ctx.scale(100, -100)

    for points, color in pieces:
        for i in range(2):
            ctx.move_to(*triscale(*points[0]))
            for v in points[1:]:
                ctx.line_to(*triscale(*v))
            ctx.close_path()

            match i:
                case 1:
                    ctx.set_line_width(0.025)
                    ctx.set_source_rgb(0, 0, 0)  # black
                    ctx.stroke()

                case 0:
                    ctx.set_source_rgb(*colors.to_rgb(color))
                    ctx.fill()

    # Get the bounds of the recorded image.
    x, y, width, height = recording.ink_extents()
    width = math.ceil(width)
    height = math.ceil(height)

    # Replay the created image into a real surface of the appropriate size.
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
    ctx = cairo.Context(surface)
    ctx.set_source_surface(recording, -x, -y)
    ctx.paint()

    border = 10
    width += border
    height += border

    # Create slightly larger surface.
    larger = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
    ctx = cairo.Context(larger)

    # Fill the new surface with white.
    ctx.rectangle(0, 0, width, height)
    ctx.set_source_rgb(1, 1, 1)
    ctx.fill()

    # Paint the replayed image over the white image. This works because of
    # the transparent backgrounds in recording and surface.
    ctx.set_source_surface(surface, border / 2, border / 2)
    ctx.paint()

    larger.write_to_png(name)


def make_surface(pieces):
    # Use a recording surface to avoid having to calculate the bounds
    # of the image.  The background will be transparent.
    recording = cairo.RecordingSurface(cairo.FORMAT_ARGB32, None)
    ctx = cairo.Context(recording)
    ctx.scale(100, -100)

    for points, color in pieces:
        for i in range(2):
            ctx.move_to(*triscale(*points[0]))
            for v in points[1:]:
                ctx.line_to(*triscale(*v))
            ctx.close_path()

            match i:
                case 1:
                    ctx.set_line_width(0.025)
                    ctx.set_source_rgb(0, 0, 0)  # black
                    ctx.stroke()

                case 0:
                    ctx.set_source_rgb(*colors.to_rgb(color))
                    ctx.fill()

    # Get the bounds of the recorded image.
    x, y, width, height = recording.ink_extents()
    width = math.ceil(width)
    height = math.ceil(height)

    # Replay the created image into a real surface of the appropriate size.
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
    ctx = cairo.Context(surface)
    ctx.set_source_surface(recording, -x, -y)
    ctx.paint()

    return surface


def generate_piece(vertices, start, direction):
    c = Cursor(*start, direction)
    return [c.advance(a) for a in vertices]


def make_drawing(name, points):
    # Use a recording surface to avoid having to calculate the bounds
    # of the image.  The background will be transparent.
    recording = cairo.RecordingSurface(cairo.FORMAT_ARGB32, None)
    ctx = cairo.Context(recording)
    ctx.scale(100, -100)

    for i in range(2):
        ctx.move_to(*triscale(*points[0]))
        for v in points[1:]:
            ctx.line_to(*triscale(*v))
        ctx.close_path()

        match i:
            case 1:
                ctx.set_line_width(0.025)
                ctx.set_source_rgb(0, 0, 0)  # black
                ctx.stroke()

            case 0:
                ctx.set_source_rgb(*colors.to_rgb("salmon"))
                ctx.fill()

    # Get the bounds of the recorded image.
    x, y, width, height = recording.ink_extents()
    width = math.ceil(width)
    height = math.ceil(height)

    # Replay the created image into a real surface of the appropriate size.
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
    ctx = cairo.Context(surface)
    ctx.set_source_surface(recording, -x, -y)
    ctx.paint()

    border = 10
    width += border
    height += border

    # Create slightly larger surface.
    larger = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
    ctx = cairo.Context(larger)

    # Fill the new surface with white.
    ctx.rectangle(0, 0, width, height)
    ctx.set_source_rgb(1, 1, 1)
    ctx.fill()

    # Paint the replayed image over the white image. This works because of
    # the transparent backgrounds in recording and surface.
    ctx.set_source_surface(surface, border / 2, border / 2)
    ctx.paint()

    larger.write_to_png(name)


if __name__ == "__main__":
    Solver().solve()

import cairo
import colorsys
import math
import tqdm
from dataclasses import dataclass

# Here are the pieces of the puzzle as expressed by the angle at each vertex.
# An angle is a value in range(1, 6) expressed in units of 60 degrees.
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


@dataclass(slots=True)
class Cursor:
    """A class that can be used to find the vertices of a polygon
    specified by the angle at each vertex.

    """

    # x, y, and direction specify the position and rotation of
    # the figure whose vertices are computed.
    # (x, y) should be the coordinates desired for the first vertex.
    x: int
    y: int

    # direction should be the direction the edge enters the first vertex.
    # A direction is an integer in range(6), specifying an angle at 60
    # degree intervals. Zero means horizontally to the right. Increasing
    #  the value by 1 rotates the direction 60 degrees clockwise.
    direction: int

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


@dataclass(slots=True)
class StackFrame:
    border: list[int]
    position: list[int]  # x, y, direction


def circular_slice(data, start, end):
    """Return the slice starting at start and ending at end, returning
    neither endpoint. The data buffer is considered to be circular.
    if end has wrapped around, assemble the two pieces to implement
    the wrapping.
    """

    # Circular buffers have a classic problem that completely empty and
    # completely full are not distinguishable. In this case, we assume
    # the buffer is never completely full.
    return (
        data[start + 1 : end]
        if start <= end
        else data[start + 1 :] + data[:end]
    )


def valley_to_valley(piece):
    """Here is a filter that enormously speeds up the search.
    When the final piece is placed, for the solution to be
    correct, it must fill in all remaining valleys. Since
    we are blindly assembling pieces with no guiding intelligence,
    the valleys tend to get equally spread out along the border.
    This function traverses a border and finds the smallest section
    that touches every valley. If that value is larger than the
    circumference of the final piece, there is no point in trying
    every way the piece can be attached.
    """
    prev_valley = None
    first_valley = None
    max_delta = -math.inf
    i = 0
    while True:
        if 4 <= piece[i] <= 5:
            # is a valley
            if prev_valley is not None:
                delta = i - prev_valley
                if delta <= 0:
                    delta += len(piece)
                if delta > max_delta:
                    max_delta = delta
            if first_valley is None:
                first_valley = i
            elif i == first_valley:
                # We have measured every valley
                return len(piece) - max_delta
            prev_valley = i
        i = (i + 1) % len(piece)


class Solver:
    def __init__(self):
        self._progress = tqdm.tqdm(unit=" candidates")
        self._stack = [StackFrame(PIECE0, [0, 0, 0])]
        self._solutions = set()
        self._duplicate_solutions = 0

        # An answer consists of an [flip, x, y, direction] for
        # each of the three pieces.
        self._answer = 3 * [None]

        self._filtered = 0

    def _border(self):
        """Return the border on top of the stack"""
        return self._stack[-1].border

    def _cursor(self):
        return Cursor(*self._stack[-1].position)

    def _try_connect(self, i, var_id, flip_id, j) -> bool:
        p0 = self._border()

        p1 = VARIATIONS[var_id][flip_id]
        start = p0[i] + p1[j]
        if start >= 6:
            return False
        ii = i
        jj = j
        while True:
            ii = (ii + 1) % len(p0)
            jj = (jj - 1) % len(p1)
            finish = p0[ii] + p1[jj]
            if finish > 6:
                return False
            elif finish == 6:
                continue
            else:
                break

        splice = (
            circular_slice(p0, ii, i)
            + [start]
            + circular_slice(p1, j, jj)
            + [finish]
        )

        # Test if the border crosses itself.
        # This means there is an overlap or hole.
        c = self._cursor()
        for v in p0[: ii + 1]:
            c.advance(v)
        if len(set(c.advance(a) for a in splice)) != len(splice):
            return False

        # Determine the position of the newly placed piece.
        d = self._cursor()
        for v in p0[:i]:
            d.advance(v)
        d.advance(start)
        for v in p1[j + 1 :]:
            d.advance(v)

        self._answer[var_id] = [flip_id, d.x, d.y, d.direction]
        self._stack.append(StackFrame(splice, [c.x, c.y, c.direction]))
        return True

    def _connect_all(self, var_id):
        # p0 is an aggregate piece, expressed as a list of angles.
        # var_id is the index one of the VARIATIONS
        # Generate all legal ways of connecting the two pieces.
        var = VARIATIONS[var_id]
        for flip_id in range(len(var)):
            for i in range(len(self._border())):
                for j in range(len(var[flip_id])):
                    if self._try_connect(i, var_id, flip_id, j):
                        yield None

    def solve(self):
        # Try connecting each of the three variations with PIECE0,
        # deferring the other two variations to _level_two.
        self._level_one(0, 1, 2)
        self._level_one(1, 0, 2)
        self._level_one(2, 0, 1)
        self._progress.close()
        num_solutions = len(self._solutions)
        print(f"{self._filtered} candidates were filtered")
        match num_solutions:
            case 0:
                print("Found no solutions")
            case 1:
                print("Found a single solution")
                self._display_answer()
            case _:
                print(f"Found {num_solutions} solutions")
                self._display_answer()
        print(f"There are {self._duplicate_solutions-1} duplicates")

    def _level_one(self, var_id, x0, x1):
        for _ in self._connect_all(var_id):
            self._level_two(x0, x1)
            self._level_two(x1, x0)
            self._stack.pop()

    def _level_two(self, var_id, x0):
        for p1 in self._connect_all(var_id):
            self._level_three(x0)
            self._stack.pop()

    def _level_three(self, var_id):
        if valley_to_valley(self._border()) > len(VARIATIONS[var_id][0]):
            self._filtered += 1
            return
        for _ in self._connect_all(var_id):
            self._progress.update(1)
            # self._border() surronds the connection of all four puzzle pieces.
            # Test if the border is convex
            if all(a <= 3 for a in self._border()):
                # We have found a solution
                self._solutions.add(tuple(tuple(x) for x in self._answer))
                self._duplicate_solutions += 1

            self._stack.pop()

    def _display_answer(self):
        def generate_piece(vertices, x, y, direction):
            c = Cursor(x, y, direction)
            return [c.advance(a) for a in vertices]

        surfaces = []
        for solution in self._solutions:
            result = [[generate_piece(PIECE0, 0, 0, 0), COLORS[0]]]
            for i, a in enumerate(solution):
                flip_id, x, y, direction = a
                result.append(
                    [
                        generate_piece(VARIATIONS[i][flip_id], x, y, direction),
                        COLORS[i + 1],
                    ]
                )
            surfaces.append(make_surface(result))
        write_surfaces("answer.png", surfaces)


# The height of an equilateral triangle with base 1.
ROOT3H = 0.5 * math.sqrt(3)


def make_surface(pieces):
    # Use a recording surface to avoid having to calculate the bounds
    # of the image.  The background will be transparent.
    recording = cairo.RecordingSurface(cairo.FORMAT_ARGB32, None)
    ctx = cairo.Context(recording)
    ctx.scale(100, -100)

    triscale = lambda x, y: [0.5 * x, ROOT3H * y]

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
                    ctx.set_source_rgb(*color)
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


if __name__ == "__main__":
    #   import cProfile
    #   cProfile.run("Solver().solve()")
    Solver().solve()

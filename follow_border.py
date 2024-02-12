""" This progam finds the solution to George Sicherman's 2024 New Year Puzzle
See https://sicherman.net/2024/2024.html
"""

import colorsys
import math
import time
from dataclasses import dataclass
import cairo


@dataclass(slots=True)
class Cursor:
    """A class that can be used to iterate over vertices of a polygon
    specified by an angle list.
    """

    # The (x, y) coordinates of the current vertex.
    x: int
    y: int

    # The direction the incoming edge when it enters (x, y).
    # A direction is an integer in range(6), specifying an angle at 60
    # degree intervals. Zero means horizontally to the right. Increasing
    # the value by 1 rotates the direction 60 degrees counterclockwise.
    direction: int

    def advance(self, angle):
        """Advances to the next vertex counterclockwise around
        the border of the polygon.  The value of angle must be
        the angle of the current vertex, with a unit of 60 degrees.
        """

        # This function is performance critical, so we don't use all the
        # abstraction we otherwise might.

        # Adding three to incoming angle reverses it.
        # Then subtracting the angle of this vertex points us at the next one.
        self.direction = (self.direction + 3 - angle) % 6

        # Move one step in the new direction.
        dx, dy = WHEEL[self.direction]
        self.x += dx
        self.y += dy


# WHEEL[direction] is the coordinates of the point adjacent to (0, 0)
# in the given direction.  Like the spokes of a wheel.
WHEEL = [(2, 0), (1, 1), (-1, 1), (-2, 0), (-1, -1), (1, -1)]


def valley_to_valley(piece):
    """Here is a filter that enormously speeds up the search.
    When the final piece is placed, for the solution to be
    correct, it must fill in all remaining valleys. Since
    we are blindly assembling pieces with no guiding intelligence,
    the valleys tend to get equally spread out along the border.
    This function traverses a border and finds the smallest section
    that touches every valley. If that value is larger than the
    circumference of the final piece, there is no point in trying
    every way that piece can be attached.
    """
    # Find a valley
    one_valley = next((i for i, v in enumerate(piece) if 4 <= v <= 5), None)
    if one_valley is None:
        # There is no valley
        return 0

    max_delta = -math.inf
    i = (one_valley + 1) % len(piece)
    prev_valley = one_valley
    while True:
        if 4 <= piece[i] <= 5:
            # vertex i is a valley
            delta = i - prev_valley
            if delta <= 0:
                delta += len(piece)
            if delta > max_delta:
                max_delta = delta
            if i == one_valley:
                # We have measured every valley
                return len(piece) - max_delta
            prev_valley = i
        i = (i + 1) % len(piece)


@dataclass(slots=True)
class StackFrame:
    """The type of Solver._stack[i]"""

    # The border of the the assembled pieces, expressed as a list of angles.
    border: list[int]

    # The position [x, y, direction] of border.
    position: list[int]


class Solver:
    """Solves George Sicherman's 2024 New Year Puzzle."""

    def __init__(self, piece0, variations):
        # Each piece of the puzzle are described as the list of angles
        # at each vertex of the polygon. An angle is a value in range(1, 6)
        # expressed in units of 60 degrees.
        # For all but the first piece, we have a list of list of variations
        # to allow for fliping a piece.

        # The subsequent pieces have variations to allow for flipping the piece.
        self._variations = variations

        # We implement the search tree using a stack.
        self._stack = [StackFrame(piece0, [0, 0, 0])]

        # An answer consists of an [flip, x, y, direction] for
        # each of the three pieces.  None means the piece is unplaced.
        self._answer = 3 * [None]

        # Used to record every solution we find.
        # It is a set to filter out duplicate solutions.
        self._solutions = set()

        self._duplicate_solutions = 0
        self._candidates_tested = 0

        self._piece0 = piece0

    def _border(self):
        """Return the border on top of the stack"""
        return self._stack[-1].border

    def _position(self):
        """Return the starting position for the border on top of stack."""
        return self._stack[-1].position

    def _try_connect(self, i, var_id, flip_id, j):
        """Test if is possible to connect the piece at
        self._variations[var_id][flip_id] with the border on top of the
        stack at position (i, j). If it is possible, pushes the combined
        border onto the stack and return True. If not, return False.
        """
        p0 = self._border()
        p1 = self._variations[var_id][flip_id]

        start = p0[i] + p1[j]
        if start >= 6:
            return False
        next_i = i
        prev_j = j
        while True:
            next_i = (next_i + 1) % len(p0)
            prev_j = (prev_j - 1) % len(p1)
            finish = p0[next_i] + p1[prev_j]
            if finish > 6:
                return False
            if finish == 6:
                continue
            break

        # Compute [finish] + p0[next_i + 1 : j] + [start] + p1[j + 1 : prev_j]
        # implementing wraparound in p0 and p1.
        splice = [finish]
        if next_i <= i:
            splice += p0[next_i + 1 : i]
        else:
            splice += p0[next_i + 1 :]
            splice += p0[:i]
        splice.append(start)
        if j <= prev_j:
            splice += p1[j + 1 : prev_j]
        else:
            splice += p1[j + 1 :]
            splice += p1[:prev_j]

        # Test if the border crosses itself.
        # This means there is an overlap or hole.
        if test_for_overlap(splice):
            return False

        # Find the starting positions for the splice and the
        # newly placed piece.
        c = Cursor(*self._position())
        for v in p0[:i]:
            c.advance(v)
        c.advance(start)
        if j <= prev_j:
            for v in p1[j + 1 : prev_j]:
                c.advance(v)
        else:
            for v in p1[j + 1 :]:
                c.advance(v)
            for v in p1[:prev_j]:
                c.advance(v)

        # The new starting location for splice.
        splice_x = c.x
        splice_y = c.y
        splice_d = c.direction

        # Iterate the rest of p1 to get its starting position.
        for v in p1[prev_j:]:
            c.advance(v)

        # Record the position of the placed piece, in case this is
        # part of a solution.
        self._answer[var_id] = [flip_id, c.x, c.y, c.direction]

        # Push the splice onto the stack to be used by deeper levels.
        self._stack.append(StackFrame(splice, [splice_x, splice_y, splice_d]))
        return True

    def _connect_all(self, var_id):
        """var_id is the index one of the pieces in self._variations.
        Generate all legal ways of attaching this piece to the aggregate
        on top of the stack.

        This function has a peculiar interface. It is a generator that returns
        a series on None values. Each time we yield a value, we also push
        the new combination on the stack. It is the responsibility of the
        caller to pop this value when they are done with it.
        """
        var = self._variations[var_id]
        for flip_id in range(len(var)):
            for i in range(len(self._border())):
                for j in range(len(var[flip_id])):
                    if self._try_connect(i, var_id, flip_id, j):
                        yield None

    def solve(self):
        """Solve the puzzle returing:
        [number of non-unique soltions,
         set of unique solutions,
         running time in nanoseconds
        ]
        """
        start_time = time.perf_counter_ns()

        # Try connecting each of the variations with piece0,
        # and then explore all the ways of connecting the rest
        # of the pieces.

        iota = list(range(len(self._variations)))
        for i, var_id in enumerate(iota):
            not_i = iota[:]
            not_i.pop(i)
            self._explore(var_id, not_i)

        elapsed_time = time.perf_counter_ns() - start_time

        return [
            self._duplicate_solutions,
            self._solutions,
            self._candidates_tested,
            elapsed_time,
        ]

    def _explore(self, var_id, rest):
        """Examine all ways of attaching the piece specified by var_id
        to what is on the top of the stack, and then recursively examine
        all the var_ids in rest.
        """
        if len(rest) == 0:
            if valley_to_valley(self._border()) > len(
                self._variations[var_id][0]
            ):
                # A very effective optimization.
                return

            for _ in self._connect_all(var_id):
                # self._border() surronds the connection of all four puzzle
                # pieces. Test if that border is convex.
                self._candidates_tested += 1
                if all(a <= 3 for a in self._border()):
                    # We have found a solution
                    self._solutions.add(tuple(tuple(x) for x in self._answer))
                    self._duplicate_solutions += 1
                self._stack.pop()
                self._answer[var_id] = None
        else:
            for _ in self._connect_all(var_id):
                for i, r in enumerate(rest):
                    not_i = rest[:]
                    not_i.pop(i)
                    self._explore(r, not_i)
                self._stack.pop()
                self._answer[var_id] = None


def test_for_overlap(border):
    """Test if the border crosses itself.
    This means there is an overlap or hole.
    """
    vertices = set()
    s = Cursor(0, 0, 0)
    for i, a in enumerate(border):
        s.advance(a)
        vertices.add((s.x, s.y))
        if len(vertices) != i + 1:
            return True
    return False


def verify_pieces(pieces):
    """Check the pieces for consistency."""
    for p in pieces:
        assert len(p) >= 3

        # The angles must be multiples of 60 degrees
        assert all(isinstance(a, int) and 1 <= a <= 5 for a in p)

        # Sum of angles for a polygon formula
        total = sum(p)
        expected = (len(p) - 2) * 3
        if total != expected:
            print("Bad piece", p)
            print(f"sum = {total}, expected = {expected}")
            raise RuntimeError

        if test_for_overlap(p):
            print("Overlapping piece", p)
            raise RuntimeError


def solve_puzzle(pieces):
    """The main program."""
    verify_pieces(pieces)
    write_pieces(pieces)

    variations = [[p, list(reversed(p))] for p in pieces[1:]]
    duplicates, solutions, tested, elapsed_time = Solver(
        pieces[0], variations
    ).solve()

    print(f"Time {elapsed_time*1.0e-9:.3f} seconds")
    print(f"{tested:,} candidates were tested")

    num_solutions = len(solutions)
    if num_solutions == 0:
        print("Found no solutions")
    elif num_solutions == 1:
        print("Found a single solution")
        display_answer(pieces[0], variations, solutions)
    else:
        print(f"Found {num_solutions} solutions")
        display_answer(pieces[0], variations, solutions)
    if duplicates > 1:
        print(f"There are {duplicates - 1} duplicates")


# The colors that are used to distinguish the different pieces in the
# output file.
COLORS = [
    colorsys.hls_to_rgb(hue / 360.0, 0.5, 0.7) for hue in [0, 60, 150, 270]
]


def generate_piece(angles, x, y, direction):
    """Convert a list of angles to a list of (x, y) coordinates.
    x and y determines the first coordinate.
    direction determines the rotation of the piece.
    """
    c = Cursor(x, y, direction)
    result = []
    for a in angles:
        c.advance(a)
        result.append([c.x, c.y])
    return result


def write_answer(answer, piece0, variations):
    """Return a Cairo surface representing a (possibly partial) answer."""

    result = [[generate_piece(piece0, 0, 0, 0), COLORS[0]]]
    for i, a in enumerate(answer):
        if a:
            flip_id, x, y, direction = a
            result.append(
                [
                    generate_piece(variations[i][flip_id], x, y, direction),
                    COLORS[i + 1],
                ]
            )
    return make_surface(result)


def write_pieces(pieces):
    """Write out the pieces in "pieces.png"""
    write_surfaces(
        "pieces.png",
        [
            make_surface([[generate_piece(p, 0, 0, 0), COLORS[i]]])
            for i, p in enumerate(pieces)
        ],
    )


def display_answer(piece0, variations, solutions):
    """Write out the answer in the file answer.png."""

    write_surfaces(
        "answer.png",
        [
            write_answer(s, piece0, variations)
            for i, s in enumerate(solutions)
            if i < 25
        ],
    )


# The height of an equilateral triangle with base 1.
ROOT3H = 0.5 * math.sqrt(3)


def make_surface(pieces):
    """Creates a Cairo surface containing a drawing of a solution."""
    # Use a recording surface to avoid having to calculate the bounds
    # of the image.  The background will be transparent.
    recording = cairo.RecordingSurface(cairo.FORMAT_ARGB32, None)
    ctx = cairo.Context(recording)
    ctx.scale(100, -100)

    def triscale(x, y):
        """Convert integral (x, y) coordinates to properly scaled
        values suitable for passing to Cairo.
        """
        return [0.5 * x, ROOT3H * y]

    for points, color in pieces:
        for i in range(2):
            ctx.move_to(*triscale(*points[0]))
            for v in points[1:]:
                ctx.line_to(*triscale(*v))
            ctx.close_path()

            if i == 1:
                ctx.set_line_width(0.025)
                ctx.set_source_rgb(0, 0, 0)  # black
                ctx.stroke()

            elif i == 0:
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


def write_surfaces(filename, surfaces):
    """Takes a list of Cairo surfaces and writes them all into a single
    .png file."""
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
    surface.write_to_png(filename)


if __name__ == "__main__":
    solve_puzzle(
        [
            [2, 1, 5, 1, 3, 1, 3, 2],
            [1, 3, 2, 1, 4, 2, 1, 4],
            [2, 1, 4, 1, 3, 1, 3],
            [2, 2, 1, 5, 2, 1, 3, 2],
        ]
    )

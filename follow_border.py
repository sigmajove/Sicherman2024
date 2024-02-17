""" This progam finds the solution to George Sicherman's 2024 New Year Puzzle
See https://sicherman.net/2024/2024.html
"""

import colorsys
import math
import time
from dataclasses import dataclass
import cairo


def _has_rotational_symmetry(piece, s):
    lp = len(piece)
    if s > lp or lp % s != 0:
        return None
    period = lp // s
    for i in range(period):
        val = piece[i]
        for j in range(1, s):
            if val != piece[i + j * period]:
                return None
    return period


def rotational_symmetry(piece):
    """A piece has rotational symmetry if it has the value
    n * P for some in integer n, and list P.
    If the piece has rotational symmetry, return len(P),
    otherwise len len(piece).
    """
    for s in [6, 3, 2]:
        period = _has_rotational_symmetry(piece, s)
        if period is not None:
            return (6 // s, period)
    return None


def has_mirror_symmetry(piece):
    """Return whether flipping the pieces leaves it the same modulo rotations."""
    mirrored = piece[:]
    mirrored.reverse()
    for _ in range(len(piece)):
        if mirrored == piece:
            return True
        mirrored = mirrored[1:] + [mirrored[0]]
    return False


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
        self.x += (2, 1, -1, -2, -1, 1)[self.direction]
        self.y += (0, 1, 1, 0, -1, -1)[self.direction]


def find_valley(piece):
    """Return the index of a valley in the piece, or None if there is not
    a valley.
    """
    return next((i for i, v in enumerate(piece) if 4 <= v <= 5), None)


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
    one_valley = find_valley(piece)
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


def splice_paths(path0, first0, last0, path1, first1, last1):
    """Splice the subpath of path0 from first0 to last0 with the subpath
    of path1 from from first1 to last1. The outputs at the two join points
    are the sum of the correpsonding inputs. The endpoints are inclusive,
    and may wrap around. The first element of the returned path will
    be path0[first0] + path1[last1].

    Example:
      path0 = [1, 2, 3, 4, 5, 6, 7]
      first0 = 5
      last0 = 1
      path1 = [10, 20, 30, 40, 50, 60, 70, 80, 90]
      first1 = 1
      last1 = 6
      splice_path returns [76, 7, 1, 22, 30, 40, 50, 60]
    """

    splice = [path0[first0] + path1[last1]]

    if first0 <= last0:
        splice += path0[first0 + 1 : last0]
    else:
        splice += path0[first0 + 1 :]
        splice += path0[:last0]

    splice.append(path0[last0] + path1[first1])

    if first1 <= last1:
        splice += path1[first1 + 1 : last1]
    else:
        splice += path1[first1 + 1 :]
        splice += path1[:last1]

    return splice


@dataclass(slots=True)
class StackFrame:
    """The type of Solver._stack[i]"""

    # The border of the the assembled pieces, expressed as a list of angles.
    border: list[int]

    # The position [x, y, direction] of border.
    position: tuple[int, int, int]


class EarlyExit(Exception):
    """For testing. Raised to abandon the search early."""


class Solver:
    """Solves George Sicherman's 2024 New Year Puzzle."""

    def __init__(self, pieces):
        # Each piece of the puzzle are described as the list of angles
        # at each vertex of the polygon. An angle is a value in range(1, 6)
        # expressed in units of 60 degrees.
        # For all but the first piece, we have a list of list of variations
        # to allow for fliping a piece.

        self._piece0 = pieces[0]

        # Add the mirror image unless the piece has mirror symmetry.
        self._variations = tuple(
            tuple([p] + ([] if has_mirror_symmetry(p) else [list(reversed(p))]))
            for p in pieces[1:]
        )
        self._rotational_symmetry = [rotational_symmetry(p) for p in pieces[1:]]

        # We implement the search tree using a stack.
        self._stack = [StackFrame(pieces[0], (0, 0, 0))]

        # An answer consists of an [flip, x, y, direction] for
        # each of the variations.  None means the piece is unplaced.
        self._answer = len(self._variations) * [None]

        # Used to record every solution we find.
        # It is a set to filter out duplicate solutions.
        self._solutions = set()

        self._duplicate_solutions = 0
        self._candidates_tested = 0

    def _border(self):
        """Return the border on top of the stack"""
        return self._stack[-1].border

    def _position(self):
        """Return the starting position for the border on top of stack."""
        return self._stack[-1].position

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
            for border_start in range(len(self._border())):
                for piece_finish in range(len(var[flip_id])):
                    if self._test_connection(
                        var_id=var_id,
                        flip_id=flip_id,
                        border_start=border_start,
                        piece_finish=piece_finish,
                        is_leaf=False,
                    ):
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

        try:
            iota = list(range(len(self._variations)))
            for i, var_id in enumerate(iota):
                not_i = iota[:]
                not_i.pop(i)
                self._explore(var_id, not_i)
                print(f"Finished {i} of {len(iota)}")
        except EarlyExit:
            pass

        elapsed_time = time.perf_counter_ns() - start_time

        return [
            self._duplicate_solutions,
            self._solutions,
            self._candidates_tested,
            elapsed_time,
        ]

    def display_answer(self):
        """Write out the answer in the file answer.png."""

        write_surfaces(
            "answer.png",
            [
                write_answer(s, self._piece0, self._variations)
                for i, s in enumerate(self._solutions)
                if i < 25
            ],
        )

        # Write all the answers to a text file.
        with open("answer.txt", "w", encoding="utf-8") as f:
            for s in self._solutions:
                f.write(str(s))
                f.write("\n")

    def _record_solution(self, var_id):
        self._solutions.add(tuple(tuple(x) for x in self._answer))
        self._duplicate_solutions += 1
        self._answer[var_id] = None

    def _check_hill_valley(self, var_id, flip_id, piece_join, border_join):
        """border[border_join] is a valley, and piece[piece_join] is
        a matching hill. Check whether it is possible to join the two
        at that point, and if so, record the solution.
        """
        piece = self._variations[var_id][flip_id]
        border = self._border()

        piece_next = piece_join
        border_prev = border_join
        while True:
            piece_next = (piece_next + 1) % len(piece)
            border_prev = (border_prev - 1) % len(border)
            joined = piece[piece_next] + border[border_prev]
            if joined < 6:
                break
            if joined > 6:
                # There is an overlap
                return

        piece_prev = piece_join
        border_next = border_join
        while True:
            piece_prev = (piece_prev - 1) % len(piece)
            border_next = (border_next + 1) % len(border)
            joined = piece[piece_prev] + border[border_next]
            if joined < 6:
                break
            if joined > 6:
                # There is an overlap
                return

        if self._test_candidate(
            var_id=var_id,
            flip_id=flip_id,
            splice=splice_paths(
                piece,
                piece_next,
                piece_prev,
                border,
                border_next,
                border_prev,
            ),
            border_start=border_prev,
            piece_finish=piece_next,
            is_leaf=True,
        ):
            self._record_solution(var_id)

    def _record_position(self, var_id, flip_id, cursor):
        """Record the position specified by cursor of the piece specified by
        var_id and flip_id. If the piece is symmetrical, normalize the position
        with respect to the symmetry of the piece.
        """
        sym = self._rotational_symmetry[var_id]
        self._answer[var_id] = (
            (flip_id, cursor.x, cursor.y, cursor.direction)
            if sym is None
            else (
                flip_id,
                *_normalize(
                    self._variations[var_id][flip_id],
                    sym,
                    (cursor.x, cursor.y, cursor.direction),
                ),
            )
        )

    def _test_candidate(
        self,
        var_id,
        flip_id,
        splice,
        border_start,
        piece_finish,
        is_leaf,
    ):
        border = self._border()
        piece = self._variations[var_id][flip_id]

        self._candidates_tested += 1

        if is_leaf and not all(angle <= 3 for angle in splice):
            # The final border is not convex.
            return False

        if test_for_overlap(splice):
            return False

        # Compute the positions.
        c = Cursor(*self._position())
        for v in border[:border_start]:
            c.advance(v)

        # c is now the position of the splice.
        if not is_leaf:
            # Push the splice onto the stack to be used by deeper levels.
            self._stack.append(StackFrame(splice, (c.x, c.y, c.direction)))
            # Caution: at this point border != self.border().

        c.advance(border[border_start] + piece[piece_finish])
        for v in piece[piece_finish + 1 :]:
            c.advance(v)

        # c is now the position where piece is placed.
        self._record_position(var_id, flip_id, c)

        return True

    def _test_connection(
        self, var_id, flip_id, border_start, piece_finish, is_leaf
    ):
        border = self._border()
        piece = self._variations[var_id][flip_id]

        limit = (6, 3)[is_leaf]
        if border[border_start] + piece[piece_finish] > limit:
            # There is an overlap.
            return False
        border_finish = border_start
        piece_start = piece_finish
        while True:
            border_finish = (border_finish + 1) % len(border)
            piece_start = (piece_start - 1) % len(piece)
            joined = border[border_finish] + piece[piece_start]
            if joined > limit:
                # There is an overlap.
                return False
            if joined != 6:
                break

        # splice is the border of the joined pieces. It begins with the
        # join of piece[piece_start] and border[border_finish]

        return self._test_candidate(
            var_id=var_id,
            flip_id=flip_id,
            splice=splice_paths(
                piece,
                piece_finish,
                piece_start,
                border,
                border_finish,
                border_start,
            ),
            border_start=border_start,
            piece_finish=piece_finish,
            is_leaf=is_leaf,
        )

    def _place_last_piece(self, var_id):
        """Try to place the last piece of the puzzle, which is var_id"""
        final_piece = self._variations[var_id]

        border = self._border()

        if valley_to_valley(border) > len(final_piece[0]):
            # The final piece is not big enough to reach
            # all the remaining valleys on the border.
            return

        valley_id = find_valley(border)
        if valley_id is None:
            # The unusual case that there are no valleys on the border.
            # We have to place the last piece somewhere on the border
            # without creating any new valleys.
            for flip_id, piece in enumerate(final_piece):
                for border_start in range(len(border)):
                    for piece_finish in range(len(piece)):
                        if self._test_connection(
                            var_id=var_id,
                            flip_id=flip_id,
                            border_start=border_start,
                            piece_finish=piece_finish,
                            is_leaf=True,
                        ):
                            self._record_solution(var_id)
        else:
            # There is at least one valley on the border.
            # The final piece must be placed to fill that valley.

            # The angle on the piece that must match the valley
            # on the border.
            hill = 6 - border[valley_id]

            for flip_id, piece in enumerate(final_piece):
                for piece_join, angle in enumerate(piece):
                    if angle == hill:
                        self._check_hill_valley(
                            var_id=var_id,
                            flip_id=flip_id,
                            piece_join=piece_join,
                            border_join=valley_id,
                        )

    def _explore(self, var_id, rest):
        """Examine all ways of attaching the piece specified by var_id
        to what is on the top of the stack, and then recursively examine
        all the var_ids in rest.
        """
        if rest:
            # Recursively grind through all the combinations.
            for _ in self._connect_all(var_id):
                for i, r in enumerate(rest):
                    not_i = rest[:]
                    not_i.pop(i)
                    self._explore(r, not_i)
                self._stack.pop()
                self._answer[var_id] = None
        else:
            self._place_last_piece(var_id)


def _normalize(piece, symmetry, position):
    """Given the position of a piece with rotational symmetry,
    normalize the position of that piece to use an approved angle.
    """
    max_dir, period = symmetry
    c = Cursor(*position)
    i = iter(piece)
    while c.direction >= max_dir:
        for _ in range(period):
            c.advance(next(i))
    return (c.x, c.y, c.direction)


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

        # Check that the piece is a closed loop
        c = Cursor(0, 0, 0)
        for a in p:
            c.advance(a)
        assert c.x == 0 and c.y == 0


def solve_puzzle(pieces):
    """The main program."""
    verify_pieces(pieces)
    write_pieces(pieces)

    solver = Solver(pieces)
    duplicates, solutions, tested, elapsed_time = solver.solve()

    print(f"Time {elapsed_time*1.0e-9:.3f} seconds")
    print(f"{tested:,} candidates were tested")

    num_solutions = len(solutions)
    if num_solutions == 0:
        print("Found no solutions")
    elif num_solutions == 1:
        print("Found a single solution")
        solver.display_answer()
    else:
        print(f"Found {num_solutions} solutions")
        solver.display_answer()
    if duplicates > 1:
        print(f"There are {duplicates - 1} duplicates")


# The colors that are used to distinguish the different pieces in the
# output file. We support at most five colors. Obviously that would be
# easy to fix if anyone cared.
COLORS = [
    colorsys.hls_to_rgb(hue / 360.0, 0.5, 0.7) for hue in [0, 60, 150, 270]
] + [(192.0 / 255, 192.0 / 255, 192.0 / 255)]


def generate_piece(angles, x, y, direction):
    """Convert a list of angles to a list of (x, y) coordinates.
    x and y determines the first coordinate.
    direction determines the rotation of the piece.
    """
    def generate():
        c = Cursor(x, y, direction)
        for a in angles:
            yield (c.x, c.y)
            c.advance(a)
    return tuple(generate())


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

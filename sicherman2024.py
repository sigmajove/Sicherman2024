""" This progam finds the solution to George Sicherman's 2024 New Year Puzzle
See https://sicherman.net/2024/2024.html
"""

import math
from PIL import Image, ImageDraw, ImageFont
import tqdm

# I need a coordinate system for moving triangles around the plane.
# I choose a triangular tiling with the Archimedian coloring 121212, see
# https://en.wikipedia.org/wiki/Triangular_tiling#/media/File:Uniform_triangular_tiling_121212.png
# We distinguish between the up-pointing triangles (colored red in the diagram)
# and the down-pointing triangles (colored yellow in the diagram).
# We can designate each triangular cell with integer coordinates(x, y), where
# y determines the horizontal row, and x selects a triangle from that row.
# We define (0, 0) to be the origin, and arbitrarily define it to
# be up-pointing.

# Up-pointing triangles are adjacent to three others, which I call
# NORTHEAST, NORTHWEST, and SOUTH. Down-pointing triangles also
# have three adjacent triangles, which I call NORTH, SOUTHEAST, and SOUTHWEST.
NORTH = 0
NORTHWEST = 1
SOUTHWEST = 2
SOUTH = 3
SOUTHEAST = 4
NORTHEAST = 5

NUM_DIRECTIONS = 6

DIRECTION_NAME = [
    "north",
    "northwest",
    "southwest",
    "south",
    "southeast",
    "northeast",
]

# The opposite direction.
# NORTH -> SOUTH, SOUTHWEST -> NORTHEAST, etc.
OPPOSITE_DIRECTION = [SOUTH, SOUTHEAST, NORTHEAST, NORTH, NORTHWEST, SOUTHWEST]

# A right turn
CLOCKWISE_DIRECTION = [NORTHWEST, SOUTHWEST, SOUTH, SOUTHEAST, NORTHEAST, NORTH]

# A left turn
COUNTERCLOCKWISE_DIRECTION = [
    NORTHEAST,
    NORTH,
    NORTHWEST,
    SOUTHWEST,
    SOUTH,
    SOUTHEAST,
]


def turn_angle(d0, d1):
    """Describe the turn we make when we change direction from d0 to d1.
    Here are the different results:
      0:    no turn (straight)
      1:    60 degree left turn
      2:    120 degree left turn
      -1:   60 degree right turn
      -2:   120 degree right turn
      None: U-turn
    """
    return turn_angle.TURNING_POINTS[d1 - d0 + 5]


turn_angle.TURNING_POINTS = [-2, -1, None, 1, 2, 0, -2, -1, None, 1, 2]


# Puzzle pieces are represented by lists, or tuples where appropriate,
# of cell tuples of the form ((x, y), p). The values x and y are integer
# coordinates, as described above. The value p is in range(0, 4), and
# represents from which piece the cell originally came from. As we build
# bigger pieces by sticking together smaller ones, the piece numbers let
# us parse the final result.


def points_up(x, y):
    """Return whether the triangle at (x, y) is up-pointing."""
    return (x + y) % 2 == 0


def vertex_id(x, y, direction):
    """We can identify a vertex with the coordinates of a cell,
    and a direction that points from the center of the cell to the vertex.
    But using this convention, a vertex can have three different names.
    Instead use the convention it is the NORTH vertex of a cell. That
    name is unique.
    """
    dx, dy = vertex_id.VERTEX_ID_TABLE[direction]
    return (x + dx, y + dy)


vertex_id.VERTEX_ID_TABLE = [
    (00, 00),  # NORTH
    (-1, 00),  # NORTHWEST
    (-1, +1),  # SOUTHWEST
    (00, +1),  # SOUTH
    (+1, +1),  # SOUTHEAST
    (+1, 00),  # NORTHEAST
]


def adjacent_cell(x, y, direction):
    """Return the coordinates of the adjacent cell in the given
    direction. Raises an exception we cannot go in that direction from (x, y).
    """
    if points_up(x, y):
        # Upward pointing triangle
        if direction == NORTHEAST:
            return (x + 1, y)
        if direction == NORTHWEST:
            return (x - 1, y)
        if direction == SOUTH:
            return (x, y + 1)
    else:
        # Downward pointing triangle
        if direction == SOUTHEAST:
            return (x + 1, y)
        if direction == SOUTHWEST:
            return (x - 1, y)
        if direction == NORTH:
            return (x, y - 1)

    raise RuntimeError(f"Cannot move {DIRECTION_NAME[direction]}")


def adjacent(x, y):
    """Generate information about the three cells that are adjacent to (x, y).
    For each adjacent cell, returns [(x, y), direction]
    """
    if points_up(x, y):
        yield [(x + 1, y), NORTHEAST]
        yield [(x - 1, y), NORTHWEST]
        yield [(x, y + 1), SOUTH]
    else:
        yield [(x + 1, y), SOUTHEAST]
        yield [(x - 1, y), SOUTHWEST]
        yield [(x, y - 1), NORTH]


def vertices(x, y):
    """Generate the three floating point coordinates of the vertices
    of the triangle. Intended for use with drawing packages.
    """
    side = 100  # width of a trianglar cell in pixels

    half_side = 0.5 * side
    center = half_side * x
    left = center - half_side
    right = center + half_side

    triangle_height = math.sqrt(3) * half_side
    top = triangle_height * (y - 1)
    bottom = triangle_height * y

    if points_up(x, y):
        yield (center, top)
        yield (right, bottom)
        yield (left, bottom)
    else:
        yield (left, top)
        yield (right, top)
        yield (center, bottom)


def normalize(cells):
    """Return a canonical, hashable, representation of a list of cells.
    A normalized tuple has three properties:
      (1) it is sorted.
      (2) The first tuple, if there is one, has x == 0 or x == 1.
      (3) The first tuple, if there is one, has y == 0.
    """
    min_x = min(x for (x, _), __ in cells)
    min_y = min(y for (_, y), __ in cells)

    # Pieces cannot be translated arbitrarily.
    # if min_x + min_y is odd, the translation converts up-pointing
    # triangles to down-pointing triangles, and vice versa with disasterous
    # results.
    if (min_x + min_y) % 2 == 1:
        min_x -= 1

    renumbered = [((x - min_x, y - min_y), p) for (x, y), p in cells]
    renumbered.sort()
    return tuple(renumbered)


class PieceScanner:
    """A worker class that can compoute different metrics for a piece.
    Since count_dents_or_points and is_convex are intrusive, at most
    one of them should be called.
    """

    def __init__(self, piece):
        """Find the boundary of the piece.  For each instance of two adjacent
        cells (inside, outside) where inside is in the piece but outside is not,
        store the border between them in self._border_map.
        Set self._early_exit if we detect a hole.
        """

        in_piece = {cell for cell, _ in piece}
        self._border_map = {}
        for (x, y), _ in piece:
            for adj, direction in adjacent(x, y):
                if adj not in in_piece:
                    # Store the border
                    key = vertex_id(x, y, COUNTERCLOCKWISE_DIRECTION[direction])
                    if key in self._border_map:
                        # The border intersects itself, which can only happen
                        # if there is a hole in the piece.
                        self._early_exit = True
                        return
                    self._border_map[key] = (
                        vertex_id(x, y, CLOCKWISE_DIRECTION[direction]),
                        direction,
                    )
        self._early_exit = False

    def count_dents_or_points(self, dents_not_points):
        """Return the number of dents or points in the piece.
        dents_not_points is a boolean that selects which feature we count.
        Return None if the piece has a hole.
        """
        if self._early_exit:
            # There is a hole.
            return None
        if not self._border_map:
            return 0

        # Nondestructively select an arbitrary starting point.
        # We can't count start yet, since we don't have its predecessor.
        start = next(iter(self._border_map.values()))
        total = 0
        vertex = start
        while True:
            # Delete and count every vertex in the loop.
            next_vertex = self._border_map.pop(vertex[0])
            turns = turn_angle(vertex[1], next_vertex[1])
            total += max(0, turns if dents_not_points else -turns)
            vertex = next_vertex

            # When we have counted every vertex we are done.
            if vertex == start:
                break
        # if there are any vertices left in _border_map, we have a hole.
        return None if self._border_map else total

    def is_convex(self):
        """Return whether the piece has no holes or dents.
        Equivalent to count_dents_or_points(self, dents_not_points=True) == 0,
        but it is faster because it stops when it encounters the first dent.
        """
        if self._early_exit:
            return False

        if not self._border_map:
            return True

        start = next(iter(self._border_map.values()))
        vertex = start
        while True:
            next_vertex = self._border_map.pop(vertex[0])
            if turn_angle(vertex[1], next_vertex[1]) > 0:
                return False
            vertex = next_vertex
            if vertex == start:
                break
        return not self._border_map


class Piece:
    """
    One of the original pieces of the puzzle.

    Attributes:
        points (int): metric for trimming the search.
        variations (list[list[((x, y), id)]]:
            all the ways the piece can be rotated and reflected.
    """

    def __init__(self, piece_id, picture):
        cells = make_cells(piece_id, picture)
        points = PieceScanner(cells).count_dents_or_points(
            dents_not_points=False
        )
        # Puzzle pieces should not have holes.
        assert points is not None
        self.points = points
        self.variations = Piece._make_variations(cells)

    @staticmethod
    def _rotate_cell(x, y):
        """Return the cell rotated 60 degrees (counterclockwise) around the
        north vertex of the triangle with coordinates (0, 0).
        """
        # It took me a couple of days to discover this formula.
        # I don't have a simple explanation for why it works.
        # The complicated explanation involves a lot of trigonometry
        # and algebra.
        return (y - (-x - y - 1) // 2, (y - x) // 2)

    @staticmethod
    def _rotate(piece):
        """Rotates a piece 60 degrees counterclockwise"""
        return normalize([[Piece._rotate_cell(x, y), p] for (x, y), p in piece])

    @staticmethod
    def _flip(piece):
        """Creates a mirror image of a piece by flipping it on the y axis"""
        return normalize([[(-x, y), p] for (x, y), p in piece])

    @staticmethod
    def _make_variations(cells):
        """Return a list of the twelve variations of the piece, considering
        rotations and reflections. There will be no duplicates in this list
        because none of the puzzle pieces are symmetrical.
        """
        result = [cells]
        for _ in range(5):
            result.append(Piece._rotate(result[-1]))
        result.append(Piece._flip(result[-1]))
        for _ in range(5):
            result.append(Piece._rotate(result[-1]))
        return result


def make_cells(piece_id, picture):
    """Read an ASCII Art representation of a piece, and return the
    list of tuples that represents that piece.  We don't check that the
    artwork is well-formed; we just scan it for key patterns to get the
    proper x, y coordinates.
    """
    result = []
    i = 1
    y = 0
    while i < len(picture):
        text = picture[i]
        # Look for down-pointing triangles.
        x = 1
        j = 0
        while j + 4 < len(text):
            if (
                text[j : j + 4] == "\\  /"
                and picture[i - 1][j + 1 : j + 3] == "__"
            ):
                assert not points_up(x, y)
                result.append(((x, y), piece_id))
            j += 2
            x += 1
        # Look for up-pointing triangles.
        x = 1
        j = 1
        while j + 2 < len(text):
            if text[j : j + 2] == "/\\" and picture[i + 1][j : j + 2] == "__":
                assert points_up(x, y)
                result.append(((x, y), piece_id))
            j += 2
            x += 1
        i += 2
        y += 1
    return normalize(result)


# The four pieces of the puzzle.
# We do not create any variations for P0 to avoid reporting solutions that
# are rotations or reflections of one another.
P0 = make_cells(
    0,
    [
        r" __  __    ",
        r"\  /\  /   ",
        r" \/__\/__  ",
        r"  \  /\  / ",
        r"   \/__\/  ",
    ],
)
P1 = Piece(
    1,
    [
        r" __      ",
        r"\  /\    ",
        r" \/__\   ",
        r" /\  /\  ",
        r"/__\/__\ ",
        r"\  /     ",
        r" \/      ",
    ],
)

P2 = Piece(
    2,
    [
        r"           ",
        r"   /\      ",
        r"  /__\ __  ",
        r" /\  /\  / ",
        r"/__\/__\/  ",
    ],
)

P3 = Piece(
    3,
    [
        r" __        ",
        r"\  /\  /\  ",
        r" \/__\/__\ ",
        r"  \  /\  / ",
        r"   \/__\/  ",
    ],
)


def border_table(piece):
    """Locate all the cells in this piece that are adjacent to one or
    more cells not in the piece. Return the results as a table. For each
    direction d, border[d] contains the list of cells that are adjacent
    to a non-piece cell in the direction d.
    """
    result = [[] for _ in range(NUM_DIRECTIONS)]

    # The set of xy coordinates in this piece
    in_piece = {e for e, _ in piece}

    for e, _ in piece:
        for xy, direction in adjacent(*e):
            if xy not in in_piece:
                result[direction].append(e)

    return result


def contains_duplicate(cells):
    """Sorts cells and returns whether it contains duplicate coordinates."""
    if not cells:
        return False
    cells.sort()
    prev = cells[0]
    for p in cells[1:]:
        if prev[0] == p[0]:
            return True
        prev = p
    return False


def remove_duplicates(cells):
    """Sorts cells and filters out any duplicates."""
    if not cells:
        return cells

    cells.sort()
    prev = cells[0]
    result = [prev]
    for p in cells[1:]:
        if prev != p:
            result.append(p)
        prev = p

    return result


class Finder:
    """Contains the algorithm for solving the puzzle"""

    def __init__(self):
        # The process for generating candidates generates duplicates.
        # Since testing the candidates is expensive, we use this table
        # so we test each unique candidate only once.
        self._tested = set()

        self._point_prunes = 0
        self._hole_prunes = 0

        # Collects the solutions we find.
        self._solutions = []

        # Reports progress to the goal.
        self._progress = tqdm.tqdm(unit=" candidates")

    def evaluate_candidates(self, candidates):
        """Tests all the candidates and adds the convex ones to
        self._solutions.  Ignores any candidates we have already tested.
        """
        for candidate in candidates:
            normalized = normalize(candidate)
            if normalized not in self._tested:
                self._tested.add(normalized)
                if PieceScanner(normalized).is_convex():
                    self._solutions.append(normalized)

        self._progress.update(len(candidates))

    def _is_viable_piece(self, joined, point_limit):
        """Determine whether the newly joined piece satisifies
        all the restrictions. As a side effect, sorts joined.
        """

        if contains_duplicate(joined):
            return False

        if point_limit is None:
            # Don't bother with any more checking.
            # That will be done in evaluate_candidates.
            return True

        # Count the number of dents in joined.
        dent_count = PieceScanner(joined).count_dents_or_points(
            dents_not_points=True
        )
        if dent_count is None:
            # There is a hole
            self._hole_prunes += 1
            return False

        # Only continue if there are enough points left in the remaining
        # pieces to fill all  the existing dents.
        if dent_count > point_limit:
            self._point_prunes += 1
            return False

        return True

    def find_joins(self, big_piece, other, point_limit):
        """Find the ways big_piece and other.variations can be stuck together
        to form a new piece. Return the list of all possible joins.
        """
        new_pieces = []
        bt0 = border_table(big_piece)
        for piece1 in other.variations:
            bt1 = border_table(piece1)

            for direction in range(NUM_DIRECTIONS):
                for x0, y0 in bt0[direction]:
                    for x1, y1 in bt1[OPPOSITE_DIRECTION[direction]]:
                        # The two other can by stuck together at cell0 and
                        # cell1.  We will renumber piece1 so that cell1 has
                        # the coordinates (x, y)

                        x2, y2 = adjacent_cell(x0, y0, direction)
                        dx = x2 - x1
                        dy = y2 - y1

                        # Start with a copy of big_piece.
                        joined = [*big_piece]

                        # Append all the cells of piece1, adjusting their
                        # coordinates to be consistent with big_piece.
                        for (x, y), p in piece1:
                            joined.append(((x + dx, y + dy), p))

                        if self._is_viable_piece(joined, point_limit):
                            new_pieces.append(joined)

        # The join algorithm can produce duplicate pieces.
        # Detect and remove them.
        return remove_duplicates(new_pieces)

    def assemble_pieces(self, big_piece, pieces):
        """Called with a list of pieces, where each piece is list [x, y, p]
        where x and y are the position, and p the id of the original piece.
        """
        if len(pieces) == 1:
            self.evaluate_candidates(
                self.find_joins(big_piece, pieces[0], point_limit=None)
            )
        else:
            for i, piece_i in enumerate(pieces):
                omit_i = pieces[:i] + pieces[i + 1 :]
                for join in self.find_joins(
                    big_piece,
                    piece_i,
                    point_limit=sum(p.points for p in omit_i),
                ):
                    self.assemble_pieces(join, omit_i)

    def find_solutions(self):
        """The main entry point of the class."""
        self.assemble_pieces(P0, [P1, P2, P3])
        self._progress.close()

        print(f"Pruned {self._point_prunes} for points")
        print(f"Pruned {self._hole_prunes} for holes")
        n = len(self._solutions)
        print(f"Found {n} solution{'' if n==1 else 's'}")

        # Write the solutions into the file "answer.png".
        # There should be only one solution, but bugs have been
        # known to produce thousands.  Only write out the first five.
        if self._solutions:
            self._solutions = self._solutions[:5]
            answer = list(map(draw_piece, self._solutions))
            combined = show_images(answer)
            combined.save("answer.png", "PNG")
            combined.close()
            for a in answer:
                a.close()


# The colors for the pieces are chosen somewhat subjectively.
COLORS = [f"hsl({hue}, 70%, 50%)" for hue in [0, 60, 150, 270]]


def draw_piece(cells):
    """Draws a list of cells, returning a PIL image."""
    min_x = math.inf
    max_x = -math.inf
    min_y = math.inf
    max_y = -math.inf

    for (x, y), _ in cells:
        for vx, vy in vertices(x, y):
            min_x = min(min_x, vx)
            max_x = max(max_x, vx)
            min_y = min(min_y, vy)
            max_y = max(max_y, vy)

    width = int(math.ceil(1 + max_x - min_x))
    height = int(math.ceil(1 + max_y - min_y))
    im = Image.new("RGB", (width, height), color="white")
    d = ImageDraw.Draw(im)

    for (x, y), p in cells:
        edges = [((vx - min_x), (vy - min_y)) for vx, vy in vertices(x, y)]
        d.polygon(edges, fill=COLORS[p], outline="white")

    return im


def fingerprint(piece):
    """Return an eight-digit hash code to identify the piece.
    For debugging.
    """
    return hash(normalize(piece)) % 100000000


FONT = ImageFont.truetype("arial.ttf", size=15)


def map_piece(cells):
    """For debugging.
    Return an image of the piece annotated with the coordinates of each cell,
    and a hashed fingerprint for the piece.
    """
    min_x = math.inf
    max_x = -math.inf
    min_y = math.inf
    max_y = -math.inf

    for (x, y), _ in cells:
        for vx, vy in vertices(x, y):
            min_x = min(min_x, vx)
            max_x = max(max_x, vx)
            min_y = min(min_y, vy)
            max_y = max(max_y, vy)

    width = int(math.ceil(1 + max_x - min_x))
    height = int(math.ceil(1 + max_y - min_y) + 22)
    im = Image.new("RGB", (width, height), color="white")
    d = ImageDraw.Draw(im)

    for (x, y), p in cells:
        edges = [((vx - min_x), (vy - min_y)) for vx, vy in vertices(x, y)]
        d.polygon(edges, fill=COLORS[p], outline="white")

        av_x = sum(x for x, _ in edges) / len(edges)
        av_y = sum(y for _, y in edges) / len(edges)
        if not points_up(x, y):
            av_y -= 3
        text = f"{x} {y}"
        d.text((av_x - 13, av_y), text, font=FONT, fill="white")

    d.text(
        (3, height - 21), f"{fingerprint(cells):08}", font=FONT, fill="black"
    )

    return im


def show_images(images):
    """Combines a list of images into a single image and returns it."""
    separator = 16  # Number of pixels between images

    # compute the size of the joined image
    height = sum(im.height for im in images) + separator * (len(images) - 1)
    width = max(im.width for im in images)
    im = Image.new("RGB", (width, height), color="white")
    d = ImageDraw.Draw(im)
    pos = 0
    for p in images:
        Image.Image.paste(im, p, (0, pos))
        pos += p.height
        p.close()
        y = pos + separator / 2
        d.line([0, y, width - 1, y], fill="black", width=2)
        pos += separator
    return im


if __name__ == "__main__":
    Finder().find_solutions()

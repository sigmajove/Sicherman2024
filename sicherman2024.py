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


# Puzzle pieces are represented by lists, or tuples where appropriate,
# of cell tuples of the form ((x, y), p). The values x and y are integer
# coordinates, as described above. The value p is in range(0, 4), and
# represents from which piece the cell originally came from. As we build
# bigger pieces by sticking together smaller ones, the piece numbers let
# us parse the final result.


def points_up(x, y):
    """Return whether the triangle at (x, y) is up-pointing."""
    return (x + y) % 2 == 0


def adjacent_cell(xy, direction):
    """Return the (x, y) coordinates of the adjacent cell in the given
    direction. Raises an exception we cannot go in that direction from xy.
    """
    x, y = xy
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


def adjacent(xy):
    """Generate information about the three cells that are adjacent to xy.
    For each adjacent cell, returns [(x, y), direction]
    """
    x, y = xy
    if points_up(x, y):
        yield [(x + 1, y), NORTHEAST]
        yield [(x - 1, y), NORTHWEST]
        yield [(x, y + 1), SOUTH]
    else:
        yield [(x + 1, y), SOUTHEAST]
        yield [(x - 1, y), SOUTHWEST]
        yield [(x, y - 1), NORTH]


def vertices(xy):
    """Return the three floating point coordinates of the vertices of the
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


def normalize(cells):
    """Return a canonical, hashable, representation of a list of cells."""
    min_x = min(c[0][0] for c in cells)
    min_y = min(c[0][1] for c in cells)

    # Pieces cannot be translated arbitrarily.
    # if min_x + min_y is odd, the translation converts up-pointing
    # triangles to down-pointing triangles, and vice versa with disasterous
    # results.
    if (min_x + min_y) % 2 == 1:
        min_x -= 1

    renumbered = [((c[0][0] - min_x, c[0][1] - min_y), c[1]) for c in cells]
    renumbered.sort()
    return tuple(renumbered)


def rotate_cell(xy):
    """Return the cell rotated 60 degrees (counterclockwise) around the
    north vertex of the triangle with coordinates (0, 0).
    """
    x, y = xy
    # It took me a couple of days to discover this formula.
    # I don't have a simple explanation for why it works.
    # The complicated explanation involves a lot of trigonometry and algebra.
    return (y - (-x - y - 1) // 2, (y - x) // 2)


def mirror_cell(xy):
    """Return the cell reflected about the y-axis."""
    x, y = xy
    return (-x, y)


def rotate(piece):
    """Rotates a piece 60 degrees counterclockwise"""
    return normalize([[rotate_cell(p[0]), p[1]] for p in piece])


def flip(piece):
    """Creates a mirror image of a piece by flipping it on the y axis"""
    return normalize([[mirror_cell(p[0]), p[1]] for p in piece])


def find_border(piece, report_border_cell):
    """Use depth-first search to find the boundary of the piece.
    For each instance of two adjacent cells (inside, outside) where inside
    is in the piece but outside is not, call report_border_cell(inside, d),
    where is the direction from inside to outside. report_border_cell
    is never called more than once with the same arguments.
    """
    if len(piece) == 0:
        return set()

    visited = set()
    in_piece = set(p[0] for p in piece)
    examine = {piece[0][0]}
    while examine:
        cell = examine.pop()
        visited.add(cell)
        for adj, direction in adjacent(cell):
            if adj in in_piece:
                if adj not in visited:
                    examine.add(adj)
            else:
                report_border_cell(cell, direction)
    return in_piece


def vertex_id(x, y, direction):
    """We can identify a vertex with the coordinates of a cell,
    and a direction that points from the center of the cell to the vertix.
    But using this convention, a vertex can have three different names.
    Instead use the convention it is the NORTH vertix of a cell. That
    name is unique.
    """
    if direction == NORTH:
        return (x, y)
    if direction == NORTHEAST:
        return (x + 1, y)
    if direction == NORTHWEST:
        return (x - 1, y)
    if direction == SOUTHEAST:
        return (x + 1, y + 1)
    if direction == SOUTHWEST:
        return (x - 1, y + 1)
    if direction == SOUTH:
        return (x, y + 1)
    raise ValueError(f"{direction} is not a direction")


def how_many_holes(piece):
    """Return the number of internal holes in the piece. The algorithm only
    cares about zero vs nonzero. We return the number to increase testing
    confidence. But be forwarned that the code consideres two holes that
    touch at a diagonal to be a single hole.
    """
    border_map = {}
    border_touches = False

    def report_border_cell(b, direction):
        nonlocal border_touches
        # Store the border
        x, y = b
        key = vertex_id(x, y, COUNTERCLOCKWISE_DIRECTION[direction])
        if key in border_map:
            border_touches = True
        else:
            border_map[key] = vertex_id(x, y, CLOCKWISE_DIRECTION[direction])

    find_border(piece, report_border_cell)

    if border_touches:
        # Wierd corner case.
        # There is a hole, but it touches the border.
        # In this case, the algorithm cannot detect the number of holes.
        # Lie. If there is a hole, we don't actually care how many there are.
        return 1

    # border_map now contains all the border edges, for both the
    # piece and any holes. We count the loops and subtract one.
    # Unless there are no loops, which can only happen if the
    # piece is empty.

    num_loops = -1
    while border_map:
        start, next_one = border_map.popitem()
        while next_one != start:
            next_one = border_map.pop(next_one)
        num_loops += 1

    return max(0, num_loops)


def how_many_points(piece):
    """Return the number of acute angles on the piece.  Return one point
    for every 60 degree angle and two points for each 120 degree angle.
    This function is only called for three of the four puzzle pieces,
    so we don't worry about holes.
    """
    # For each cell on the border, counts the edges (at least one)
    # that are adjacent to an outside cell.
    border = {}

    def report_border_cell(b, _):
        border[b] = border.get(b, 0) + 1

    in_piece = find_border(piece, report_border_cell)

    # Now count the number of points.
    points = sum(1 if v == 2 else 3 if v == 3 else 0 for v in border.values())
    for b in border:
        for a, a_dir in adjacent(b):
            if a in border:
                # b and a are two adjacent border cells.
                # Usually this means the border turns 120 degrees.
                # Unfortunately, there is a weird corner case we
                # have to rule out.
                b_dir = OPPOSITE_DIRECTION[a_dir]
                b_turn_1 = CLOCKWISE_DIRECTION[b_dir]
                b_turn_2 = COUNTERCLOCKWISE_DIRECTION[b_dir]
                a_turn_1 = COUNTERCLOCKWISE_DIRECTION[a_dir]
                a_turn_2 = CLOCKWISE_DIRECTION[a_dir]
                if (
                    adjacent_cell(a, a_turn_1) not in in_piece
                    and adjacent_cell(b, b_turn_1) not in in_piece
                ) or (
                    adjacent_cell(a, a_turn_2) not in in_piece
                    and adjacent_cell(b, b_turn_2) not in in_piece
                ):
                    points += 1
    return points


def how_convex(piece):
    result = 0
    # We reject a polygon if one of the angles is 300 degrees or 240 degrees.
    # We use depth-first search to find all the border cells that surround the
    # piece. We detect the bad angles by examining those border cells

    # Check for vacuous case.
    if len(piece) == 0:
        return 0

    # The map keys are cells in the piece that are adjacent to one or more
    # cells not in the piece. The value of the map is the direction from
    # the border cell to an adjacent cell in the piece.
    surrounding = {}

    # Use depth-first search to populate surrounding.
    visited = set()
    in_piece = set(p[0] for p in piece)
    examine = {piece[0][0]}
    while examine:
        e = examine.pop()
        visited.add(e)
        for adj, _ in adjacent(e):
            if adj in in_piece:
                if not adj in visited:
                    examine.add(adj)
            else:
                if adj in surrounding:
                    # If a border cell is adjacent to two cells in the piece,
                    # there must be a 300 degree vertex.
                    result += 1
                new_value = surrounding.get(adj, 0) + 1
                surrounding[adj] = new_value
                if new_value == 3:
                    # A triangular shaped hole
                    result += 1

    # Next we detect 240 degree vertices by examining all pairs of
    # adjacent border cells.  We are looking for this pattern:
    #
    #      #####/\
    #      ####/  \
    #      ###/    \
    #      ##/      \
    #      #/________\
    #      #\        /
    #      ##\      /
    #      ###\    /
    #      ####\  /
    #      #####\/
    #
    # which contains a 240 degree vertex.
    #
    # We ignore this pattern:
    #
    #      #####/\
    #      ####/  \
    #      ###/    \
    #      ##/      \
    #      #/________\
    #       \        /#
    #        \      /##
    #         \    /###
    #          \  /####
    #           \/#####
    #

    for b in surrounding:
        for a, a_dir in adjacent(b):
            if a in surrounding:
                b_dir = OPPOSITE_DIRECTION[a_dir]
                b_turn_1 = CLOCKWISE_DIRECTION[b_dir]
                b_turn_2 = COUNTERCLOCKWISE_DIRECTION[b_dir]
                a_turn_1 = COUNTERCLOCKWISE_DIRECTION[a_dir]
                a_turn_2 = CLOCKWISE_DIRECTION[a_dir]
                if (
                    adjacent_cell(a, a_turn_1) in in_piece
                    and adjacent_cell(b, b_turn_1) in in_piece
                ) or (
                    adjacent_cell(a, a_turn_2) in in_piece
                    and adjacent_cell(b, b_turn_2) in in_piece
                ):
                    result += 1

    return result


def make_variations(cells):
    """Return a list of the twelve variations of the piece, considering
    rotations and reflections. There will be no duplicates in this list
    because none of the puzzle pieces are symmetrical.
    """
    result = [cells]
    for _ in range(5):
        result.append(rotate(result[-1]))
    result.append(flip(result[-1]))
    for _ in range(5):
        result.append(rotate(result[-1]))
    return result


class Piece:
    """
    A piece that gets added to the puzzle.

    Attributes:
        points (int): metric for trimming the search.
        variations (list[list[((x, y), id)]]:
            all the ways the piece can be rotated and reflected.
    """

    def __init__(self, piece_id, picture):
        cells = make_cells(piece_id, picture)
        self.points = how_many_points(cells)
        self.variations = make_variations(cells)


def make_cells(piece_id, picture):
    """Reads an ASCII Art representation of a piece, and returns the
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
                result.append(((x, y), piece_id))
            j += 2
            x += 1
        # Look for up-pointing triangles.
        x = 1
        j = 1
        while j + 2 < len(text):
            if text[j : j + 2] == "/\\" and picture[i + 1][j : j + 2] == "__":
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

# P2 = Piece(2, [(1, 1), (2, 0), (2, 1), (3, 1), (4, 1)])
# P3 = Piece(3, [(1, 0), (2, 0), (2, 1), (3, 1), (4, 0), (4, 1)])

# The colors for the pieces are chosen somewhat subjectively.
COLORS = [f"hsl({hue}, 70%, 50%)" for hue in [0, 60, 150, 270]]


def border_table(piece):
    """Locate all the cells in this piece that are adjacent to one or
    more cells not in the piece. Return the results as a table. For each
    direction d, border[d] contains the list of cells that are adjacent
    to a non-piece cell in the direction d.
    """
    result = [[] for _ in range(NUM_DIRECTIONS)]

    if len(piece) != 0:
        # Use depth-first search to populate result.
        visited = set()
        in_piece = set(p[0] for p in piece)
        examine = {piece[0][0]}
        while examine:
            e = examine.pop()
            visited.add(e)
            for xy, direction in adjacent(e):
                if xy in in_piece:
                    if not xy in visited:
                        examine.add(xy)
                else:
                    result[direction].append(e)

    return result


def contains_duplicate(piece):
    """Sorts piece and returns whether it contains duplicate coordinates."""
    if not piece:
        return False
    piece.sort()
    prev = piece[0]
    for p in piece[1:]:
        if prev[0] == p[0]:
            return True
        prev = p
    return False


def find_joins(piece0, pieces, point_limit):
    """Find the ways piece0 and pieces can be stuck together to form a new
    piece. Return the list of all possible joins.
    """
    new_pieces = []
    bt0 = border_table(piece0)
    for piece1 in pieces.variations:
        bt1 = border_table(piece1)

        for direction in range(NUM_DIRECTIONS):
            for cell0 in bt0[direction]:
                for cell1 in bt1[OPPOSITE_DIRECTION[direction]]:
                    # The two pieces can by stuck together at cell0 and cell1.

                    # We will renumber piece1 so that cell1 has the
                    # coordinates (x, y)

                    x, y = adjacent_cell(cell0, direction)
                    dx = x - cell1[0]
                    dy = y - cell1[1]

                    joined = [*piece0]

                    # Append all the cells of piece1, adjusting the coordinates
                    # to be consistent with piece0.
                    for p in piece1:
                        joined.append(((p[0][0] + dx, p[0][1] + dy), p[1]))

                    if (
                        not contains_duplicate(joined)
                        and how_many_holes(joined) == 0
                        and (
                            point_limit is None
                            or how_convex(joined) <= point_limit
                        )
                    ):
                        new_pieces.append(joined)

    if not new_pieces:
        return []

    new_pieces.sort()
    prev = new_pieces[0]
    result = [prev]
    for p in new_pieces[1:]:
        if prev != p:
            result.append(p)
        prev = p
    return result


def has_triangular_hole(piece):
    # Check for vacuous case.
    if len(piece) == 0:
        return False

    surrounding = {}

    # Use depth-first search to populate surrounding.
    visited = set()
    in_piece = set(p[0] for p in piece)
    examine = {piece[0][0]}
    while examine:
        e = examine.pop()
        visited.add(e)
        for adj, _ in adjacent(e):
            if adj in in_piece:
                if not adj in visited:
                    examine.add(adj)
            else:
                new_val = surrounding.get(adj, 0) + 1
                if new_val >= 3:
                    return True
                surrounding[adj] = new_val
    return False


class Finder:
    """Class wrapper to allow assemble_pieces to have global variables."""

    def __init__(self):
        # The process for generating candidates generates duplicates.
        # Since testing the candidates is expensive, we use this table
        # so we test each unique candidate only once.
        self._tested = set()

        # The number of candidates we tested with how_convex
        self._num_candidates = 0

        self._alert_every = 100_000
        self._solutions = []

        self._level = 0
        self._stop = False
        self._progress = tqdm.tqdm(unit=" candidates")

    def evaluate_candidates(self, candidates):
        for candidate in candidates:
            hashable = normalize(candidate)
            if hashable not in self._tested:
                self._tested.add(hashable)

                # If the candidate is convex, add it to the list of solutions.
                # We don't need an explicit check for holes, since any pieces
                # with a hole will fail to how_convex check.
                if how_convex(candidate) == 0:
                    self._solutions.append(candidate)
            self._num_candidates += 1

        self._progress.update(len(candidates))

    # Called with a list of pieces, where each piece is list [x, y, p]
    # where x and y are the position, and i the id of the original piece.
    def assemble_pieces(self, big_piece, pieces):
        self._level += 1
        if len(pieces) == 1:
            self.evaluate_candidates(find_joins(big_piece, pieces[0], None))
        else:
            for i, piece_i in enumerate(pieces):
                omit_i = pieces[:i] + pieces[i + 1 :]
                point_limit = sum(p.points for p in omit_i)
                for join in find_joins(big_piece, piece_i, point_limit):
                    self.assemble_pieces(join, omit_i)
                    if self._stop:
                        break

        self._level -= 1

    def find_solutions(self):
        self.assemble_pieces(P0, [P1, P2, P3])
        self._progress.close()

        print(f"Tested {self._num_candidates} candidates")
        print(f"Found {len(self._solutions)} solutions")

        # Write all the solutions into a file.
        if self._solutions:
            answer = list(map(draw_piece, self._solutions))
            combined = show_images(answer)
            combined.save("answer.png", "PNG")
            combined.close()
            for a in answer:
                a.close()


# Draws a list of cells, returning a PIL image
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
    im = Image.new("RGB", (width, height), color="white")
    d = ImageDraw.Draw(im)

    for c in cells:
        edges = [((v[0] - min_x), (v[1] - min_y)) for v in vertices(c[0])]
        d.polygon(edges, fill=COLORS[c[1]], outline="white")

    return im


# For debugging.
# Return an image of the piece annotated with the coordinates of each cell.

font = ImageFont.truetype("arial.ttf", size=15)


def map_piece(cells):
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
    im = Image.new("RGB", (width, height), color="white")
    d = ImageDraw.Draw(im)

    for c in cells:
        edges = [((v[0] - min_x), (v[1] - min_y)) for v in vertices(c[0])]
        d.polygon(edges, fill=COLORS[c[1]], outline="white")

        av_x = sum(e[0] for e in edges) / len(edges)
        av_y = sum(e[1] for e in edges) / len(edges)
        if not points_up(*c[0]):
            av_y -= 3
        text = "{} {}".format(*c[0])
        d.text((av_x - 13, av_y), text, font=font, fill="white")

    return im


def show_images(images):
    """Combines a list of images into a single image and displays it."""
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

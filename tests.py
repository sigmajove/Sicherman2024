import pytest
import sicherman2024 as main


def test_rotate():
    for x in range(-5, 5):
        for y in range(-5, 5):
            start = (x, y)
            orbit = start
            for _ in range(6):
                orbit = main.rotate_cell(orbit)
            assert orbit == start


def test_turns():
    for d0 in range(0, main.NUM_DIRECTIONS):
        for d1 in range(0, main.NUM_DIRECTIONS):
            turn = main.turn_angle(d0, d1)
            assert turn is None or turn in range(-2, 3)

    for d in range(0, main.NUM_DIRECTIONS):
        assert main.turn_angle(d, d) == 0
        assert main.turn_angle(d, main.OPPOSITE_DIRECTION[d]) is None

        # Test right turns.
        assert main.turn_angle(d, main.CLOCKWISE_DIRECTION[d]) == -2
        assert (
            main.turn_angle(
                d, main.CLOCKWISE_DIRECTION[main.CLOCKWISE_DIRECTION[d]]
            )
            == -1
        )

        # Test left turns.
        assert main.turn_angle(d, main.COUNTERCLOCKWISE_DIRECTION[d]) == 2
        assert (
            main.turn_angle(
                d,
                main.COUNTERCLOCKWISE_DIRECTION[
                    main.COUNTERCLOCKWISE_DIRECTION[d]
                ],
            )
            == 1
        )


up_triangle = main.make_cells(
    3,
    [
        r"       ",
        r"   /\  ",
        r"  /__\ ",
    ],
)

down_triangle = main.make_cells(
    2,
    [
        r" __  ",
        r"\  / ",
        r" \/  ",
    ],
)

big_piece = main.make_cells(
    1,
    [
        r" __  __  __  __  __  __  __  __  __  __  __  ",
        r"\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  / ",
        r" \/__\/__\/__\/__\/__\/__\/__\/__\/__\/__\/  ",
        r" /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  ",
        r"/__\/__\/__\/__\/__\/__\/__\/__\/__\/__\/__\ ",
        r"\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  / ",
        r" \/__\/__\/__\/__\/__\/__\/__\/__\/__\/__\/  ",
        r" /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  ",
        r"/__\/__\/__\/__\/__\/__\/__\/__\/__\/__\/__\ ",
        r"\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  / ",
        r" \/__\/__\/__\/__\/__\/__\/__\/__\/__\/__\/  ",
        r" /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  ",
        r"/__\/__\/__\/__\/__\/__\/__\/__\/__\/__\/__\ ",
    ],
)

round_piece = main.make_cells(
    1,
    [
        r"         __  __  __  __  __  __  __  __  __  ",
        r"       /\  /\  /\  /\  /\  /\  /\  /\  /\  / ",
        r"      /__\/__\/__\/__\/__\/__\/__\/__\/__\/  ",
        r"     /\  /\  /\  /\  /\  /\  /\  /\  /\  /   ",
        r"    /__\/__\/__\/__\/__\/__\/__\/__\/__\/    ",
        r"   /\  /\  /\  /\  /\  /\  /\  /\  /\  /     ",
        r"  /__\/__\/__\/__\/__\/__\/__\/__\/__\/      ",
        r" /\  /\  /\  /\  /\  /\  /\  /\  /\  /       ",
        r"/__\/__\/__\/__\/__\/__\/__\/__\/__\/        ",
        r"\  /\  /\  /\  /\  /\  /\  /\  /\  /         ",
        r" \/__\/__\/__\/__\/__\/__\/__\/__\/          ",
        r"  \  /\  /\  /\  /\  /\  /\  /\  /           ",
        r"   \/__\/__\/__\/__\/__\/__\/__\/            ",
    ],
)

notch_piece = main.make_cells(
    1,
    [
        r"         __  __  __  __  __  __  __  __  __  ",
        r"       /\  /\  /\  /\  /\  /\  /\  /\  /\  / ",
        r"      /__\/__\/__\/__\/__\/__\/__\/__\/__\/  ",
        r"     /\  /\  /\  /\  /\  /\  /\  /\  /\  /   ",
        r"    /__\/__\/__\/__\/__\/__\/__\/__\/__\/    ",
        r"   /\  /\  /\  /\  /\  /\  /\  /\  /\  /     ",
        r"  /__\/__\/__\/__\/__\/__\/__\/__\/__\/      ",
        r" /\  /\  /\  /\  /\  /\  /\  /\  /\  /       ",
        r"/__\/__\/__\/__\/__\/__\/__\/__\/__\/        ",
        r"\  /\  /\  /\  /\  /\  /\  /\  /\  /         ",
        r" \/__\/__\/__\/__\/__\/__\/__\/__\/          ",
        r"  \  /\  /\  /\  /\  /\  /\  /\  /           ",
        r"   \/__\/__\/__\/  \/__\/__\/__\/            ",
    ],
)

round_piece = main.make_cells(
    1,
    [
        r"         __  __  __  __  __  __  __  __  __  ",
        r"       /\  /\  /\  /\  /\  /\  /\  /\  /\  / ",
        r"      /__\/__\/__\/__\/__\/__\/__\/__\/__\/  ",
        r"     /\  /\  /\  /\  /\  /\  /\  /\  /\  /   ",
        r"    /__\/__\/__\/__\/__\/__\/__\/__\/__\/    ",
        r"   /\  /\  /\  /\  /\  /\  /\  /\  /\  /     ",
        r"  /__\/__\/__\/__\/__\/__\/__\/__\/__\/      ",
        r" /\  /\  /\  /\  /\  /\  /\  /\  /\  /       ",
        r"/__\/__\/__\/__\/__\/__\/__\/__\/__\/        ",
        r"\  /\  /\  /\  /\  /\  /\  /\  /\  /         ",
        r" \/__\/__\/__\/__\/__\/__\/__\/__\/          ",
        r"  \  /\  /\  /\  /\  /\  /\  /\  /           ",
        r"   \/__\/__\/__\/__\/__\/__\/__\/            ",
    ],
)

single_hole_piece = main.make_cells(
    1,
    [
        r"         __  __  __  __  __  __  __  __  __  ",
        r"       /\  /\  /\  /\  /\  /\  /\  /\  /\  / ",
        r"      /__\/__\/__\/__\/__\/__\/__\/__\/__\/  ",
        r"     /\  /\  /\  /\  /\  /\  /\  /\  /\  /   ",
        r"    /__\/__\/__\/__\/__\/__\/__\/__\/__\/    ",
        r"   /\  /\  /\  /\##/\  /\  /\  /\  /\  /     ",
        r"  /__\/__\/__\/__\/__\/__\/__\/__\/__\/      ",
        r" /\  /\  /\  /\  /\  /\  /\  /\  /\  /       ",
        r"/__\/__\/__\/__\/__\/__\/__\/__\/__\/        ",
        r"\  /\  /\  /\  /\  /\  /\  /\  /\  /         ",
        r" \/__\/__\/__\/__\/__\/__\/__\/__\/          ",
        r"  \  /\  /\  /\  /\  /\  /\  /\  /           ",
        r"   \/__\/__\/__\/__\/__\/__\/__\/            ",
    ],
)

double_hole_piece = main.make_cells(
    1,
    [
        r"         __  __  __  __  __  __  __  __  __  ",
        r"       /\  /\  /\  /\  /\  /\  /\  /\  /\  / ",
        r"      /__\/__\/__\/__\/__\/__\/__\/__\/__\/  ",
        r"     /\  /\  /\  /\  /\  /\  /\  /\  /\  /   ",
        r"    /__\/__\/__\/  \/__\/__\/__\/__\/__\/    ",
        r"   /\  /\  /\  /\  /\  /\  /\  /\  /\  /     ",
        r"  /__\/__\/__\/__\/__\/__\/__\/__\/__\/      ",
        r" /\  /\  /\  /\  /\  /\  /\  /\  /\  /       ",
        r"/__\/__\/__\/__\/__\/__\/__\/__\/__\/        ",
        r"\  /\  /\  /\  /\  /\  /\  /\  /\  /         ",
        r" \/__\/__\/__\/__\/__\/__\/__\/__\/          ",
        r"  \  /\  /\  /\  /\  /\  /\  /\  /           ",
        r"   \/__\/__\/__\/__\/__\/__\/__\/            ",
    ],
)

three_hole_piece = main.make_cells(
    1,
    [
        r"         __  __  __  __  __  __  __  __  __  ",
        r"       /\  /\  /\  /\  /\  /\  /\  /\  /\  / ",
        r"      /__\/__\/__\/__\/__\/__\/__\/__\/__\/  ",
        r"     /\  /\  /\  /\  /\  /\  /\  /\  /\  /   ",
        r"    /__\/__\/__\/  \/__\/__\/__\/__\/__\/    ",
        r"   /\  /\  /\  /\  /\  /\  /\  /\  /\  /     ",
        r"  /__\/__\/__\/__\/__\/__\/__\/__\/__\/      ",
        r" /\  /\  /\  /\  /\  /\  /\  /\  /\  /       ",
        r"/__\/__\/  \/__\/__\/__\/__\/__\/__\/        ",
        r"\  /\  /\  /\  /\  /\  /\##/\  /\  /         ",
        r" \/__\/__\/__\/__\/__\/__\/__\/__\/          ",
        r"  \  /\  /\  /\  /\  /\  /\  /\  /           ",
        r"   \/__\/__\/__\/__\/__\/__\/__\/            ",
    ],
)

edge_hole_piece = main.make_cells(
    1,
    [
        r"         __  __  __  __               ",
        r"       /\  /\  /\  /\  /\             ",
        r"      /__\/__\/  \/__\/__\            ",
        r"     /\  /\  /\  /\  /\  /\           ",
        r"    /__\/__\/__\/  \/__\/__\          ",
        r"   /\  /\  /\  /\  /\  /\  /          ",
        r"  /__\/__\/__\/__\/__\/__\/__  __     ",
        r" /\  /\  /\  /\  /\  /\  /\  /\  /\   ",
        r"/__\/__\/  \/__\/__\/__\/__\/__\/__\  ",
    ],
)


gutter_piece = main.make_cells(
    1,
    [
        r"         __  __  __  __  __      __      __  ",
        r"       /\  /\  /\  /\  /\  /   /\  /   /\  / ",
        r"      /__\/__\/__\/__\/__\/__ /__\/   /__\/  ",
        r"                     /\  /\  /\  /   /\  /   ",
        r"     __  __  __  __ /__\/__\/__\/   /__\/    ",
        r"   /\  /\  /\  /\  /\  /           /\  /     ",
        r"  /__\/__\/__\/__\/__\/    __  __ /__\/      ",
        r" /\  /\  /\  /   /\  /   /\  /\  /\  /       ",
        r"/__\/__\/__\/   /__\/__ /__\/__\/__\/        ",
        r"\  /\  /\  /   /\  /\  /\  /\  /\  /         ",
        r" \/__\/__\/   /__\/__\/__\/__\/__\/          ",
        r"  \  /\  /   /\  /       /\  /\  /           ",
        r"   \/__\/   /__\/       /__\/__\/            ",
    ],
)


def how_concave(piece, answer):
    assert (
        main.PieceScanner(piece).count_dents_or_points(dents_not_points=True)
        == answer
    )
    assert main.PieceScanner(piece).is_convex() == (answer == 0)


def test_concave():
    how_concave([], 0)
    how_concave(up_triangle, 0)
    how_concave(down_triangle, 0)
    how_concave(big_piece, 12)
    how_concave(round_piece, 0)
    how_concave(notch_piece, 1)


def check_points(piece, answer):
    assert (
        main.PieceScanner(piece).count_dents_or_points(dents_not_points=False)
        == answer
    )


def test_points():
    check_points([], 0)
    check_points(up_triangle, 3)
    check_points(down_triangle, 3)
    check_points(big_piece, 12)
    check_points(notch_piece, 13)
    check_points(gutter_piece, 28)


def check_holes(piece, answer):
    for b in True, False:
        assert answer == (
            main.PieceScanner(piece).count_dents_or_points(bool) is None
        )
    if answer:
        assert not main.PieceScanner(piece).is_convex()


def test_holes():
    check_holes([], False)
    check_holes(up_triangle, False)
    check_holes(down_triangle, False)
    check_holes(big_piece, False)
    check_holes(round_piece, False)
    check_holes(notch_piece, False)
    check_holes(single_hole_piece, True)
    check_holes(double_hole_piece, True)
    check_holes(three_hole_piece, True)
    check_holes(edge_hole_piece, True)

import pytest
import follow_border as main


def direction_0(piece):
    vertices = []
    c = main.Cursor(3, 27, 0)
    for angle in piece:
        c.advance(angle)
        vertices.append((c.x, c.y))

    # Check position and direction
    assert vertices[-1] == (3, 27)

    # Direction zero means horizontally to the right.
    assert vertices[-2] == (1, 27)


def direction_1(piece):
    vertices = []
    c = main.Cursor(16, 9, 1)
    for angle in piece:
        c.advance(angle)
        vertices.append((c.x, c.y))

    # Check position and direction
    assert vertices[-1] == (16, 9)

    # Direction one means means up and to the right.
    assert vertices[-2] == (15, 8)


def test_direction():
    data = [
        [2, 1, 5, 1, 3, 1, 3, 2],
        [1, 3, 2, 1, 4, 2, 1, 4],
        [2, 1, 4, 1, 3, 1, 3],
        [2, 2, 1, 5, 2, 1, 3, 2],
    ]
    for p in data:
        direction_0(p)
        direction_1(p)


def test_valley_to_valley():
    assert main.valley_to_valley([]) == 0
    assert main.valley_to_valley([1, 1, 1, 1, 1]) == 0
    assert main.valley_to_valley([1, 1, 4, 1, 1]) == 0
    assert main.valley_to_valley([4, 1, 1, 1, 1]) == 0
    assert main.valley_to_valley([1, 1, 1, 1, 4]) == 0
    assert main.valley_to_valley([4]) == 0
    assert main.valley_to_valley([1, 1, 1, 4, 4, 1, 1, 1]) == 1
    assert main.valley_to_valley([1, 1, 4, 4, 4, 1, 1, 1]) == 2
    assert main.valley_to_valley([1, 4, 1, 1, 4, 1, 1]) == 3
    assert main.valley_to_valley([1, 4, 1, 1, 1, 1, 4]) == 2

    test_piece = [1, 3, 2, 1, 4, 3, 3, 1, 3, 1, 4, 1, 4]
    #         **  **
    #       * /\**/\* /\
    #      * /__\/__\/__\
    #     * /\  /\  /\  /\
    #    * /__\/__\/__\/__\
    #    * \  /*
    #     * \/*
    #       **
    #
    # The shortest subpath of the border that touches all the
    # valleys has length 7.
    for _ in range(2):
        for _ in range(len(test_piece)):
            assert main.valley_to_valley(test_piece) == 7
            test_piece = test_piece[1:] + [test_piece[0]]
        test_piece.reverse()


def check_puzzle(pieces, golden):
    main.verify_pieces(pieces)

    variations = [[p, list(reversed(p))] for p in pieces[1:]]
    duplicates, solutions, tested, elapsed_time = main.Solver(
        pieces[0], variations
    ).solve()

    assert len(golden) == len(solutions)
    for a, b in zip(golden, solutions):
        assert a == b


def test_george():
    check_puzzle(
        [
            [2, 1, 5, 1, 3, 1, 3, 2],
            [1, 3, 2, 1, 4, 2, 1, 4],
            [2, 1, 4, 1, 3, 1, 3],
            [2, 2, 1, 5, 2, 1, 3, 2],
        ],
        [((0, 4, 2, 1), (1, 1, -1, 0), (1, 6, 0, 1))],
    )


def test_sawtooth():
    check_puzzle(
        [
            [2, 1, 3, 4, 1, 2, 2, 3],
            [3, 2, 3, 1, 2, 4, 3, 3, 1, 2],
            [2, 2, 1, 5, 1, 5, 2, 1, 3, 2, 3],
            [2, 2, 1, 5, 1, 5, 4, 1, 2, 2, 3, 2, 3],
        ],
        [((0, 2, 0, 1), (0, -3, 3, 4), (0, 3, 3, 1))],
    )


def test_scott():
    check_puzzle(
        [
            [3, 1, 3, 3, 2, 1, 3, 4, 1],
            [3, 1, 2, 3, 4, 1, 2, 2, 3],
            [4, 3, 3, 1, 2, 3, 3, 2, 3, 2, 1, 3],
            [3, 1, 2, 3, 3, 2, 2, 1, 4, 3],
        ],
        [
            ((0, -3, 1, 5), (0, -2, 0, 3), (0, 1, 1, 5)),
            ((0, -3, 1, 5), (0, 0, 4, 0), (0, 1, 1, 5)),
        ],
    )

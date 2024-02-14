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

    duplicates, solutions, tested, elapsed_time = main.Solver(pieces).solve()

    assert len(golden) == len(solutions)
    print(solutions)
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
            [3, 2, 3, 1, 2, 4, 3, 3, 1, 2, 3],
            [2, 2, 1, 5, 1, 5, 2, 1, 3, 2, 3],
            [2, 2, 1, 5, 1, 5, 4, 1, 2, 2, 3, 2, 3],
        ],
        [((0, 4, 2, 1), (0, -3, 3, 4), (0, 3, 3, 1))],
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


def test_convex():
    # Test the case where all the pieces are convex.
    pieces = [
        [1, 2, 1, 2],
        [1, 2, 1, 2],
        [1, 1, 1],
        [1, 1, 1],
    ]
    main.verify_pieces(pieces)

    duplicates, solutions, tested, elapsed_time = main.Solver(pieces).solve()

    # Because of all the symmetries, there are a large number of solutions.
    # Since there are duplicate pieces, the actual number of solutions is 15.
    assert len(solutions) == 60


def check_rotational_symmetry(piece, answer):
    lp = len(piece)
    rotated = piece[:]
    for _ in range(lp):
        assert main.rotational_symmetry(rotated)[1] == answer
        rotated = rotated[1:] + [rotated[0]]

    for i, p in enumerate(piece):
        assert (
            main.rotational_symmetry(piece[:i] + [10 + p] + piece[i + 1 :])
            is None
        )


def test_rotational_symmetry():
    for i in range(4, 20):
        assert main.rotational_symmetry(list(range(i))) is None
    for n in [6, 3, 2]:
        check_rotational_symmetry(n * [1], 1)
        check_rotational_symmetry(n * [1, 2], 2)
        check_rotational_symmetry(n * [1, 2, 3, 4], 4)


def check_mirror_symmetry(piece):
    assert main.has_mirror_symmetry(piece)
    for i, p in enumerate(piece):
        # Two consecutive unique elements breaks mirror symmetry.
        c = piece[:]
        c[i] = 1000
        c[(i + 1) % len(c)] = 1001
        assert not main.has_mirror_symmetry(c)


def test_mirror_symmetry():
    for i in range(4, 20):
        assert not main.has_mirror_symmetry(list(range(i)))

    for i in range(4, 9):
        # Even mirror symmetry
        m = list(range(i)) + list(range(i - 1, -1, -1))
        for _ in range(len(m)):
            check_mirror_symmetry(m)
            m = m[1:] + [m[0]]

        # Odd mirror symmetry
        m = list(range(1, i)) + [0] + list(range(i - 1, 0, -1))
        for _ in range(len(m)):
            check_mirror_symmetry(m)
            m = m[1:] + [m[0]]


def test_normalize():
    piece = [1, 3, 3, 2, 1, 3, 3, 2]

    # Piece with twofold rotational symmetry
    symmetry = main.rotational_symmetry(piece)
    assert symmetry == (3, 4)

    assert main._normalize(piece, symmetry, (1, 1, 0)) == (1, 1, 0)
    assert main._normalize(piece, symmetry, (-4, 4, 3)) == (1, 1, 0)

    piece = [2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3]

    # Piece with sixfold rotational symmetry
    symmetry = main.rotational_symmetry(piece)
    assert symmetry == (1, 2)

    assert main._normalize(piece, symmetry, (6, 2, 0)) == (6, 2, 0)
    assert main._normalize(piece, symmetry, (8, 4, 1)) == (6, 2, 0)
    assert main._normalize(piece, symmetry, (6, 6, 2)) == (6, 2, 0)
    assert main._normalize(piece, symmetry, (2, 6, 3)) == (6, 2, 0)
    assert main._normalize(piece, symmetry, (0, 4, 4)) == (6, 2, 0)
    assert main._normalize(piece, symmetry, (2, 2, 5)) == (6, 2, 0)

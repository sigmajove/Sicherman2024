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
    def widen(angles):
        return main.annotate(angles, 0, 0, 0)

    assert main.valley_to_valley(widen([])) == 0
    assert main.valley_to_valley(widen([1, 1, 1, 1, 1])) == 0
    assert main.valley_to_valley(widen([1, 1, 4, 1, 1])) == 0
    assert main.valley_to_valley(widen([4, 1, 1, 1, 1])) == 0
    assert main.valley_to_valley(widen([1, 1, 1, 1, 4])) == 0
    assert main.valley_to_valley(widen([4])) == 0
    assert main.valley_to_valley(widen([1, 1, 1, 4, 4, 1, 1, 1])) == 1
    assert main.valley_to_valley(widen([1, 1, 4, 4, 4, 1, 1, 1])) == 2
    assert main.valley_to_valley(widen([1, 4, 1, 1, 4, 1, 1])) == 3
    assert main.valley_to_valley(widen([1, 4, 1, 1, 1, 1, 4])) == 2

    test_piece = widen([1, 3, 2, 1, 4, 3, 3, 1, 3, 1, 4, 1, 4])
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


def test_splice():
    path0 = [1, 3, 2, 2, 4, 2, 1, 4, 2, 3, 1, 5]
    main.verify_pieces([path0])

    def run_test(first0, last0, path1, first1, last1, result):
        main.verify_pieces([path1])
        splice = main.splice_paths(
            path0=main.annotate(path0, 10, 20, 4),
            first0=first0,
            last0=last0,
            path1=path1,
            first1=first1,
            last1=last1,
        )
        angles = [t[0] for t in splice]
        main.verify_pieces([angles])
        assert angles == result

    run_test(
        4,
        1,
        [1, 2, 2, 3, 2, 2, 1, 4, 4],
        0,
        6,
        [4, 2, 2, 3, 2, 2, 5, 2, 1, 4, 2, 3, 1, 5, 1],
    )
    run_test(
        4,
        1,
        [1, 4, 4, 1, 2, 2, 3, 2, 2],
        3,
        0,
        [4, 2, 2, 3, 2, 2, 5, 2, 1, 4, 2, 3, 1, 5, 1],
    )

    run_test(
        8,
        6,
        [1, 2, 2, 2, 1, 4],
        3,
        1,
        [3, 1, 4, 1, 4, 3, 1, 5, 1, 3, 2, 2, 4, 2],
    )

    run_test(
        8,
        6,
        [2, 1, 4, 1, 2, 2],
        0,
        4,
        [3, 1, 4, 1, 4, 3, 1, 5, 1, 3, 2, 2, 4, 2],
    )


def test_zeroize():
    input = [
        (3, -4, 1),
        (6, 4, 2),
        (-23, -5, 7),
        (-3, 15, 3),
        (-9, 18, 7),
        (9, 11, 7),
        (-4, -19, 6),
        (2, 24, 7),
        (-16, -10, 3),
        (0, 9, 7),
        (8, -11, 7),
        (18, -25, 7),
        (11, 5, 7),
        (-10, -8, 7),
        (-13, -17, 2),
        (15, -9, 7),
        (-12, 2, 6),
        (13, 19, 7),
        (25, 6, 7),
        (-8, -10, 5),
        (7, 1, 7),
    ]

    assert main.zeroize(input) == (
        (0, 20, 7),
        (7, 15, 3),
        (10, 8, 2),
        (11, 27, 6),
        (13, 17, 7),
        (14, 43, 7),
        (15, 15, 5),
        (19, 6, 6),
        (20, 40, 3),
        (23, 34, 7),
        (25, 49, 7),
        (26, 21, 1),
        (29, 29, 2),
        (30, 26, 7),
        (31, 14, 7),
        (32, 36, 7),
        (34, 30, 7),
        (36, 44, 7),
        (38, 16, 7),
        (41, 0, 7),
        (48, 31, 7),
    )


def check_puzzle(pieces, golden):
    main.verify_pieces(pieces)

    duplicates, solutions, tested, elapsed_time = main.Solver(pieces).solve()

    answers = list(solutions.items())
    answers.sort()
    answers = tuple(value for key, value in answers)
    print(answers)

    assert len(golden) == len(answers)
    for a, b in zip(golden, answers):
        assert a == b


def test_george():
    check_puzzle(
        [
            [2, 1, 5, 1, 3, 1, 3, 2],
            [1, 3, 2, 1, 4, 2, 1, 4],
            [2, 1, 4, 1, 3, 1, 3],
            [2, 2, 1, 5, 2, 1, 3, 2],
        ],
        [((1, 3, -1, 0), (0, 4, 0, 0), (1, 1, -1, 3))],
    )


def test_tiny():
    check_puzzle(
        [
            [1, 2, 1, 2],
            [1, 3, 1, 2, 2],
            [1, 1, 1],
        ],
        [
            ((0, -1, 1, 1), (0, -4, 0, 3)),
            ((0, -3, -1, 4), (0, -1, -1, 1)),
            ((0, 1, -3, 5), (0, 0, -2, 5)),
            ((0, -2, 2, 2), (0, 0, -2, 5)),
        ],
    )


def test_sawtooth():
    check_puzzle(
        [
            [2, 1, 3, 4, 1, 2, 2, 3],
            [3, 2, 3, 1, 2, 4, 3, 3, 1, 2, 3],
            [2, 2, 1, 5, 1, 5, 2, 1, 3, 2, 3],
            [2, 2, 1, 5, 1, 5, 4, 1, 2, 2, 3, 2, 3],
        ],
        [((1, 0, -2, 0), (1, -1, -3, 3), (0, -5, -1, 2))],
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
            ((1, -1, -1, 5), (1, 0, 0, 5), (1, -3, -1, 4)),
            ((1, -1, -1, 5), (1, -5, -3, 2), (1, -3, -1, 4)),
        ],
    )


def test_convex():
    check_puzzle(
        [
            [1, 2, 1, 2],
            [1, 2, 1, 2],
            [1, 1, 1],
            [1, 1, 1],
        ],
        [
            ((0, -1, 1, 1), (0, -3, 1, 2), (0, -3, -1, 4)),
            ((0, 1, 1, 1), (0, -4, -2, 4), (0, -3, -1, 4)),
            ((0, -4, 0, 3), (0, -4, 0, 4), (0, -3, -1, 4)),
            ((0, -2, -2, 5), (0, 0, 2, 2), (0, -3, -1, 4)),
            ((0, -3, -3, 5), (0, -4, -2, 4), (0, -3, -1, 4)),
            ((0, 1, 1, 1), (0, 2, 2, 1), (0, -3, -1, 4)),
        ],
    )


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

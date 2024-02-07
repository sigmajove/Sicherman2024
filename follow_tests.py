import pytest
import follow_border as main


def direction_0(piece):
    vertices = []
    c = main.Cursor(3, 27, 0)
    for angle in piece:
        c.advance(angle)
        vertices.append(c.vertex())

    # Check position and direction
    assert vertices[-1] == (3, 27)

    # Direction zero means horizontally to the right.
    assert vertices[-2] == (1, 27)


def direction_1(piece):
    vertices = []
    c = main.Cursor(16, 9, 1)
    for angle in piece:
        c.advance(angle)
        vertices.append(c.vertex())

    # Check position and direction
    assert vertices[-1] == (16, 9)

    # Direction one means means up and to the right.
    assert vertices[-2] == (15, 8)


def test_direction():
    direction_0(main.PIECE0)
    direction_1(main.PIECE0)
    for p in main.VARIATIONS:
        for v in p:
            direction_0(v)
            direction_1(v)


def test_valley_to_valley():
    # PIECE0 has only one valley
    assert main.valley_to_valley(main.PIECE0) == 0

    assert (
        main.valley_to_valley(main.VARIATIONS[0][0])
        == main.valley_to_valley(main.VARIATIONS[0][1])
        == 3
    )

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

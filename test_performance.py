import numpy as np
from performance import *
import pytest

FLOAT_MARGIN = 0.00001
y_true_binary = np.asarray([[0., 0., 0., 0.],
                            [0., 1., 1., 0.],
                            [0., 1., 1., 0.],
                            [0., 0., 0., 0.]])
y_true_multiclass = np.asarray([[0., 1., 1., 0.],
                                [0., 1., 1., 0.],
                                [2., 2., 0., 3.],
                                [2., 2., 0., 0.]])


def test_iou_dice_score_binary_image():

    nb_classes = 2

    y_pred = np.asarray([[0., 0., 0., 0.],
                         [0., 0., 0., 0.],
                         [0., 0., 1., 1.],
                         [0., 0., 1., 1.]])

    iou, dice = iou_dice_score_multiclass(y_true_binary, y_pred, nb_classes=nb_classes)

    assert iou == pytest.approx([9 / (12 + 12 - 9), 1 / (4 + 4 - 1)], FLOAT_MARGIN)
    assert dice == pytest.approx([2 * 9 / (12 + 12), 2 * 1 / (4 + 4)], FLOAT_MARGIN)


def test_iou_dice_score_multiclass_image():
    nb_classes = 4

    y_pred = np.asarray([[0., 1., 1., 0.],
                         [0., 1., 1., 0.],
                         [0., 2., 2., 0.],
                         [0., 2., 2., 0.]])

    iou, dice = iou_dice_score_multiclass(y_true_multiclass, y_pred, nb_classes=nb_classes)

    expected_iou = [5 / (7 + 8 - 5),
                    1.,
                    2 / (4 + 4 - 2),
                    0.]

    expected_dice = [2 * 5 / (7 + 8),
                     1.,
                     2 * 2 / (4 + 4),
                     0.]

    assert iou == pytest.approx(expected_iou, FLOAT_MARGIN)
    assert dice == pytest.approx(expected_dice, FLOAT_MARGIN)


def test_dice_loss():

    y_pred = np.asarray([[0., 0., 0., 0.],
                         [0., 0., 0., 0.],
                         [0., 0., 1., 1.],
                         [0., 0., 1., 1.]])

    dice_loss = binary_dice_loss(y_true_binary, y_pred)

    assert dice_loss == pytest.approx(1 - 2 * 1 / (4 + 4), FLOAT_MARGIN)

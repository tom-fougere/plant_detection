import numpy as np


def iou_dice_score_multiclass(y_true, y_pred, nb_classes):
    """
    Computes IoU (Intersection-Over-Union) and Dice Score for each class of the data.

    :param y_true: True data, np-array
    :param y_pred: Predicted data, np-array
    :param nb_classes: Number of classes in the data, integer, >0
    :return: iou_score_per_class, dice_score_per_class
        - iou_score_per_class: list of IoU scores, float
        - dice_score_per_class: list of Dice scores, float
    """

    class_wise_iou = []
    class_wise_dice_score = []

    smoothing_factor = 0.00001

    for i in range(nb_classes):
        intersection_area = np.sum((y_pred == i) * (y_true == i))
        y_true_area = np.sum((y_true == i))
        y_pred_area = np.sum((y_pred == i))
        union_area = y_true_area + y_pred_area

        iou = intersection_area / (union_area - intersection_area + smoothing_factor)
        class_wise_iou.append(iou)

        dice_score = 2 * (intersection_area / (union_area + smoothing_factor))
        class_wise_dice_score.append(dice_score)

    return class_wise_iou, class_wise_dice_score


def mean_iou_dice_score_multiclass(y_true, y_pred, nb_classes):
    """
    Computes IoU (Intersection-Over-Union) and Dice Score for class of the data.
    The data are lists. The results is the mean of all data

    :param y_true: List of True data, np-array
    :param y_pred: List of Predicted data, np-array
    :param nb_classes: Number of classes in the data, integer, >0
    :return: iou_score_per_class, dice_score_per_class
        - iou_score_per_class: list of IoU scores, float
        - dice_score_per_class: list of Dice scores, float
    """

    average_iou = np.zeros(nb_classes)
    average_dice = np.zeros(nb_classes)
    count = 0.
    for gt, pred in zip(y_true, y_pred):
        count += 1

        # Compute performance
        cls_wise_iou, cls_wise_dice_score = iou_dice_score_multiclass(gt, pred, nb_classes)

        average_iou = np.add(average_iou, cls_wise_iou)
        average_dice = np.add(average_dice, cls_wise_dice_score)

    average_iou = average_iou / count
    average_dice = average_dice / count

    return average_iou, average_dice


def binary_dice_loss(y_true, y_pred):
    """
    Compute the dice score loss of a binary data (1 - dice_score)
    (Only use the values = 1)
    :param y_true: True data, np-array
    :param y_pred: Predicted data, np-array
    :return: dice score: float
    """

    iou_score, dice_score = iou_dice_score_multiclass(y_true, y_pred, nb_classes=2)

    return 1 - dice_score[1]
import numpy as np
from processing_functions import postprocessing_masks_prediction


def class_wise_metrics(y_true, y_pred, nb_classes):
    """
    Computes the class-wise IOU and Dice Score.

    Args:
      y_true (tensor) - ground truth label maps
      y_pred (tensor) - predicted label maps
    """
    class_wise_iou = []
    class_wise_dice_score = []

    smoothing_factor = 0.00001

    for i in range(nb_classes):
        intersection = np.sum((y_pred == i) * (y_true == i))
        y_true_area = np.sum((y_true == i))
        y_pred_area = np.sum((y_pred == i))
        combined_area = y_true_area + y_pred_area

        iou = (intersection) / (combined_area - intersection + smoothing_factor)
        class_wise_iou.append(iou)

        dice_score = 2 * ((intersection) / (combined_area + smoothing_factor))
        class_wise_dice_score.append(dice_score)

    return class_wise_iou, class_wise_dice_score


def mean_class_wise_metrics(y_true, y_pred, nb_classes):
    """
    Computes the class-wise IOU and Dice Score.

    Args:
      y_true (tensor) - ground truth label maps
      y_pred (tensor) - predicted label maps
    """

    average_iou = 0.
    average_dice = 0.
    count = 0.
    for gt, pred in zip(y_true, y_pred):
        count += 1
        # Compute performance
        cls_wise_iou, cls_wise_dice_score = class_wise_metrics(gt,
                                                               postprocessing_masks_prediction(pred,
                                                                                               new_size=(gt.shape[0],
                                                                                                         gt.shape[1])),
                                                               nb_classes)

        average_iou += cls_wise_iou[0]
        average_dice += cls_wise_dice_score[0]

    average_iou = average_iou / count
    average_dice = average_dice / count

    return average_iou, average_dice

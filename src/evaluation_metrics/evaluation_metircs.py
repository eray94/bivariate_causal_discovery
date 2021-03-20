import numpy as np
from typing import List
from sklearn.metrics import roc_auc_score, accuracy_score

from src.constant import ModelConstants
from src.errors import AucError, SizeMismatchError

class EvaluationMetrics:
    @staticmethod
    def accuracy(predictions: List) -> float:
        """Calculate accuracy of predictions.

                Args:
                    predictions (List): List of causal relation predictions.
                Returns:
                    float: Accuracy of predictions.
                Raises: SizeMismatchError if predictions and ground truth size are not match.
                        TypeError if expected type of argument is not satisfied.
        """
        try:
            accuracy = accuracy_score(np.array(ModelConstants.LABELS), np.array(predictions))
        except:
            if len(ModelConstants.LABELS) != len(predictions):
                raise SizeMismatchError("Labels length and prediction length are not match !!!")
            else:
                raise TypeError

        return accuracy

    @staticmethod
    def auc(predictions: List) -> float:
        """Calculate AUC score of predictions.

                Args:
                    predictions (List): List of causal relation predictions.
                Returns:
                    float: AUC score of predictions.
                Raises: SizeMismatchError if predictions and ground truth size are not match.
                        AucError if ground truth only contain example from one class.
                        TypeError if expected type of argument is not satisfied.
        """
        try:
            auc_score = roc_auc_score(np.array(ModelConstants.LABELS), np.array(predictions))
        except:
            if len(ModelConstants.LABELS) != len(predictions):
                raise SizeMismatchError("Labels length and prediction length are not match !!!")
            elif np.array(ModelConstants.LABELS).sum() == 0 or np.array(ModelConstants.LABELS).mean() == 1:
                raise AucError("Labels or predictions only contains data from one class. Auc cannot be calculate !!!")
            else:
                raise TypeError

        return auc_score

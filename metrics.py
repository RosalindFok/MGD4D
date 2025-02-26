import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score

class Metrics:
    @staticmethod
    def AUC(prob : np.ndarray, true : np.ndarray) -> float:
        assert prob.shape == true.shape, f"prob.shape {prob.shape} != true.shape {true.shape}"
        return roc_auc_score(true, prob)

    @staticmethod
    def ACC(pred : np.ndarray, true : np.ndarray) -> float:
        assert pred.shape == true.shape, f"pred.shape {pred.shape} != true.shape {true.shape}"
        return accuracy_score(true, pred)

    @staticmethod
    def PRE(pred : np.ndarray, true : np.ndarray) -> float:
        assert pred.shape == true.shape, f"pred.shape {pred.shape} != true.shape {true.shape}"
        return precision_score(true, pred)

    @staticmethod
    def SEN(pred : np.ndarray, true : np.ndarray) -> float:
        assert pred.shape == true.shape, f"pred.shape {pred.shape} != true.shape {true.shape}"
        return recall_score(true, pred)

    @staticmethod
    def F1S(pred : np.ndarray, true : np.ndarray) -> float:
        assert pred.shape == true.shape, f"pred.shape {pred.shape} != true.shape {true.shape}"
        return f1_score(true, pred)

import numpy as np

# 多标签分类
class Metric:
    def __init__(self):
        self.pred = [] # [[0.23, 0.91, 0.83, 0.12], ...]
        self.truth = [] # [[0, 0, 1, 1], ...]

        self.best = -1
        self.try_times = 0

    def clear(self):
        self.pred = []
        self.truth = []


    def store(self, pred, truth):
        self.pred += pred
        self.truth += truth

    def precision_recall_f1(self, threshold=0.5):
        pred = np.array(self.pred)
        pred = np.where(pred > threshold, 1, 0)
        truth = np.array(self.truth)

        TP = np.sum(truth * pred, axis=0)
        FP = np.sum((1 - truth) * pred, axis=0)
        FN = np.sum(truth * (1 - pred), axis=0)
        # TN = np.sum((1 - truth) * (1 - pred), axis=0)

        # each class
        precisions = TP / (TP + FP + 1e-5)
        recalls = TP / (TP + FN + 1e-5)
        f1s = 2 * precisions * recalls / (precisions + recalls + 1e-5)

        # macro average
        macro_precision = precisions.mean()
        macro_recall = recalls.mean()
        macro_f1 = f1s.mean()

        # micro average
        tp = TP.sum()
        fp = FP.sum()
        fn = FN.sum()

        micro_precision = tp / (tp + fp + 1e-5)
        micro_recall = tp / (tp + fn + 1e-5)
        micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall + 1e-5)

        return (precisions, recalls, f1s), (macro_precision, macro_recall, macro_f1), (micro_precision, micro_recall, micro_f1)

    def is_improved(self, x, patience=10):
        if x > self.best:
            self.best = x
            self.try_times = 0
            return True, False
        elif self.try_times < patience:
            self.try_times += 1
            return False, False
        else:
            self.try_times += 1
            return False, True


# 二分类
class MetricForBinary:
    def __init__(self):
        self.pred = [] # [0.23, 0.91, 0.83, 0.12, ...]
        self.truth = [] # [0, 0, 1, 1, ...]

        self.best = -1
        self.try_times = 0

    def clear(self):
        self.pred = []
        self.truth = []


    def store(self, pred, truth):
        self.pred += pred
        self.truth += truth

    def acc_precision_recall_f1(self, threshold=0.5):
        pred = np.array(self.pred)
        pred = np.where(pred > threshold, 1, 0)
        truth = np.array(self.truth)

        TP = np.sum(truth * pred)
        FP = np.sum((1 - truth) * pred)
        FN = np.sum(truth * (1 - pred))
        TN = np.sum((1 - truth) * (1 - pred))

        precision = TP / (TP + FP + 1e-5)
        recall = TP / (TP + FN + 1e-5)
        f1 = 2 * precision * recall / (precision + recall + 1e-5)

        acc = (TP + TN) / (TP + TN + FP + FN)

        return acc, precision, recall, f1

    def is_improved(self, x, patience=10):
        if x > self.best:
            self.best = x
            self.try_times = 0
            return True, False
        elif self.try_times < patience:
            self.try_times += 1
            return False, False
        else:
            self.try_times += 1
            return False, True
import numpy as np


# 多分类指标
class Metric:
    def __init__(self, num_labels):
        self.pred = [] # [[0.9, 0.01, 0.001, 0.02], ...]
        self.truth = [] # [0, ...]
        self.num_labels = num_labels

        self.best = -1
        self.try_times = 0

    def clear(self):
        self.pred = []
        self.truth = []

    def store(self, pred, truth):
        self.pred += pred
        self.truth += truth

    def get_pred(self):
        return np.argmax(self.pred, axis=1)

    def acc(self):
        num = 0
        preds = np.argmax(self.pred, axis=1)
        for pred, truth in zip(preds, self.truth):
            if pred == truth:
                num += 1
        return num / len(self.pred)

    def precision_recall_f1(self):
        pred = np.argmax(self.pred, axis=1)

        confusion_matrix = np.zeros((self.num_labels, self.num_labels), dtype=np.int)
        for p, t in zip(pred, self.truth):
            confusion_matrix[p][int(t)] += 1

        pred = np.eye(self.num_labels, dtype=np.int)[pred] # one hot
        truth = np.eye(self.num_labels, dtype=np.int)[self.truth]

        TP = np.sum(truth * pred, axis=0)
        FP = np.sum((1 - truth) * pred, axis=0)
        FN = np.sum(truth * (1 - pred), axis=0)
        # TN = np.sum((1 - truth) * (1 - pred), axis=0)

        # each class
        precisions = TP / (TP + FP + 1e-5)
        recalls = TP / (TP + FN + 1e-5)
        f1s = 2 * precisions * recalls / (precisions + recalls + 1e-5)

        # macro average
        macro_precision = 0.0
        macro_recall = 0.0
        macro_f1 = 0.0

        # micro average
        tp = 0.0
        fp = 0.0
        fn = 0.0

        for i in range(1, self.num_labels):
            macro_precision += precisions[i]
            macro_recall += recalls[i]
            macro_f1 += f1s[i]

            tp += TP[i]
            fp += FP[i]
            fn += FN[i]

        macro_precision /= (self.num_labels - 1)
        macro_recall /= (self.num_labels - 1)
        macro_f1 /= (self.num_labels - 1)

        micro_precision = tp / (tp + fp + 1e-5)
        micro_recall = tp / (tp + fn + 1e-5)
        micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall + 1e-5)

        return (precisions, recalls, f1s), \
               (macro_precision, macro_recall, macro_f1), \
               (micro_precision, micro_recall, micro_f1), \
               confusion_matrix.tolist()

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

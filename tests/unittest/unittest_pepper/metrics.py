import unittest
from pepper.metrics import require_probas
from sklearn import metrics


class TestRequireProbas(unittest.TestCase):

    def test_require_probas_with_probabilistic_metrics(self):
        self.assertTrue(require_probas(metrics.roc_auc_score))
        self.assertTrue(require_probas(metrics.brier_score_loss))
        self.assertTrue(require_probas(metrics.average_precision_score))
        self.assertTrue(require_probas(metrics.precision_recall_curve))
        self.assertTrue(require_probas(metrics.log_loss))
        self.assertTrue(require_probas(metrics.f1_score))
        self.assertTrue(require_probas(metrics.log_loss))

    def test_require_probas_with_non_probabilistic_metrics(self):
        self.assertFalse(require_probas(metrics.accuracy_score))
        self.assertFalse(require_probas(metrics.confusion_matrix))
        self.assertFalse(require_probas(metrics.matthews_corrcoef))


if __name__ == '__main__':
    unittest.main()

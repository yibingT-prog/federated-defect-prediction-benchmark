"""Independent small cases; no benchmark data or training required."""
import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments/completed/fedcpdp-paper-lopo-20261110'))
from run import classification,effort_metrics,split_source

class MetricsTests(unittest.TestCase):
    def test_perfect_classification(self):
        result=classification(np.array([0,1,0,1]),np.array([.1,.9,.2,.8]),.5)
        self.assertTrue(all(value==1 for value in result.values()))
    def test_constant_classifier(self):
        result=classification(np.array([0,1]),np.array([.1,.1]),.5)
        self.assertEqual(result['MCC'],0)
        self.assertEqual(result['ROC_AUC'],.5)
    def test_fractional_budget_and_optimal_order(self):
        result=effort_metrics(np.array([1,0]),np.array([.9,.1]),np.array([1.,1.]))
        self.assertAlmostEqual(result['CostEffect@20'],.4)
        self.assertAlmostEqual(result['Popt@20'],1.)
    def test_reverse_order(self):
        result=effort_metrics(np.array([1,0]),np.array([.1,.9]),np.array([1.,1.]))
        self.assertAlmostEqual(result['Popt@20'],0.)
    def test_invalid_loc(self):
        result=effort_metrics(np.array([1,0]),np.array([.9,.1]),np.array([0.,1.]))
        self.assertIsNone(result['Popt@20'])
        self.assertIn('reason',result)
    def test_duplicate_groups_stay_together(self):
        groups=np.repeat(np.arange(20),2); y=np.repeat(np.arange(20)%2,2)
        tr,va=split_source(y,groups)
        self.assertFalse(set(groups[tr])&set(groups[va]))
        self.assertEqual(len(np.unique(y[tr])),2)
        self.assertEqual(len(np.unique(y[va])),2)

if __name__=='__main__': unittest.main()

import numpy as np
from src.shift.abstention import ShiftEvaluator, AbstentionTier

def test_shift_evaluator():
    evaluator = ShiftEvaluator(degraded_threshold=0.1, abstain_threshold=0.5)
    
    # We use a deterministic distribution to avoid random test failures
    np.random.seed(42)
    cal_scores = np.random.normal(0.5, 0.1, 1000)
    evaluator.calibrate(cal_scores)
    
    assert evaluator.is_calibrated
    
    # Test identical distribution
    test_scores_same = np.random.normal(0.5, 0.1, 500)
    tier, shift = evaluator.evaluate(test_scores_same)
    assert shift < 0.1
    assert tier == AbstentionTier.CERTIFIED
    
    # Test shifted distribution (degraded)
    test_scores_degraded = np.random.normal(0.6, 0.1, 500)
    tier, shift = evaluator.evaluate(test_scores_degraded)
    # The KS distance between N(0.5, 0.1) and N(0.6, 0.1) is approx 0.38
    assert 0.1 <= shift <= 0.5
    assert tier == AbstentionTier.DEGRADED
    
    # Test heavily shifted distribution (abstain)
    test_scores_abstain = np.random.normal(0.9, 0.1, 500)
    tier, shift = evaluator.evaluate(test_scores_abstain)
    assert shift > 0.5
    assert tier == AbstentionTier.ABSTAIN

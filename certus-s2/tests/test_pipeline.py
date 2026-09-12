import pytest
import numpy as np
from src.pipeline.runner import CertusPipeline, BlockedInfrastructureError
from src.observation.operator import EffectiveObservationOperator
from src.risk.engine import RiskEngine
from src.decision.proposal import D1ProposalStage
from src.certification.track_a import TrackACertifier
from src.certification.track_b import TrackBCertifier
from src.shift.abstention import ShiftEvaluator

def test_pipeline_blocked():
    # Instantiate all modules without SR backbone
    op = EffectiveObservationOperator()
    risk = RiskEngine()
    prop = D1ProposalStage()
    cert_a = TrackACertifier()
    cert_b = TrackBCertifier()
    shift = ShiftEvaluator()
    
    pipeline = CertusPipeline(op, risk, prop, cert_a, cert_b, shift)
    
    l2a_mock = np.zeros((4, 10, 10))
    
    # Must raise BlockedInfrastructureError because sr_backbone_loaded is False
    with pytest.raises(BlockedInfrastructureError):
        pipeline.run_inference(l2a_mock)

def test_pipeline_sr_integration():
    # Instantiate with SR backbone provided
    op = EffectiveObservationOperator()
    op.is_fitted = True
    risk = RiskEngine(backend_type="numpy")
    prop = D1ProposalStage()
    cert_a = TrackACertifier()
    cert_b = TrackBCertifier()
    shift = ShiftEvaluator()
    
    class MockSRWrapper:
        def predict_numpy(self, x):
            c, h, w = x.shape[-3:]
            return np.ones((c, 4 * h, 4 * w), dtype=np.float32) * 0.15
            
    pipeline = CertusPipeline(op, risk, prop, cert_a, cert_b, shift, sr_wrapper=MockSRWrapper())
    assert pipeline._sr_backbone_loaded is True
    
    l2a_input = np.ones((4, 10, 10), dtype=np.float32) * 0.15
    sr_out = pipeline.reconstruct_sr(l2a_input)
    assert sr_out.shape == (4, 40, 40)
    assert np.all(np.isfinite(sr_out))
    assert np.all(sr_out == 0.15)

def test_pipeline_compute_analytical_layers():
    op = EffectiveObservationOperator()
    op.is_fitted = True
    risk = RiskEngine(backend_type="numpy")
    prop = D1ProposalStage()
    cert_a = TrackACertifier()
    cert_b = TrackBCertifier()
    shift = ShiftEvaluator()
    
    class MockSRWrapper:
        def predict_numpy(self, x):
            c, h, w = x.shape[-3:]
            return np.ones((c, 4 * h, 4 * w), dtype=np.float32) * 0.15
            
    pipeline = CertusPipeline(op, risk, prop, cert_a, cert_b, shift, sr_wrapper=MockSRWrapper())
    
    l2a_input = np.ones((4, 10, 10), dtype=np.float32) * 0.15
    layers = pipeline.compute_analytical_layers(l2a_input)
    
    expected_keys = [
        "sr_image",
        "observation_residual",
        "observation_residual_10m",
        "measurement_support",
        "epistemic_uncertainty",
        "aleatoric_uncertainty",
        "risk_map",
        "provenance"
    ]
    for k in expected_keys:
        assert k in layers, f"Missing key {k} in analytical layers"
        
    # Check consistent 2.5m spatial dimensions across visualization layers
    target_shape = (4, 40, 40)
    assert layers["sr_image"].shape == target_shape
    assert layers["observation_residual"].shape == target_shape
    assert layers["measurement_support"].shape == target_shape
    assert layers["epistemic_uncertainty"].shape == target_shape
    assert layers["aleatoric_uncertainty"].shape == target_shape
    assert layers["risk_map"].shape == target_shape
    assert layers["observation_residual_10m"].shape == (4, 10, 10)
    
    # Check bounds
    assert np.all((layers["measurement_support"] >= 0.0) & (layers["measurement_support"] <= 1.0))
    assert np.all((layers["risk_map"] >= 0.0) & (layers["risk_map"] <= 1.0))



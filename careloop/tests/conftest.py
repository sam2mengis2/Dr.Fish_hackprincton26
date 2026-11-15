"""Tests always run offline in mock mode, whatever config.yaml is set to for live demos
(real LLM, real X-ray model, Photon)."""
import careloop.runtime as runtime

_real_load_config = runtime.load_config


def _mock_config(*args, **kwargs):
    cfg = _real_load_config(*args, **kwargs)
    cfg["llm"]["provider"] = "mock"
    cfg["llm"]["reasoner_backend"] = "auto"
    cfg["cv"]["cxr_backend"] = "mock"
    cfg["messenger"] = "console"
    return cfg


runtime.load_config = _mock_config

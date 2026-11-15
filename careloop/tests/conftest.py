"""Tests always run offline in mock mode, whatever config.yaml is set to for live demos
(real LLM, real X-ray model, Photon), and never send LangSmith traces: tracing would also read through
the scripted fake model's reply list and break the Deep Agent tests."""
import os

for _k in ("LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2", "LANGCHAIN_TRACING"):
    os.environ[_k] = "false"

import careloop.runtime as runtime  # noqa: E402

_real_load_config = runtime.load_config


def _mock_config(*args, **kwargs):
    cfg = _real_load_config(*args, **kwargs)
    cfg["llm"]["provider"] = "mock"
    cfg["llm"]["reasoner_backend"] = "auto"
    cfg["cv"]["cxr_backend"] = "mock"
    cfg["messenger"] = "console"
    return cfg


runtime.load_config = _mock_config

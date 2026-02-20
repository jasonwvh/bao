from __future__ import annotations

from bao.benchmark.agents import build_default_agents
from bao.benchmark.calibration import SharedCalibrator
from bao.benchmark.policies import build_policies
from bao.benchmark.types import BenchmarkConfig, FlowRecord


def test_all_policies_share_same_config_instance():
    cfg = BenchmarkConfig()
    agents = build_default_agents()
    rows = [FlowRecord(flow_id="f1", label=0, features={"packet_count": 1.0})]
    calibrator = SharedCalibrator(params={})

    policies = build_policies(cfg, rows, agents, calibrator, seed=1)

    for policy in policies.values():
        assert policy.cfg is cfg

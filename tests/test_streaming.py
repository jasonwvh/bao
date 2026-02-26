from __future__ import annotations

import unittest

from agents.common.streaming import GLOBAL_STREAM_ID, derive_stream_id


class StreamingIdTests(unittest.TestCase):
    def test_explicit_stream_id_takes_priority(self) -> None:
        sid = derive_stream_id(
            flow_features={"proto": "tcp", "service": "http", "state": "FIN"},
            explicit_stream_id="session-42",
        )
        self.assertEqual(sid, "session-42")

    def test_five_tuple_is_used_when_available(self) -> None:
        sid = derive_stream_id(
            flow_features={
                "srcip": "10.0.0.1",
                "dstip": "10.0.0.2",
                "sport": 1234,
                "dport": 443,
                "proto": "tcp",
            }
        )
        self.assertEqual(sid, "10.0.0.1|10.0.0.2|1234|443|tcp")

    def test_unsw_fallback_is_stable(self) -> None:
        sid = derive_stream_id(
            flow_features={
                "proto": "udp",
                "service": "dns",
                "state": "CON",
                "is_sm_ips_ports": 0,
            }
        )
        self.assertEqual(sid, "unsw|udp|dns|CON|0")

    def test_global_fallback_when_no_features(self) -> None:
        sid = derive_stream_id(flow_features={})
        self.assertEqual(sid, GLOBAL_STREAM_ID)


if __name__ == "__main__":
    unittest.main()

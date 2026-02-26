from __future__ import annotations

import unittest

import numpy as np

from agents.lstm_autoencoder.service import StreamSequenceBuffer


class LSTMStreamBufferTests(unittest.TestCase):
    def test_sequence_state_changes_with_history(self) -> None:
        buf = StreamSequenceBuffer(window_size=3, max_streams=4)
        v1 = np.array([1.0, 0.0], dtype=np.float32)
        v2 = np.array([2.0, 0.0], dtype=np.float32)
        v3 = np.array([3.0, 0.0], dtype=np.float32)

        s1 = buf.append_and_get("s", v1)
        s2 = buf.append_and_get("s", v2)
        s3 = buf.append_and_get("s", v3)

        self.assertEqual(s1.shape, (3, 2))
        self.assertEqual(s2.shape, (3, 2))
        self.assertEqual(s3.shape, (3, 2))
        self.assertTrue(np.allclose(s1[0], v1))
        self.assertTrue(np.allclose(s1[-1], v1))
        self.assertTrue(np.allclose(s2[-1], v2))
        self.assertTrue(np.allclose(s3[0], v1))
        self.assertTrue(np.allclose(s3[1], v2))
        self.assertTrue(np.allclose(s3[2], v3))


if __name__ == "__main__":
    unittest.main()

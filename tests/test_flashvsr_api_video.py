import unittest

from flashvsr_api.video import padded_frame_count, target_dimensions


class VideoGeometryTests(unittest.TestCase):
    def test_legacy_15x_target(self):
        self.assertEqual(target_dimensions(1376, 768, 1.5), (2048, 1152))

    def test_default_14x_target(self):
        self.assertEqual(target_dimensions(1376, 768, 1.4), (1920, 1024))

    def test_padding_preserves_all_source_frames(self):
        self.assertEqual(padded_frame_count(243), 249)
        self.assertGreaterEqual(padded_frame_count(243) - 4, 243)

    def test_padding_is_minimal(self):
        for frames in range(1, 301):
            padded = padded_frame_count(frames)
            self.assertEqual(padded % 8, 1)
            self.assertGreaterEqual(padded - 4, frames)
            self.assertLess(padded - 12, frames)

    def test_invalid_scale(self):
        with self.assertRaises(ValueError):
            target_dimensions(1376, 768, 0)


if __name__ == "__main__":
    unittest.main()

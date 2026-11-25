import unittest

import numpy as np

try:
    from training.data.datasets_pano2pano_four_view_online import (
        _augment_depth_down_up_blur,
        _normalise_depth_aug_config,
        _normalise_depth_aug_prob,
    )
except ImportError as exc:
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


@unittest.skipIf(_IMPORT_ERROR is not None, f"four-view dataset import failed: {_IMPORT_ERROR}")
class TestFourViewDepthAug(unittest.TestCase):
    def test_none_mode_disables_aug(self):
        mode, scale, kernel, sigma = _normalise_depth_aug_config("none", None, None, None)

        self.assertEqual(mode, "none")
        self.assertEqual(scale, 1.0)
        self.assertEqual(kernel, 1)
        self.assertEqual(sigma, 0.0)

    def test_custom_mode_normalises_even_kernel(self):
        mode, scale, kernel, sigma = _normalise_depth_aug_config("custom", 0.5, 4, 1.25)

        self.assertEqual(mode, "custom")
        self.assertEqual(scale, 0.5)
        self.assertEqual(kernel, 5)
        self.assertEqual(sigma, 1.25)

    def test_mild_preset_matches_current_training_configuration(self):
        mode, scale, kernel, sigma = _normalise_depth_aug_config("mild", None, None, None)

        self.assertEqual(mode, "mild")
        self.assertEqual(scale, 0.125)
        self.assertEqual(kernel, 11)
        self.assertEqual(sigma, 0.0)

    def test_aug_prob_bounds(self):
        self.assertEqual(_normalise_depth_aug_prob(0.5), 0.5)
        with self.assertRaises(ValueError):
            _normalise_depth_aug_prob(1.5)

    def test_down_up_blur_preserves_shape_and_smooths(self):
        depth = np.indices((16, 32)).sum(axis=0).astype(np.float32) % 2

        augmented = _augment_depth_down_up_blur(depth, 0.25, 5, 0.0)

        self.assertEqual(augmented.shape, depth.shape)
        self.assertEqual(augmented.dtype, np.float32)
        self.assertFalse(np.allclose(augmented, depth))
        self.assertGreater(float(augmented.min()), 0.0)
        self.assertLess(float(augmented.max()), 1.0)


if __name__ == "__main__":
    unittest.main()

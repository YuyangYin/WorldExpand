import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from reconstruction.erp import cut_tangent_views, erp_rays
from reconstruction.mirror import prepare_gs_data


class ReconstructionContractTests(unittest.TestCase):
    def test_tangent_prior_matches_radial_depth_and_camera_axes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            center = root / 'center.png'
            Image.new('RGB', (64, 32)).save(center)
            depth = root / 'depth.npy'
            np.save(depth, np.full((32, 64), 2, dtype=np.float32))
            views = [(center, np.eye(4)), (center, np.eye(4))]
            prior = cut_tangent_views(views, depth, root / 'mirror', face_count=20, size=25)
            cameras = json.loads(prior.read_text())
            self.assertEqual(len(cameras['extrinsics']), 42)
            np.testing.assert_allclose(cameras['extrinsics'][0]['matrix'], np.eye(4), atol=1e-7)
            for entry in cameras['extrinsics']:
                self.assertAlmostEqual(np.linalg.det(np.asarray(entry['matrix'])[:3, :3]), 1)
            z = np.load(root / 'mirror/depth/view_0000.npy')
            self.assertEqual(z[12, 12], 2)
            self.assertLess(z[0, 0], 2)
            self.assertEqual(len(list((root / 'mirror/depth').glob('*.npy'))), 42)
            self.assertFalse(np.load(root / 'mirror/depth/view_0021.npy').any())

    def test_erp_rays_look_forward_at_center_and_up_at_top(self):
        rays = erp_rays(3, 6)
        self.assertTrue((rays[0, :, 1] < 0).all())
        self.assertTrue((rays[1, 2:4, 2] > 0.8).all())
        np.testing.assert_allclose(np.linalg.norm(rays, axis=-1), 1, atol=1e-6)

    def test_gs_data_inverts_camera_and_preserves_fp16_depth_bits(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / 'mirror'
            target = Path(tmp) / 'gs'
            (source / 'depth').mkdir(parents=True)
            images = Path(tmp) / 'tangent' / 'images'
            images.mkdir(parents=True)
            depth = np.array([[2.5, np.inf], [-1.0, 4]], dtype=np.float32)
            np.save(source / 'depth/depth_0000.npy', depth)
            Image.new('RGB', (2, 2), (50, 100, 150)).save(images / 'view_0000.png')
            (source / 'points.ply').write_bytes(b'ply fixture')
            c2w = np.eye(4)
            c2w[0, 3] = 3
            c2w[3, 0] = 6e-8  # Roundoff from the public pipeline inverse.
            record = {'extrinsics': [{'camera_id': 0, 'matrix': c2w.tolist()}],
                      'intrinsics': [{'camera_id': 0, 'matrix': np.eye(3).tolist()}]}
            (source / 'camera_params.json').write_text(json.dumps(record))
            prepare_gs_data(source, images, target, expected_views=1)
            with Image.open(target / 'images/view_0000.png') as image:
                self.assertEqual(image.getpixel((0, 0)), (50, 100, 150))
            result = json.loads((target / 'cameras.json').read_text())
            self.assertEqual(result['view_0000']['extrinsic'][0][3], -3)
            with Image.open(target / 'depths/view_0000.png') as image:
                decoded = np.asarray(image, dtype=np.uint16).view(np.float16)
            np.testing.assert_array_equal(decoded, [[2.5, 0], [0, 4]])
            self.assertEqual((target / 'points.ply').read_bytes(), b'ply fixture')
            self.assertFalse((target / 'points.ply').is_symlink())


if __name__ == '__main__':
    unittest.main()

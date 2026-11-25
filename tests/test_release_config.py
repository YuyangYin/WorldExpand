"""Portable release configuration: no model/GPU dependencies."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from worldexpand.config import load_config, resolve_path, validate_case_id, expected_outputs


class ReleaseConfigTests(unittest.TestCase):
    def test_relative_paths_follow_config_location_not_cwd(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = Path(d) / 'paths.json'
            cfg.write_text(json.dumps({'paths': {'model': '../weights', 'output': 'out'}}))
            got = load_config(cfg)
            self.assertEqual(got['paths']['model'], str((Path(d) / '../weights').resolve()))
            self.assertEqual(got['paths']['output'], str(Path(d) / 'out'))

    def test_env_override_and_home_expansion(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = Path(d) / 'paths.json'
            cfg.write_text('{"paths": {"model": "unused"}}')
            with patch.dict(os.environ, {'WE_MODEL': d, 'WE_TEST_ROOT': d}):
                self.assertEqual(load_config(cfg)['paths']['model'], d)
                self.assertEqual(resolve_path('${WE_TEST_ROOT}/model', cfg.parent), str(Path(d) / 'model'))

    def test_python_overrides_keep_the_virtual_environment_executable(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            base = root / 'base-python'
            base.touch()
            venv = root / 'venv/bin/python'
            venv.parent.mkdir(parents=True)
            venv.symlink_to(base)
            cfg = root / 'paths.json'
            cfg.write_text('{"paths": {"python": null, "difix_python": "venv/bin/python"}}')
            with patch.dict(os.environ, {'WE_PYTHON': str(venv)}):
                paths = load_config(cfg)['paths']
            self.assertEqual(paths['python'], str(venv))
            self.assertEqual(paths['difix_python'], str(venv))

    def test_unset_env_in_path_is_an_error(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, 'Unresolved'):
                resolve_path('${WE_MISSING}/model', Path('/tmp'))

    def test_case_id_cannot_escape_input_or_output_root(self):
        for value in ('../case', '/tmp/case', 'a/b', '', '.', '..', 'a\\b'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_case_id(value)
        self.assertEqual(validate_case_id('case-004_bb'), 'case-004_bb')

    def test_checkpoint_attention_schedule_overrides_release_profile(self):
        from omegaconf import OmegaConf
        from worldexpand.cli import prepare_inference

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'checkpoint/lora').mkdir(parents=True)
            (root / 'start/case').mkdir(parents=True)
            (root / 'warp/case').mkdir(parents=True)
            (root / 'start/case/origin.png').touch()
            (root / 'warp/case/erp_aux_0.png').touch()
            (root / 'warp/case/mask_aux_0.png').touch()
            profile = root / 'profile.yaml'
            profile.write_text('num_views: 1\nlocal_layer_indices: "2,3"\ndata_config: {}\n')
            cfg = {'paths': {'model': str(root), 'checkpoint': str(root / 'checkpoint'),
                             'training_config': str(profile), 'start_root': str(root / 'start'),
                             'warp_root': str(root / 'warp')}}
            variants = (({}, '2,3'), ({'local_layer_indices': '1,4'}, '1,4'),
                        ({'local_layer_indices': None}, None))
            for meta, expected in variants:
                with self.subTest(meta=meta):
                    (root / 'checkpoint/meta.json').write_text(json.dumps(meta))
                    prepare_inference(cfg, root, ['case'], {})
                    self.assertEqual(OmegaConf.load(root / 'runtime.yaml').local_layer_indices, expected)

    def test_expected_outputs_require_every_view(self):
        out = expected_outputs(Path('/tmp/out'), ['case'], 4)
        self.assertEqual(len(out), 4)
        self.assertEqual(out[-1], Path('/tmp/out/case/aux_3/output.png'))


if __name__ == '__main__':
    unittest.main()

class ResourceMappingTests(unittest.TestCase):
    def test_longest_mapping_wins(self):
        from worldexpand.paths import map_prefix
        mapping = {'/old': '/new', '/old/model': '/weights'}
        self.assertEqual(map_prefix('/old/model/a.bin', mapping), '/weights/a.bin')
        self.assertEqual(map_prefix('/older/a.bin', mapping), '/older/a.bin')

    def test_repo_references_follow_the_release(self):
        from worldexpand.paths import resolve_resource
        from worldexpand.config import REPO_ROOT
        self.assertEqual(resolve_resource('@repo/configs/release/local.json'),
                         str(REPO_ROOT / 'configs/release/local.json'))

class OptionalPathTests(unittest.TestCase):
    def test_null_optional_path_can_be_supplied_from_environment(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = Path(d) / 'paths.json'
            cfg.write_text('{"paths": {"train_json": null}}')
            with patch.dict(os.environ, {'WE_TRAIN_JSON': d + '/train.json'}):
                self.assertEqual(load_config(cfg)['paths']['train_json'], d + '/train.json')

import json
import unittest
from pathlib import Path
from typing import Any, Optional

from threedattack.rendering_core import SceneConfDictKey, get_scene_config_errors


class TestSceneConfig(unittest.TestCase):
    def setUp(self):
        test_conf_path = (
            Path(__file__).resolve().parent.parent.parent
            / "test_resources"
            / "test_scene.json"
        )
        self.test_conf = json.loads(test_conf_path.read_text())

    def test_happy_path(self):
        self.assert_no_error(self.test_conf)

    def test_missing_key(self):
        for key in self.test_conf.keys():
            with self.subTest(key):
                error_conf = self.test_conf.copy()
                del error_conf[key]
                self.assert_incorrect_conf(error_conf, key)

    def test_extra_key(self):
        extra_key = "extra"
        self.test_conf[extra_key] = 5
        self.assert_incorrect_conf(self.test_conf, extra_key)

    def test_incorrect_type(self):
        self.assert_incorrect_conf([])

    def test_invalid_value(self):
        invalid_values: list[tuple[SceneConfDictKey, Any]] = [
            ("shadow_blur", -1.0),
            ("shadow_blur", None),
            ("shadow_blur", 3),
            ("shadow_map_resolution", 0.5),
            ("shadow_map_resolution", -1.0),
            ("dir_ambient_hpr_weight", -1.0),
            ("dir_ambient_hpr_weight", 1.1),
            ("nondir_ambient_light_r", -1.0),
            ("nondir_ambient_light_g", -1.0),
            ("nondir_ambient_light_b", -1.0),
            ("shadow_area_root", "**/+Light"),
            ("force_shadow_0", 13),
            ("force_shadow_0", ["**/+Light"]),
        ]
        for key, invalid_value in invalid_values:
            if key == "shadow_area_root":
                ai = 2
            with self.subTest(f"{key}={invalid_value}"):
                incorrect_conf = self.test_conf.copy()

                incorrect_conf[key] = invalid_value

                self.assert_incorrect_conf(incorrect_conf)

    def assert_incorrect_conf(self, conf: Any, keyword: Optional[str] = None) -> None:
        errors = get_scene_config_errors(conf)
        self.assertGreaterEqual(len(errors), 1)

        if keyword is not None:
            self.assertTrue(any((keyword in msg) for msg in errors))

    def assert_no_error(self, conf: Any) -> None:
        errors = get_scene_config_errors(conf)
        self.assertEqual(errors, [])

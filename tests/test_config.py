import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config_manager import ConfigError, ConfigManager


@pytest.fixture()
def minimal_config(tmp_path) -> str:
    cfg = {
        "camera_index": 0,
        "frame_width": 640,
        "frame_height": 480,
        "min_contour_area": 500,
        "blur_kernel_size": 5,
        "tracker_max_disappeared": 30,
        "tracker_max_distance": 80,
        "trajectory_max_length": 50,
        "colors": {
            "blue": {
                "lower": [100, 150, 50],
                "upper": [130, 255, 255],
                "display_color": [255, 0, 0],
                "dual_range": False,
            }
        }
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")
    return str(path)


@pytest.fixture()
def red_dual_config(tmp_path) -> str:
    cfg = {
        "colors": {
            "red": {
                "lower":  [0,   120, 70],
                "upper":  [10,  255, 255],
                "lower2": [170, 120, 70],
                "upper2": [180, 255, 255],
                "display_color": [0, 0, 255],
                "dual_range": True,
            }
        }
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")
    return str(path)


class TestConfigManagerLoading:
    def test_loads_valid_config(self, minimal_config):
        cfg = ConfigManager(minimal_config)
        assert cfg.camera_index == 0
        assert cfg.frame_width == 640
        assert cfg.frame_height == 480
        assert cfg.min_contour_area == 500

    def test_colors_property_returns_dict(self, minimal_config):
        cfg = ConfigManager(minimal_config)
        colors = cfg.colors
        assert isinstance(colors, dict)
        assert "blue" in colors

    def test_file_not_found_raises(self):
        with pytest.raises(ConfigError, match="not found"):
            ConfigManager("/nonexistent/path/config.json")

    def test_invalid_json_raises(self, tmp_path):
        bad = tmp_path / "config.json"
        bad.write_text("{NOT VALID JSON}", encoding="utf-8")
        with pytest.raises(ConfigError, match="Invalid JSON"):
            ConfigManager(str(bad))

    def test_missing_colors_key_raises(self, tmp_path):
        bad = tmp_path / "config.json"
        bad.write_text('{"camera_index": 0}', encoding="utf-8")
        with pytest.raises(ConfigError, match="at least one color"):
            ConfigManager(str(bad))


class TestConfigManagerValidation:
    def test_invalid_hue_value_raises(self, tmp_path):
        cfg = {
            "colors": {
                "bad": {
                    "lower": [200, 0, 0],
                    "upper": [179, 255, 255],
                    "display_color": [0, 0, 0],
                    "dual_range": False,
                }
            }
        }
        path = tmp_path / "config.json"
        path.write_text(json.dumps(cfg), encoding="utf-8")
        with pytest.raises(ConfigError, match="out of range"):
            ConfigManager(str(path))

    def test_dual_range_missing_lower2_raises(self, tmp_path):
        cfg = {
            "colors": {
                "red": {
                    "lower": [0, 120, 70],
                    "upper": [10, 255, 255],
                    "display_color": [0, 0, 255],
                    "dual_range": True,
                }
            }
        }
        path = tmp_path / "config.json"
        path.write_text(json.dumps(cfg), encoding="utf-8")
        with pytest.raises(ConfigError, match="lower2"):
            ConfigManager(str(path))


class TestConfigManagerColorFilter:
    def test_get_active_colors_all(self, minimal_config):
        cfg = ConfigManager(minimal_config)
        active = cfg.get_active_colors(None)
        assert "blue" in active

    def test_get_active_colors_filtered(self, minimal_config):
        cfg = ConfigManager(minimal_config)
        active = cfg.get_active_colors(["blue"])
        assert "blue" in active

    def test_get_active_colors_unknown_filter(self, minimal_config):
        cfg = ConfigManager(minimal_config)
        active = cfg.get_active_colors(["nonexistent"])
        assert active == {}


class TestConfigManagerSave:
    def test_update_color_saves_to_disk(self, minimal_config):
        cfg = ConfigManager(minimal_config)
        new_lower = [110, 160, 60]
        new_upper = [125, 240, 240]
        cfg.update_color("blue", new_lower, new_upper)

        cfg2 = ConfigManager(minimal_config)
        assert cfg2.colors["blue"]["lower"] == new_lower
        assert cfg2.colors["blue"]["upper"] == new_upper

    def test_blur_kernel_forced_odd(self, tmp_path):
        cfg_data = {
            "blur_kernel_size": 4,
            "colors": {
                "blue": {
                    "lower": [100, 150, 50],
                    "upper": [130, 255, 255],
                    "display_color": [255, 0, 0],
                    "dual_range": False,
                }
            }
        }
        path = tmp_path / "config.json"
        path.write_text(json.dumps(cfg_data), encoding="utf-8")
        cfg = ConfigManager(str(path))
        assert cfg.blur_kernel_size % 2 == 1

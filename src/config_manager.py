import json
import os
import copy
from typing import Any


class ConfigError(Exception):
    pass


class ConfigManager:
    _HSV_BOUNDS = {"H": (0, 179), "S": (0, 255), "V": (0, 255)}

    def __init__(self, config_path: str = "config.json"):
        self.config_path = os.path.abspath(config_path)
        self._data: dict[str, Any] = {}
        self._load()
        self._validate()

    @property
    def camera_index(self) -> int:
        return int(self._data.get("camera_index", 0))

    @property
    def frame_width(self) -> int:
        return int(self._data.get("frame_width", 640))

    @property
    def frame_height(self) -> int:
        return int(self._data.get("frame_height", 480))

    @property
    def min_contour_area(self) -> int:
        return int(self._data.get("min_contour_area", 500))

    @property
    def blur_kernel_size(self) -> int:
        value = int(self._data.get("blur_kernel_size", 5))
        return value if value % 2 == 1 else value + 1

    @property
    def tracker_max_disappeared(self) -> int:
        return int(self._data.get("tracker_max_disappeared", 30))

    @property
    def tracker_max_distance(self) -> int:
        return int(self._data.get("tracker_max_distance", 80))

    @property
    def trajectory_max_length(self) -> int:
        return int(self._data.get("trajectory_max_length", 50))

    @property
    def colors(self) -> dict:
        return copy.deepcopy(self._data.get("colors", {}))

    def get_active_colors(self, filter_names: list[str] | None = None) -> dict:
        all_colors = self.colors
        if not filter_names:
            return all_colors
        return {k: v for k, v in all_colors.items() if k in filter_names}

    def update_color(self, color_name: str, lower: list, upper: list,
                     lower2: list | None = None, upper2: list | None = None):
        self._validate_hsv_range(lower, upper, color_name)
        entry = self._data["colors"].setdefault(color_name, {})
        entry["lower"] = lower
        entry["upper"] = upper
        entry["dual_range"] = lower2 is not None
        if lower2 is not None and upper2 is not None:
            self._validate_hsv_range(lower2, upper2, f"{color_name}_2")
            entry["lower2"] = lower2
            entry["upper2"] = upper2
        self.save()

    def save(self):
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=4)

    def _load(self):
        if not os.path.isfile(self.config_path):
            raise ConfigError(f"Config file not found: {self.config_path}")
        with open(self.config_path, "r", encoding="utf-8") as f:
            try:
                self._data = json.load(f)
            except json.JSONDecodeError as e:
                raise ConfigError(f"Invalid JSON in config file: {e}") from e

    def _validate(self):
        if "colors" not in self._data or not self._data["colors"]:
            raise ConfigError("Config must define at least one color.")

        for name, cfg in self._data["colors"].items():
            for key in ("lower", "upper", "display_color"):
                if key not in cfg:
                    raise ConfigError(
                        f"Color '{name}' is missing required key '{key}'."
                    )
            self._validate_hsv_range(cfg["lower"], cfg["upper"], name)
            if cfg.get("dual_range"):
                if "lower2" not in cfg or "upper2" not in cfg:
                    raise ConfigError(
                        f"Color '{name}' has dual_range=true but is missing "
                        "'lower2' or 'upper2'."
                    )
                self._validate_hsv_range(cfg["lower2"], cfg["upper2"],
                                         f"{name}_2")

    @staticmethod
    def _validate_hsv_range(lower: list, upper: list, name: str):
        bounds = [("H", 0, 179), ("S", 0, 255), ("V", 0, 255)]
        for (ch, lo, hi), lv, uv in zip(bounds, lower, upper):
            if not (lo <= lv <= hi):
                raise ConfigError(
                    f"Color '{name}': lower {ch}={lv} out of range [{lo}, {hi}]"
                )
            if not (lo <= uv <= hi):
                raise ConfigError(
                    f"Color '{name}': upper {ch}={uv} out of range [{lo}, {hi}]"
                )

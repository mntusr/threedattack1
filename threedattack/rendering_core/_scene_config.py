import json
import re
from itertools import filterfalse
from pathlib import Path
from typing import Any, Literal, TypedDict, TypeGuard

SceneConfDictKey = Literal[
    "shadow_blur",
    "shadow_map_resolution",
    "nondir_ambient_light_r",
    "nondir_ambient_light_g",
    "nondir_ambient_light_b",
    "dir_ambient_hpr_weight",
    "dir_ambient_hpr_diff",
    "shadow_area_root",
    "force_shadow_0",
]


class SceneConfigDict(TypedDict):
    shadow_blur: float
    shadow_map_resolution: int
    nondir_ambient_light_r: float
    nondir_ambient_light_g: float
    nondir_ambient_light_b: float
    dir_ambient_hpr_weight: float
    dir_ambient_hpr_diff: float
    shadow_area_root: str
    force_shadow_0: list[str]
    world_size_x: float
    world_size_y: float
    world_size_z: float


def load_scene_config(path: Path) -> SceneConfigDict | list[str]:
    json_data = json.loads(path.read_text())
    scene_config_errors = get_scene_config_errors(json_data)
    if len(scene_config_errors) == 0:
        return json_data
    else:
        return scene_config_errors


def get_scene_config_errors(conf_dict: object) -> list[str]:
    errors: list[str] = []

    if not isinstance(conf_dict, dict):
        return ["The json data does not contain a valid dictionary."]

    def check_key_type_and_not_none(key: str, target_type: type) -> bool:
        if key is None:
            return False
        if not isinstance(conf_dict[key], target_type):
            errors.append(f'The key "{key}" does not have type "{target_type}".')
            return False
        return True

    def check_val_not_neg(key: str):
        if conf_dict[key] < 0:
            errors.append(f'The key "{key}" contains a negative value.')

    def check_val_not_in_01_range(key: str):
        if not (0 <= conf_dict[key] <= 1):
            errors.append(f'0<="{key}"<=1 does not hold.')

    NAME_PATTERN = "^[a-zA-Z0-9_]+$"

    def is_name(obj: object) -> bool:
        if not isinstance(obj, str):
            return False
        if not re.match(NAME_PATTERN, obj):
            return False

        return True

    def check_is_name(obj: object) -> bool:
        if not is_name(obj):
            errors.append(
                f'The name {repr(obj)} is not a string or does not match to regular expression "{NAME_PATTERN}"'
            )
            return False
        else:
            return True

    expected_key_set = set(conf_dict.keys())
    actual_key_set = set(SceneConfigDict.__required_keys__)
    if expected_key_set != actual_key_set:
        return [
            f"Scene configuration error. The expected key set {expected_key_set} and the actual key set {actual_key_set} do not match."
        ]

    if check_key_type_and_not_none("shadow_blur", float):
        check_val_not_neg("shadow_blur")
    if check_key_type_and_not_none("shadow_map_resolution", int):
        check_val_not_neg("shadow_map_resolution")
    if check_key_type_and_not_none("nondir_ambient_light_r", float):
        check_val_not_neg("nondir_ambient_light_r")
    if check_key_type_and_not_none("nondir_ambient_light_g", float):
        check_val_not_neg("nondir_ambient_light_g")
    if check_key_type_and_not_none("nondir_ambient_light_b", float):
        check_val_not_neg("nondir_ambient_light_b")
    if check_key_type_and_not_none("dir_ambient_hpr_weight", float):
        check_val_not_in_01_range("dir_ambient_hpr_weight")
    if check_key_type_and_not_none("dir_ambient_hpr_diff", int):
        check_val_not_neg("dir_ambient_hpr_diff")
    if check_key_type_and_not_none("world_size_x", float):
        check_val_not_neg("world_size_x")
    if check_key_type_and_not_none("world_size_y", float):
        check_val_not_neg("world_size_y")
    if check_key_type_and_not_none("world_size_z", float):
        check_val_not_neg("world_size_z")
    if check_key_type_and_not_none("force_shadow_0", list):
        for val in conf_dict["force_shadow_0"]:
            check_is_name(val)
    check_is_name(conf_dict["shadow_area_root"])

    return errors

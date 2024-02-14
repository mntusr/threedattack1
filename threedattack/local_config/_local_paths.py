import json
import shutil
from pathlib import Path
from typing import Any, NamedTuple, TypedDict


def get_local_config_json() -> "LocalConfig":
    config_json_path = Path(__file__).parent.parent.parent / "local_config.json"

    if not config_json_path.exists():
        default_config_path = config_json_path.with_name("local_config_default.json")
        shutil.copy(src=default_config_path, dst=config_json_path)

    json_data = json.loads(config_json_path.read_text())
    errors = get_local_paths_json_errors(json_data)

    if len(errors) > 0:
        raise Exception(f"Incorrect paths json. Errors: {errors}")

    return LocalConfig(
        mlflow_tracking_url=json_data["mlflow_tracking_url"],
        nyuv2_labeled_mat=json_data["nyuv2_labeled_mat"],
        nyuv2_splits_mat=json_data["nyuv2_splits_mat"],
    )


def get_local_paths_json_errors(data: object) -> list[str]:
    if not isinstance(data, dict):
        return [
            f"The local paths json file does not contain a single dictionary. Data: {data}"
        ]

    actual_keys = set(data.keys())
    expected_keys = set(LocalConfig._fields)
    if not actual_keys == expected_keys:
        return [
            f"The key set of the loaded json data {actual_keys} is not equal to the expected set of keys {expected_keys}"
        ]

    errors = []
    for key in expected_keys:
        if not isinstance(data[key], str):
            errors.append(f'The path value for key "{key}" is not a string.')

    return errors


class LocalConfig(NamedTuple):
    nyuv2_splits_mat: str
    nyuv2_labeled_mat: str
    mlflow_tracking_url: str

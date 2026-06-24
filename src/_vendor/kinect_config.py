"""Vendored KinectConfigFileHandler and KinectConfig from the parent repo's primary_processing package."""

import re
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict


class UnresolvedVariableError(Exception):
    """Custom exception for unresolved variables."""
    pass

class CircularDependencyError(Exception):
    """Custom exception for circular dependencies."""
    pass


class KinectConfigFileHandler:
    """
    Handles the loading and resolution of YAML configuration files with variable placeholders.
    """

    @staticmethod
    def _flatten_dict_for_resolution(data: Dict[str, Any], prefix: str = '') -> Dict[str, str]:
        items = {}
        for key, value in data.items():
            new_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                items.update(KinectConfigFileHandler._flatten_dict_for_resolution(value, new_key))
            elif isinstance(value, str):
                items[key] = value
        return items

    @staticmethod
    def _recursive_substitute(data: Any, resolved_vars: Dict[str, str]) -> Any:
        if isinstance(data, dict):
            return {k: KinectConfigFileHandler._recursive_substitute(v, resolved_vars) for k, v in data.items()}
        elif isinstance(data, list):
            return [KinectConfigFileHandler._recursive_substitute(item, resolved_vars) for item in data]
        elif isinstance(data, str):
            for _ in range(len(resolved_vars)):
                for placeholder, value in resolved_vars.items():
                    data = data.replace(f"{{{placeholder}}}", value)
            return data
        else:
            return data

    @staticmethod
    def load_and_resolve_config(filepath: str) -> Dict[str, Any]:
        with open(filepath, 'r') as f:
            config_data = yaml.safe_load(f)

        variables = KinectConfigFileHandler._flatten_dict_for_resolution(config_data)

        max_iterations = len(variables) + 1
        for i in range(max_iterations):
            changed_in_pass = False
            for key, value in variables.items():
                placeholders = re.findall(r'\{([^}]+)\}', value)
                if not placeholders:
                    continue

                for placeholder in placeholders:
                    if placeholder in variables:
                        sub_value = variables[placeholder]
                        if '{' not in sub_value:
                            new_value = value.replace(f"{{{placeholder}}}", sub_value)
                            if new_value != value:
                                variables[key] = new_value
                                value = new_value
                                changed_in_pass = True

            if not changed_in_pass:
                break
        else:
            raise CircularDependencyError("A circular dependency was detected in the configuration file.")

        final_unresolved = [
            p for val in variables.values()
            for p in re.findall(r'\{([^}]+)\}', val)
            if p in variables
        ]

        if final_unresolved:
            raise UnresolvedVariableError(f"Could not resolve variables: {set(final_unresolved)}")

        return KinectConfigFileHandler._recursive_substitute(config_data, variables)


class SessionInputs(BaseModel):
    """
    The Pydantic data model defining the schema for session inputs.
    """
    model_config = ConfigDict(extra='allow')

    session_id: str
    block_id: str
    objects_to_track: List[str]
    source_video: str
    stimulus_metadata: str
    hand_models_dir: str

    nerve_primary_dir: Optional[str] = None
    nerve_processed_dir: Optional[str] = None
    session_merged_output_dir: Optional[str] = None

    video_primary_output_dir: str
    video_processed_output_dir: str
    session_primary_output_dir: str
    session_processed_output_dir: str


class KinectConfig:
    """
    Manages and provides validated access to session configuration data.
    Automatically resolves relative paths against a provided database root.
    """
    def __init__(self, config_data: Dict[str, Any], database_path: Path):
        from pydantic import ValidationError
        try:
            if not config_data:
                raise ValueError("Configuration data is empty.")
            self.settings = SessionInputs(**config_data)
            self._database_path = database_path
            self._non_path_keys = {"session_id", "block_id", "objects_to_track"}
        except ValidationError as e:
            raise ValueError(f"Session configuration validation failed: \n{e}") from e

    def _resolve_path(self, value: Any) -> Any:
        if isinstance(value, str) and value:
            return self._database_path / value
        return value

    def __getattr__(self, name: str) -> Any:
        if hasattr(self.settings, name):
            value = getattr(self.settings, name)
            if name in self._non_path_keys:
                return value
            return self._resolve_path(value)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    @property
    def database_path(self) -> Path:
        return self._database_path

    @property
    def session_id(self) -> str:
        return self.settings.session_id

    @property
    def block_id(self) -> str:
        return self.settings.block_id

    @property
    def objects_to_track(self) -> List[str]:
        return self.settings.objects_to_track

    @property
    def source_video(self) -> Path:
        return self._resolve_path(self.settings.source_video)

    @property
    def stimulus_metadata(self) -> Path:
        return self._resolve_path(self.settings.stimulus_metadata)

    @property
    def hand_models_dir(self) -> Path:
        return self._resolve_path(self.settings.hand_models_dir)

    @property
    def nerve_primary_dir(self) -> Optional[Path]:
        if self.settings.nerve_primary_dir:
            return self._resolve_path(self.settings.nerve_primary_dir)
        return None

    @property
    def nerve_processed_dir(self) -> Optional[Path]:
        if self.settings.nerve_processed_dir:
            return self._resolve_path(self.settings.nerve_processed_dir)
        return None

    @property
    def session_merged_output_dir(self) -> Optional[Path]:
        if self.settings.session_merged_output_dir:
            return self._resolve_path(self.settings.session_merged_output_dir)
        return None

    @property
    def video_primary_output_dir(self) -> Path:
        return self._resolve_path(self.settings.video_primary_output_dir)

    @property
    def video_processed_output_dir(self) -> Path:
        return self._resolve_path(self.settings.video_processed_output_dir)

    @property
    def session_primary_output_dir(self) -> Path:
        return self._resolve_path(self.settings.session_primary_output_dir)

    @property
    def session_processed_output_dir(self) -> Path:
        return self._resolve_path(self.settings.session_processed_output_dir)

    def __repr__(self) -> str:
        return f"<KinectConfig session_id='{self.session_id}' block_id='{self.block_id}' source_video='{self.settings.source_video}'>"

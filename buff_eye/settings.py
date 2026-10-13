"""Versioned persistent settings stored outside the application directory."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .overlay_model import ROI_IDS, CaptureRegion, OverlayLayout


CURRENT_SCHEMA_VERSION = 2
APP_DATA_DIRECTORY = "LostArkBuffEye"


class SettingsError(RuntimeError):
    pass


class UnsupportedSettingsVersion(SettingsError):
    pass


def user_data_directory(
    *, environ: Mapping[str, str] | None = None, home: Path | None = None
) -> Path:
    """Return a per-Windows-user path independent of the executable location."""
    env = os.environ if environ is None else environ
    local = env.get("LOCALAPPDATA")
    base = Path(local) if local else (Path.home() if home is None else home) / "AppData" / "Local"
    return base / APP_DATA_DIRECTORY


def default_settings() -> dict[str, Any]:
    return {
        "schema_version": CURRENT_SCHEMA_VERSION,
        "selected_profile_id": "default",
        "display": {
            "capture_monitor_id": None,
            "overlay_monitor_id": None,
            "settings_monitor_id": None,
        },
        "profiles": {
            "default": {
                "green_buffs": {},
                "game_ui": {
                    "hud_scale_pct": 100,
                    "buff_scale_pct": 100,
                    "target_scale_pct": 100,
                    "buff_display_count": 15,
                    "target_display_count": 17,
                    "stack_merge": True,
                },
                "rois": [CaptureRegion(region_id).to_dict() for region_id in ROI_IDS],
                "overlay": OverlayLayout().to_dict(),
                "tracked_effects": [],
            }
        },
    }


def _fill_missing(target: dict[str, Any], defaults: Mapping[str, Any]) -> dict[str, Any]:
    """Add new defaults without deleting or overwriting user/extension fields."""
    for key, value in defaults.items():
        if key not in target:
            target[key] = copy.deepcopy(value)
        elif isinstance(value, Mapping) and isinstance(target[key], dict):
            _fill_missing(target[key], value)
    return target


def _migrate_v1_to_v2(source: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(source)
    profile = result.setdefault("profiles", {}).setdefault("default", {})
    display = result.setdefault("display", {})

    # Version 1 prototypes kept these fields at the document root. Keep any
    # unrelated keys verbatim while moving known fields into their stable homes.
    if "monitor_id" in result:
        display.setdefault("capture_monitor_id", result.pop("monitor_id"))
    if "rois" in result:
        profile.setdefault("rois", result.pop("rois"))
    overlay = profile.setdefault("overlay", {})
    if "position_locked" in result:
        overlay.setdefault("position_locked", result.pop("position_locked"))
    if "icons" in result:
        overlay.setdefault("icons", result.pop("icons"))
    result["schema_version"] = 2
    return result


def migrate_settings(source: Mapping[str, Any]) -> tuple[dict[str, Any], bool]:
    if not isinstance(source, Mapping):
        raise SettingsError("settings root must be an object")
    result = copy.deepcopy(dict(source))
    original = copy.deepcopy(result)
    version = result.get("schema_version", 1)
    if type(version) is not int or version < 1:
        raise SettingsError("schema_version must be a positive integer")
    if version > CURRENT_SCHEMA_VERSION:
        raise UnsupportedSettingsVersion(
            f"settings schema {version} is newer than supported schema {CURRENT_SCHEMA_VERSION}"
        )
    while version < CURRENT_SCHEMA_VERSION:
        if version == 1:
            result = _migrate_v1_to_v2(result)
        else:  # pragma: no cover - guarded by current sequential migration table
            raise UnsupportedSettingsVersion(f"no migration from schema {version}")
        version = result["schema_version"]
    _fill_missing(result, default_settings())
    validate_settings(result)
    return result, result != original


def validate_settings(value: Mapping[str, Any]) -> None:
    if value.get("schema_version") != CURRENT_SCHEMA_VERSION:
        raise UnsupportedSettingsVersion("settings are not at the current schema")
    display = value.get("display")
    profiles = value.get("profiles")
    selected = value.get("selected_profile_id")
    if not isinstance(display, Mapping):
        raise SettingsError("display must be an object")
    for key in ("capture_monitor_id", "overlay_monitor_id", "settings_monitor_id"):
        monitor = display.get(key)
        if monitor is not None and not isinstance(monitor, str):
            raise SettingsError(f"{key} must be a string or null")
    if not isinstance(profiles, Mapping) or not profiles:
        raise SettingsError("profiles must be a non-empty object")
    if not isinstance(selected, str) or selected not in profiles:
        raise SettingsError("selected_profile_id must name an existing profile")
    for profile_id, profile in profiles.items():
        if not isinstance(profile_id, str) or not isinstance(profile, Mapping):
            raise SettingsError("invalid profile")
        rois = profile.get("rois")
        if not isinstance(rois, list):
            raise SettingsError("profile rois must be an array")
        game_ui = profile.get("game_ui")
        if not isinstance(game_ui, Mapping):
            raise SettingsError("profile game_ui must be an object")
        allowed = {
            "hud_scale_pct": {70, 90, 100, 105, 110},
            "buff_scale_pct": {80, 90, 100, 110, 120},
            "target_scale_pct": {80, 90, 100, 110, 120},
        }
        for key, choices in allowed.items():
            if type(game_ui.get(key)) is not int or game_ui[key] not in choices:
                raise SettingsError(f"{key} must be one of {sorted(choices)}")
        for key in ("buff_display_count", "target_display_count"):
            if type(game_ui.get(key)) is not int or not 10 <= game_ui[key] <= 17:
                raise SettingsError(f"{key} must be between 10 and 17")
        if type(game_ui.get("stack_merge")) is not bool:
            raise SettingsError("stack_merge must be boolean")
        effects = profile.get("tracked_effects")
        if not isinstance(effects, list):
            raise SettingsError("tracked_effects must be an array")
        effect_ids = set()
        for effect in effects:
            if not isinstance(effect, Mapping) or not isinstance(effect.get("effect_id"), str) or not effect["effect_id"]:
                raise SettingsError("tracked effect needs a non-empty effect_id")
            if effect["effect_id"] in effect_ids:
                raise SettingsError("tracked effect IDs must be unique")
            effect_ids.add(effect["effect_id"])
            variants = effect.get("template_variants", [])
            if not isinstance(variants, list) or not all(isinstance(path, str) and path for path in variants):
                raise SettingsError("template_variants must be non-empty path strings")
            scales = effect.get("template_variant_scales", {})
            if not isinstance(scales, Mapping):
                raise SettingsError("template_variant_scales must be an object")
            if any(
                path not in variants or scale not in (80, 90, 100, 110, 120)
                for path, scale in scales.items()
            ):
                raise SettingsError("template variant scales must reference a variant and supported scale")
        try:
            regions = [CaptureRegion.from_dict(region) for region in rois]
            OverlayLayout.from_dict(profile.get("overlay", {}))
        except (KeyError, TypeError, ValueError) as exc:
            raise SettingsError(f"invalid profile {profile_id}: {exc}") from exc
        ids = [region.region_id for region in regions]
        if sorted(ids) != sorted(ROI_IDS):
            raise SettingsError(f"profile {profile_id} must contain each ROI exactly once")


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


class SettingsStore:
    """Atomic settings persistence with immutable timestamped backups."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = user_data_directory() if root is None else Path(root)
        self.path = self.root / "settings.json"
        self.backup_directory = self.root / "backups"

    def _backup_original(self, *, reason: str) -> Path | None:
        if not self.path.exists():
            return None
        content = self.path.read_bytes()
        digest = hashlib.sha256(content).hexdigest()[:12]
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        destination = self.backup_directory / f"settings.{reason}.{stamp}.{digest}.json"
        self.backup_directory.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".tmp")
        try:
            with self.path.open("rb") as source, temporary.open("xb") as target:
                shutil.copyfileobj(source, target)
                target.flush()
                os.fsync(target.fileno())
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        return destination

    def load(self) -> dict[str, Any]:
        if not self.path.exists():
            value = default_settings()
            self.save(value, backup=False)
            return value
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise SettingsError(f"cannot read settings: {exc}") from exc

        version = raw.get("schema_version", 1) if isinstance(raw, dict) else None
        needs_migration = isinstance(version, int) and version < CURRENT_SCHEMA_VERSION
        backup = self._backup_original(reason=f"schema-v{version}") if needs_migration else None
        try:
            migrated, changed = migrate_settings(raw)
            if changed:
                self.save(migrated, backup=False)
            return migrated
        except Exception as exc:
            location = f"; original backup: {backup}" if backup else ""
            if isinstance(exc, SettingsError):
                raise type(exc)(f"{exc}{location}") from exc
            raise SettingsError(f"settings migration failed: {exc}{location}") from exc

    def save(self, value: Mapping[str, Any], *, backup: bool = True) -> None:
        document = copy.deepcopy(dict(value))
        validate_settings(document)
        try:
            if backup:
                self._backup_original(reason="last-good")
            _atomic_write(self.path, _json_bytes(document))
        except OSError as exc:
            raise SettingsError(f"cannot save settings: {exc}") from exc

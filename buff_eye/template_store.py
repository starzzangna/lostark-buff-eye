"""Import user effect templates into persistent application data."""
from __future__ import annotations

import copy
import hashlib
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np

from .settings import SettingsStore


MAX_IMAGE_PIXELS = 64_000_000
SUPPORTED_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


@dataclass(frozen=True)
class RegisteredTemplate:
    effect_id: str
    relative_path: str
    absolute_path: Path
    variant_added: bool
    asset_created: bool
    scale_pct: int


def _validate_effect_id(effect_id: str) -> str:
    if not isinstance(effect_id, str) or not effect_id.strip():
        raise ValueError("effect_id must be a non-empty string")
    return effect_id.strip()


def _effect_directory(effect_id: str) -> str:
    readable = re.sub(r"[^A-Za-z0-9._-]+", "-", effect_id).strip(".-_")[:40] or "effect"
    identity = hashlib.sha256(effect_id.encode("utf-8")).hexdigest()[:12]
    return f"{readable}-{identity}"


def _decode(content: bytes) -> np.ndarray:
    if not content:
        raise ValueError("template image is empty")
    image = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("template file is not a supported image")
    height, width = image.shape[:2]
    if height <= 0 or width <= 0 or height * width > MAX_IMAGE_PIXELS:
        raise ValueError("template image dimensions are unsupported")
    return image


def _atomic_create(path: Path, content: bytes) -> bool:
    """Create a content-addressed asset atomically; return whether it was new."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise RuntimeError(f"content-address collision: {path}")
        return False
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != content:
                raise RuntimeError(f"content-address collision: {path}")
            return False
        return True
    finally:
        temporary.unlink(missing_ok=True)


class TemplateRegistrationService:
    """Persist file or preview-crop variants beneath the settings root."""

    def __init__(self, settings_store: SettingsStore | None = None) -> None:
        self.settings_store = settings_store or SettingsStore()
        self.templates_root = self.settings_store.root / "templates"

    @staticmethod
    def _profile(document: dict[str, Any], profile_id: str | None):
        selected = document["selected_profile_id"] if profile_id is None else profile_id
        profiles = document["profiles"]
        if selected not in profiles:
            raise ValueError(f"unknown profile: {selected}")
        return profiles[selected]

    def _register(self, effect_id: str, content: bytes, suffix: str,
                  profile_id: str | None, scale_pct: int) -> RegisteredTemplate:
        effect_id = _validate_effect_id(effect_id)
        image = _decode(content)
        if scale_pct not in (80, 90, 100, 110, 120):
            raise ValueError("scale_pct must be one of 80, 90, 100, 110, 120")
        height, width = image.shape[:2]
        if max(width, height) / min(width, height) > 1.35:
            raise ValueError("template image must be an icon crop with a near-square aspect ratio")
        suffix = suffix.lower()
        if suffix not in SUPPORTED_SUFFIXES:
            raise ValueError(f"unsupported template suffix: {suffix}")
        digest = hashlib.sha256(content).hexdigest()
        relative = Path("templates") / _effect_directory(effect_id) / f"{digest}{suffix}"
        destination = self.settings_store.root / relative
        document = self.settings_store.load()
        updated = copy.deepcopy(document)
        profile = self._profile(updated, profile_id)
        tracked = profile.setdefault("tracked_effects", [])
        definition = next((item for item in tracked
                           if isinstance(item, Mapping) and item.get("effect_id") == effect_id), None)
        if definition is None:
            definition = {
                "effect_id": effect_id,
                "enabled": True,
                "registration_only": True,
                "template_variants": [],
            }
            tracked.append(definition)
        variants = definition.setdefault("template_variants", [])
        stored_path = relative.as_posix()
        variant_added = stored_path not in variants
        if variant_added:
            variants.append(stored_path)
        scales = definition.setdefault("template_variant_scales", {})
        scale_changed = scales.get(stored_path) != scale_pct
        scales[stored_path] = scale_pct

        created = _atomic_create(destination, content)
        try:
            if variant_added or scale_changed:
                self.settings_store.save(updated)
        except BaseException:
            if created:
                destination.unlink(missing_ok=True)
            raise
        return RegisteredTemplate(
            effect_id, stored_path, destination, variant_added, created, scale_pct
        )

    def register_file(self, effect_id: str, source_path: str | Path,
                      *, profile_id: str | None = None,
                      scale_pct: int = 100) -> RegisteredTemplate:
        source = Path(source_path)
        suffix = source.suffix.lower()
        if suffix not in SUPPORTED_SUFFIXES:
            raise ValueError(f"unsupported template suffix: {suffix or '<none>'}")
        try:
            content = source.read_bytes()
        except OSError as exc:
            raise ValueError(f"cannot read template image: {source}") from exc
        return self._register(effect_id, content, suffix, profile_id, scale_pct)

    def remove_variant(self, effect_id: str, relative_path: str) -> None:
        """Unregister a variant; retain content for settings backup recovery."""
        document = self.settings_store.load()
        profile = self._profile(document, None)
        definition = next((item for item in profile["tracked_effects"]
                           if item["effect_id"] == effect_id), None)
        if definition is None or relative_path not in definition.get("template_variants", []):
            raise ValueError("등록된 변형을 찾을 수 없습니다.")
        definition["template_variants"].remove(relative_path)
        definition.get("template_variant_scales", {}).pop(relative_path, None)
        self.settings_store.save(document)

    def runtime_templates(self) -> tuple:
        document = self.settings_store.load()
        specs = []
        root = self.templates_root.resolve()
        for definition in self._profile(document, None)["tracked_effects"]:
            if definition.get("enabled", True) is not True:
                continue
            for relative in definition.get("template_variants", []):
                path = (self.settings_store.root / relative).resolve()
                if not path.is_relative_to(root):
                    raise ValueError("템플릿 경로가 사용자 이미지 폴더 밖에 있습니다.")
                specs.append((definition["effect_id"], path,
                              definition.get("template_variant_scales", {}).get(relative, 100)))
        return tuple(specs)

    def register_crop(self, effect_id: str, frame: np.ndarray, bbox: list[int] | tuple[int, ...],
                      *, profile_id: str | None = None,
                      scale_pct: int = 100) -> RegisteredTemplate:
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("frame must be a BGR image")
        if len(bbox) != 4 or any(type(value) is not int for value in bbox):
            raise ValueError("bbox must contain four integers")
        x, y, width, height = bbox
        if (min(x, y) < 0 or min(width, height) <= 0 or
                x + width > frame.shape[1] or y + height > frame.shape[0]):
            raise ValueError("bbox lies outside frame")
        crop = np.ascontiguousarray(frame[y:y+height, x:x+width])
        success, encoded = cv2.imencode(".png", crop)
        if not success:
            raise ValueError("cannot encode template crop")
        return self._register(effect_id, encoded.tobytes(), ".png", profile_id, scale_pct)

    def register_detection_crop(self, effect_id: str, frame: np.ndarray,
                                detection: Mapping[str, Any],
                                *, profile_id: str | None = None,
                                scale_pct: int = 100) -> RegisteredTemplate:
        if detection.get("effect_id") not in (None, effect_id):
            raise ValueError("detection effect_id does not match registration effect_id")
        return self.register_crop(
            effect_id,
            frame,
            detection.get("bbox", ()),
            profile_id=profile_id,
            scale_pct=scale_pct,
        )

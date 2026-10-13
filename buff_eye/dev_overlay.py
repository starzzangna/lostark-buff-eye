"""Transactional, display-only file reloads; never writes user settings."""
import json
from pathlib import Path
import cv2

from .core import image_read, safe_path
from .overlay_model import IconPlacement, OverlayLayout


class DevOverlay:
    def __init__(self, path, detector_config):
        self.path = Path(path)
        self.detector_config = Path(detector_config)
        self.icons = {}
        self.images = {}
        self.error = None
        self._signature = None
        self._watched = [self.path, self.detector_config]

    def poll(self):
        def stamp(path):
            try:
                stat = path.stat()
                return (str(path), stat.st_mtime_ns, stat.st_size)
            except OSError:
                return (str(path), None)
        signature = tuple(stamp(p) for p in self._watched)
        if signature == self._signature:
            return False
        try:
            data = json.loads(self.path.read_text(encoding='utf-8-sig'))
            if set(data) - {'enabled', 'icons', 'images'}:
                raise ValueError('Unknown development overlay field')
            enabled = data.get('enabled', True)
            if not isinstance(enabled, bool):
                raise ValueError('enabled must be boolean')
            icons = data.get('icons', {}) if enabled else {}
            for effect, fields in icons.items():
                if not effect or set(fields) - {'x', 'y', 'size', 'transparency_pct'}:
                    raise ValueError('Invalid icon placement fields')
                IconPlacement(effect, **fields)
            paths = {}
            if enabled:
                config = json.loads(self.detector_config.read_text(encoding='utf-8-sig'))
                paths = {effect: safe_path(self.detector_config.parent, relative)
                         for effect, relative in config.get('overlay_icons', {}).items()}
                paths.update({effect: safe_path(self.path.parent, relative)
                              for effect, relative in data.get('images', {}).items()})
            images = {effect: image_read(path) for effect, path in paths.items()}
            watched = [self.path, self.detector_config, *paths.values()]
        except (OSError, ValueError, TypeError, KeyError, AttributeError, cv2.error) as exc:
            self.error = str(exc)
            # Retry partial writes and missing/replaced images on the next tick.
            return False
        self.icons, self.images = icons, images
        self._watched = watched
        self._signature = tuple(stamp(p) for p in watched)
        self.error = None
        return True

    def layout(self, base):
        result = OverlayLayout.from_dict(base.to_dict())
        for effect, fields in self.icons.items():
            if effect in result.icons:
                values = result.icons[effect].to_dict()
                values.update(fields)
                result.icons[effect] = IconPlacement.from_dict(values)
        return result

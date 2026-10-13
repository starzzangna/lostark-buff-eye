"""Desktop application entry point."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence

from .capture_session import ManagedCaptureSession
from .core import read_json
from .support_buffs import SupportDetector as Detector
from .effect_groups import SUPPORT_EFFECTS as MERGED_COMMON, enabled_green
from .target_buffs import TARGET_SLOTS, TargetDetector
from .multi_live import MultiLiveWorker
from .live import LiveRecognitionWorker
from .overlay_model import Rect
from .ui import MonitorChoice, SettingsController, UiDependencyError, create_qt_application
from .win_capture import WindowsScreenCaptureProvider
from .game_viewport import locate as locate_game_viewport


DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "datasets" / "calibration-3440" / "detector-ocr-v6.json"


def load_user_templates(detector: Detector, controller: SettingsController) -> int:
    """Attach persistent user variants to the live detector at startup."""
    root = controller.store.root.resolve()
    templates_root = (root / "templates").resolve()
    loaded = 0
    for definition in controller.profile.get("tracked_effects", []):
        if definition.get("enabled", True) is not True:
            continue
        effect_id = definition.get("effect_id")
        scales = definition.get("template_variant_scales", {})
        for relative in definition.get("template_variants", []):
            path = (root / relative).resolve()
            if not path.is_relative_to(templates_root) or not path.is_file():
                continue
            scale = scales.get(relative, 100)
            loaded += int(detector.add_user_template(effect_id, path, scale))
    return loaded


def build_live_worker(
    controller: SettingsController,
    config_path: Path,
    *,
    provider: WindowsScreenCaptureProvider | None = None,
    target_fps: float = 20.0,
) -> LiveRecognitionWorker:
    """Build one self-buff ROI capture source from persistent settings."""
    provider = provider or WindowsScreenCaptureProvider()
    monitors = tuple(provider.enumerate_monitors())
    controller.set_monitors(
        MonitorChoice(
            monitor.stable_id,
            f"{monitor.device_name} ({monitor.bounds.width}×{monitor.bounds.height})",
            monitor.bounds.left,
            monitor.bounds.top,
            monitor.bounds.width,
            monitor.bounds.height,
            monitor.primary,
        )
        for monitor in monitors
    )
    selected_id = controller.document["display"].get("capture_monitor_id")
    monitor = next((item for item in monitors if item.stable_id == selected_id), None)
    if monitor is None:
        from .game_viewport import game_client_rect
        client = game_client_rect() if isinstance(provider, WindowsScreenCaptureProvider) else None
        if client is not None:
            x,y,w,h=client
            monitor=next((item for item in monitors if
                item.bounds.left<=x and item.bounds.top<=y and
                x+w<=item.bounds.right and y+h<=item.bounds.bottom),None)
        monitor = monitor or next((item for item in monitors if item.primary), monitors[0])
        controller.select_monitor("capture", monitor.stable_id)
    controller.select_monitor("overlay", monitor.stable_id)

    detector = Detector(config_path)
    detector.green_enabled_provider=lambda:enabled_green(controller.profile)
    load_user_templates(detector, controller)
    controller.profile['overlay_scope'] = 'support_common'
    layout = controller.overlay
    if 'support.attack.additional' in layout.icons and 'support.attack.01' in layout.icons:
        # The former fourth slot becomes the single attack slot; keep red styling.
        layout.icons['support.attack.01'].x = layout.icons['support.attack.additional'].x
        layout.icons['support.attack.01'].y = layout.icons['support.attack.additional'].y
    layout.icons = {key:value for key,value in layout.icons.items() if key in MERGED_COMMON + TARGET_SLOTS}
    controller.profile['overlay'] = layout.to_dict()
    for effect_id in MERGED_COMMON:
        controller.ensure_icon(effect_id)
    config_x, config_y, config_width, config_height = detector.config["region"]["bbox"]
    # Missing ROI means automatic HUD placement, not fixed calibration pixels.
    # ManagedCaptureSession re-resolves this document on every poll, so changing
    # monitor or ROI in the settings UI reconnects without restarting the app.
    session = ManagedCaptureSession(
        provider,
        lambda: controller.document,
        "self_buff",
        auto_cache_directory=controller.store.root/'auto-roi',
        overlay_capture_excluded=lambda: None,
        viewport_provider=(lambda monitor: locate_game_viewport(provider, monitor))
            if isinstance(provider,WindowsScreenCaptureProvider) else None,
    )
    scale = controller.profile["game_ui"]["buff_scale_pct"]
    controller.save()
    return LiveRecognitionWorker(
        session,
        detector,
        scale,
        target_fps=target_fps,
        scale_provider=lambda: controller.profile["game_ui"]["buff_scale_pct"],
        auto_scale=False,
        recognition_size=(config_width, config_height),
        native_roi=True,
    )


def build_target_worker(controller, config_path, target_fps=10):
    config=read_json(config_path)
    detector=TargetDetector(config_path.parent/'skills', config_path.parent/config['text']['bank_path'])
    # Boss absence does not block initial proportional placement.
    controller.place_target_defaults()
    controller.save()
    # No guessed ROI: the existing target region editor supplies this binding.
    provider=WindowsScreenCaptureProvider()
    session=ManagedCaptureSession(provider,lambda:controller.document,'target_debuff',
        auto_cache_directory=controller.store.root/'auto-roi',
        overlay_capture_excluded=lambda: None,
        viewport_provider=lambda monitor:locate_game_viewport(provider,monitor))
    return LiveRecognitionWorker(session,detector,100,target_fps=target_fps,
        native_roi=True,region_id='target_debuff')


def main(argv: Sequence[str] | None = None) -> int:
    raw_argv = list(sys.argv if argv is None else argv)
    parser = argparse.ArgumentParser(description="LostArk Buff Eye desktop overlay")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--no-live", action="store_true", help="open settings without capture")
    parser.add_argument("--dev-overlay", type=Path, help="watch a display-only development JSON file")
    args = parser.parse_args(raw_argv[1:])
    try:
        controller = SettingsController()
        worker = None if args.no_live else build_live_worker(controller, args.config, target_fps=args.fps)
        if worker is not None:
            worker = MultiLiveWorker(worker, build_target_worker(controller,args.config))
        desktop = create_qt_application(controller, raw_argv, live_worker=worker,
            default_preview_roi=Rect(*read_json(args.config)['region']['bbox']))
        if args.dev_overlay:
            from .dev_overlay import DevOverlay
            desktop.settings_window.enable_dev_overlay(DevOverlay(args.dev_overlay, args.config))
    except (UiDependencyError, OSError, RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return desktop.exec()


if __name__ == "__main__":
    raise SystemExit(main())

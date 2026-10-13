"""Support adapter for the shared native-pixel icon search."""
import time
from .core import Detector, iou
from .effect_groups import MERGED_COMMON, GREEN_EFFECTS
from .icon_search import IconSearch
from .reflection import ReflectionBridge
from .support_marker import AttackPairResolver
from .temporal_icon import TemporalIconEvidence


class SupportDetector(Detector):
    def __init__(self, config_path):
        super().__init__(config_path)
        assets = {key: value for key, value in self.overlay_icons.items()
                  if key in MERGED_COMMON}
        self.icon_search = IconSearch(assets, 'self_buff', max_instances=17)
        self.green_enabled_provider = lambda: ()
        self._green_enabled = ()
        self.green_search = None

    def _sync_green(self):
        enabled=tuple(k for k in GREEN_EFFECTS if k in self.green_enabled_provider())
        if enabled!=self._green_enabled:
            self._green_enabled=enabled
            # Green assets can fit a pixel smaller than the calibrated yellow
            # border box. Check adjacent sizes instead of forcing its exact size.
            self.green_search=IconSearch({k:self.overlay_icons[k] for k in enabled},'self_buff',max_instances=17,size_tolerance=1) if enabled else None
            self.reflection_bridge.reset()
            self.temporal_icons.reset()

    def reset_row_tracking(self):
        super().reset_row_tracking()
        self._native_scale = None
        self._native_probe_at = 0
        self.reflection_bridge = ReflectionBridge()
        self.attack_pairs = AttackPairResolver()
        self.temporal_icons = TemporalIconEvidence()
        if hasattr(self, 'icon_search'):
            self.icon_search.reset()

    def detect_region(self, crop, scale=100, search_mode='slots_tracked', absolute_coordinates=False):
        self._sync_green()
        sizes = self.config['scale_transfer']['native_sizes']
        now = time.monotonic()
        calibrated = []
        probe = self._native_scale is None or now >= self._native_probe_at
        if not probe:
            calibrated = super().detect_region(crop, self._native_scale, 'slots_tracked')
            probe = not calibrated or max(d['icon_score'] for d in calibrated) < .94
        if probe:
            ranked = []
            h,w = crop.shape[:2]
            for key in sizes:
                matches = self.detect(crop,int(key),search_mode='native',region_bbox=[0,0,w,h],read_text=False)
                scores = sorted((d['icon_score'] for d in matches), reverse=True)[:3]
                if scores:
                    ranked.append((sum(scores)/len(scores),int(key)))
            if ranked and max(ranked)[0] >= .90:
                self._native_scale = max(ranked)[1]
                calibrated = super().detect_region(crop,self._native_scale,'slots_tracked')
                self._native_probe_at = now+10
                # A single verified native icon suffices. No row-count gate.
                self.icon_search.size = sizes[str(self._native_scale)]
                self.icon_search.shape = crop.shape
                self.icon_search.probe_at = now+10
            else:
                self._native_scale = None
                calibrated = []
        hint = sizes[str(self._native_scale)] if calibrated else None
        candidates = self.icon_search.detect(crop, size_hint=hint)
        if self.green_search is not None:
            candidates += self.green_search.detect(crop,size_hint=hint)
        # The old calibrated extra icon is the same asset now displayed as
        # Major Code. Normalize before overlap suppression and presentation.
        result = [dict(d, effect_id='support.major_code')
                  if d['effect_id']=='support.extra.01' else d for d in calibrated]
        for candidate in candidates:
            if any(iou(candidate['bbox'], d['bbox']) > .3 for d in result):
                continue
            size = candidate['bbox'][2]
            key = min(sizes, key=lambda k: abs(sizes[k]-size))
            candidate['timer_layout'] = [max(2, round(self.config['text']['line_layout'][key][0]*size/sizes[key])),
                                         max(8, round(self.config['text']['line_layout'][key][1]*size/sizes[key]))]
            result.extend(self._finalize(crop, int(key), [candidate]))
        for candidate in self.reflection_bridge.update(crop,result,now):
            size=candidate['bbox'][2]
            key=min(sizes,key=lambda k:abs(sizes[k]-size))
            candidate['timer_layout']=[max(2,round(self.config['text']['line_layout'][key][0]*size/sizes[key])),
                                       max(8,round(self.config['text']['line_layout'][key][1]*size/sizes[key]))]
            result.extend(self._finalize(crop,int(key),[candidate]))
        for candidate in self.temporal_icons.detect(crop,result,now,self.icon_search,sizes.values()):
            size=candidate['bbox'][2]
            key=min(sizes,key=lambda k:abs(sizes[k]-size))
            candidate['timer_layout']=[max(2,round(self.config['text']['line_layout'][key][0]*size/sizes[key])),
                                       max(8,round(self.config['text']['line_layout'][key][1]*size/sizes[key]))]
            result.extend(self._finalize(crop,int(key),[candidate]))
        # Individual fallback candidates were finalized separately. Pair them
        # only after calibrated, asset and reflection results are combined.
        return [d for d in self.attack_pairs.resolve(result, now)
                if d['effect_id'] not in GREEN_EFFECTS or d['effect_id'] in self._green_enabled]

"""Target status icons: asset matching and a two-slot presentation policy."""
from pathlib import Path
import time
import cv2
import numpy as np
from .core import image_read, iou
from .text import TimeReader
from .target_text_shape import TargetShapeReader
from .target_matching import ColorSearchFrame
from .presentation import OverlayItem, _display_timer
from .overlay_model import OverlayLayout

TARGET_SLOTS = ('target.support', 'target.dark')
TARGET_LABELS = ('낙인', '암수 / 빛암수')
ASSETS = {
    'target.dark': 'battle_item_01_47.png',
    'target.light_dark': 'battle_item_01_87.png',
    'target.bard': 'buff_60.png',
    'target.holy': 'buff_67.png',
    'target.artist': 'buff_498.png',
}


class TargetDetector:
    def __init__(self, root, bank):
        self.overlay_icons = {key: image_read(Path(root)/name) for key,name in ASSETS.items()}
        from .icon_search import IconSearch
        self.icon_search = IconSearch(self.overlay_icons, 'target_debuff')
        native_root=Path(bank).parent/'target-native'
        self.native_search=IconSearch({key:image_read(native_root/(name+'.png')) for key,name in
            (('target.dark','dark'),('target.light_dark','light_dark'))},'target_debuff')
        self.templates = self.icon_search.templates
        target_bank = Path(bank).with_name('target-glyphs.json')
        self.time_reader = TimeReader(bank, extra_bank_path=target_bank if target_bank.exists() else None)
        self.shape_reader = TargetShapeReader(target_bank,small_geometry=True) if target_bank.exists() else None
        self.config = {'region': {'region_id':'target_debuff'}}
        self.reset_row_tracking()

    def reset_row_tracking(self):
        self.icon_search.reset()
        self.native_search.reset()
        self.size = None
        self.probe_at = 0
        self.last_row_search = 'target-assets'

    def set_cancel_check(self,check):
        self.icon_search.cancel_check=check
        self.native_search.cancel_check=check

    def detect_region(self, frame, scale=100, search_mode=None):
        now=time.monotonic()
        candidates = self.icon_search.detect(frame,now=now)
        candidates += self.native_search.detect(frame,now=now)
        accepted = []
        for d in sorted(candidates,key=lambda d:d['icon_score'],reverse=True):
            if self.icon_search.cancel_check():return []
            if any(iou(d['bbox'], other['bbox']) > .3 for other in accepted):
                continue
            if any(other['effect_id'] != d['effect_id'] and iou(d['bbox'],other['bbox']) > .3
                   and d['icon_score']-other['icon_score'] < .03 for other in candidates):
                continue
            x,y,w,h = d['bbox']
            if x<0 or y<0 or x+w>frame.shape[1] or y+h>frame.shape[0]:
                continue
            text_box = [x,y+h+(0 if w<=26 else 3),w,max(10,round(w*.32))]
            if text_box[1]+text_box[3] <= frame.shape[0]:
                d['time'] = self.time_reader.read_region(frame,text_box,text_color='target')
                if d['time']['status'] != 'read':
                    # Reflections desaturate orange text. A bounded luminance
                    # fallback keeps the same glyph confidence and grammar.
                    # Require all accepted alignments to agree; never choose
                    # a value merely because one preprocessing pass found it.
                    reads = []
                    for offset in ((0,1,2,3,4,5) if w<=26 else (3,4,5)):
                        yy = y+h+offset
                        th = max(10, round(w*.32))
                        if yy+th > frame.shape[0]:
                            continue
                        result = self.time_reader.read(frame[yy:yy+th,x:x+w], 'target_luminance')
                        result['bbox'] = [x,yy,w,th]
                        if result['status'] == 'read':
                            reads.append(result)
                    if reads and len({r['raw_text'] for r in reads}) == 1:
                        d['time'] = max(reads, key=lambda r:r['score'])
                if d['time']['status'] != 'read' and self.shape_reader is not None:
                    result=self.shape_reader.read(frame,d['bbox'])
                    if result['status']=='read':
                        d['time']=result
                if d['time']['status']!='read':
                    result=self.time_reader.read_small_region(frame,text_box,'target')
                    if result is not None:d['time']=result
                if d['time']['status']!='read':
                    result=self.time_reader.read_tiny_target_chroma(frame,d['bbox'])
                    if result is not None:d['time']=result
                if d['time']['status']!='read' and w>=27:
                    # Native game timers may start 6-7px below the icon. The
                    # short default box then loses lower strokes and decimals.
                    # Two bounded complete crops must agree; retain OCR gates.
                    top=y+h
                    height=min(26,max(18,round(w*.6)))
                    reads=[]
                    for extra in (0,2):
                        end=top+height+extra
                        if end>frame.shape[0]:break
                        read=self.time_reader.read(frame[top:end,x:x+w],'target')
                        if read['status']!='read':break
                        read['bbox']=[x,top,w,height+extra]
                        reads.append(read)
                    if len(reads)==2 and reads[0]['raw_text']==reads[1]['raw_text']:
                        d['time']=min(reads,key=lambda r:r['score'])
                    if d['time']['status']!='read':
                        # Background just below an intact line can join the
                        # glyphs. Require two shorter crops to agree, with all
                        # glyph bottoms still at least two pixels from the cut.
                        reads=[]
                        for th in (height-2,height-1):
                            if top+th>frame.shape[0]:break
                            read=self.time_reader.read(frame[top:top+th,x:x+w],'target')
                            chars=read.get('characters',[])
                            if (read['status']!='read' or not chars or
                                any(c['bbox'][1]+c['bbox'][3]>th-2 for c in chars)):
                                break
                            read['bbox']=[x,top,w,th]
                            reads.append(read)
                        if len(reads)==2 and reads[0]['raw_text']==reads[1]['raw_text']:
                            d['time']=min(reads,key=lambda r:r['score'])
                            d['time']['method']='target_complete_line_consensus'
            else:
                d['time'] = {'status':'unreadable','raw_text':None}
            accepted.append(d)
        if accepted:
            self.size = max(accepted,key=lambda d:d['icon_score'])['bbox'][2]
            self.probe_at = self.icon_search.probe_at
        return accepted


def target_items(detections, profile):
    selected = {}
    for d in detections:
        key=d.get('effect_id')
        if d.get('region_id')=='target_debuff' and key in ASSETS:
            if key not in selected or d.get('icon_score',0)>selected[key].get('icon_score',0):
                selected[key]=d
    layout=OverlayLayout.from_dict(profile.get('overlay',{}))
    result=[]
    dark=[selected[k] for k in ('target.light_dark','target.dark') if k in selected]
    if dark:
        # A replacement icon may coexist with the old display-hold track.
        # Show the current observation, never combine inferred hidden timers.
        current=[d for d in dark if d.get('temporal',{}).get('observed',True)
                 and d.get('time',{}).get('reason')!='display_hold']
        dark=[max(current or dark,key=lambda d:(d.get('temporal',{}).get('last_seen_ns',0),
                                                d['effect_id']=='target.light_dark',
                                                d.get('icon_score',0)))]
    support=[selected[k] for k in ('target.bard','target.holy','target.artist') if k in selected]
    if support:
        support=[max(support,key=lambda d:d.get('icon_score',0))]
    for slot, group in (('target.support', support), ('target.dark', dark)):
        if not group or slot not in layout.icons:
            continue
        d=group[0]; placement=layout.icons[slot]
        text=_display_timer(d)[0] or '?'
        result.append(OverlayItem(slot,'target_debuff',placement.x,placement.y,placement.size,
            1-placement.transparency_pct/100,d.get('icon_score'),tuple(d['bbox']),None,
            text,None,'display',None,None,'hidden',d['effect_id']))
    return tuple(result)

"""Shared image detector and strict dataset validation."""
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def image_read(path):
    im = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if im is None:
        raise ValueError(f'Cannot decode image: {path}')
    return im

def safe_path(root, relative):
    root = Path(root).resolve()
    p = (root / relative).resolve()
    if not p.is_relative_to(root):
        raise ValueError(f'Path escapes dataset: {relative}')
    return p

def box_valid(box, width, height):
    return (isinstance(box, list) and len(box) == 4 and
            all(type(v) is int for v in box) and min(box[:2]) >= 0 and
            min(box[2:]) > 0 and box[0] + box[2] <= width and box[1] + box[3] <= height)

def validate_manifest(path):
    path = Path(path)
    m = read_json(path)
    if m.get('schema_version') != 1 or not m.get('assets'):
        raise ValueError('Unsupported schema or empty manifest')
    ids, sessions, hashes = set(), {}, {}
    for a in m['assets']:
        required = {'asset_id','path','sha256','media_type','width','height','session_id','split',
                    'source_kind','buff_scale_pct','hud_scale_pct','target_scale_pct','stack_merge','tags'}
        if required - a.keys():
            raise ValueError(f'Missing asset fields: {required - a.keys()}')
        if a['asset_id'] in ids:
            raise ValueError('Duplicate asset ID')
        ids.add(a['asset_id'])
        if a['split'] not in {'calibration','train','dev','test'}:
            raise ValueError('Invalid split')
        for mapping, key in [(sessions,a['session_id']), (hashes,a['sha256'])]:
            if key in mapping and mapping[key] != a['split']:
                raise ValueError('Session or duplicate image leaks across splits')
            mapping[key] = a['split']
        p = safe_path(path.parent, a['path'])
        if digest(p) != a['sha256']:
            raise ValueError(f'Hash mismatch: {p}')
        if a['media_type'] != 'image':
            raise ValueError('This milestone supports still images only')
        im = image_read(p)
        if im.shape[:2] != (a['height'], a['width']):
            raise ValueError(f'Dimension mismatch: {p}')
    return m

def iou(a,b):
    x,y=max(a[0],b[0]),max(a[1],b[1])
    w=max(0,min(a[0]+a[2],b[0]+b[2])-x)
    h=max(0,min(a[1]+a[3],b[1]+b[3])-y)
    intersection=w*h
    return intersection/(a[2]*a[3]+b[2]*b[3]-intersection)

class Detector:
    def __init__(self, config_path):
        self.path = Path(config_path)
        self.config = read_json(config_path)
        self.time_reader = None
        self.time_shape_reader = None
        if self.config.get('text'):
            from .text import TimeReader
            self.time_reader = TimeReader(safe_path(self.path.parent,self.config['text']['bank_path']))
            shape_bank = self.config['text'].get('shape_bank_path')
            if shape_bank:
                from .target_text_shape import TargetShapeReader
                self.time_shape_reader = TargetShapeReader(safe_path(self.path.parent,shape_bank))
        self.stack_reader = None
        if self.config.get('stack'):
            from .stack import StackReader
            self.stack_reader = StackReader(safe_path(self.path.parent,self.config['stack']['bank_path']))
        self.templates=[]
        self.overlay_icons = {effect: image_read(safe_path(self.path.parent, path))
                              for effect, path in self.config.get('overlay_icons', {}).items()}
        self.merge_marker = (image_read(safe_path(self.path.parent, self.config['merge_marker_path']))
                             if self.config.get('merge_marker_path') else None)
        for t in self.config['templates']:
            im=image_read(safe_path(self.path.parent,t['path']))
            self.templates.append((t,im))
        # Live capture reuses the same scale for thousands of frames.  Keep
        # resized templates and their masks instead of rebuilding them on
        # every detect() call.
        self._prepared={}
        self._gradient_templates={}
        self._slot_proposer=None
        self.reset_row_tracking()

    def reset_row_tracking(self):
        self._row_tracking = None
        self.last_row_search = 'reset'

    def add_user_template(self, effect_id, path, scale_pct, *, threshold=0.90):
        """Load one persistent icon crop into the same detector used by live capture."""
        if not isinstance(effect_id, str) or not effect_id:
            raise ValueError('effect_id must be a non-empty string')
        if scale_pct not in (80,90,100,110,120):
            raise ValueError('unsupported user template scale')
        image=image_read(path)
        height,width=image.shape[:2]
        if max(width,height)/min(width,height)>1.35:
            raise ValueError('user template must be a near-square icon crop')
        native_sizes=self.config.get('scale_transfer',{}).get('native_sizes',{})
        native_size=native_sizes.get(str(scale_pct))
        if native_size is not None and (width,height)!=(native_size,native_size):
            image=cv2.resize(image,(native_size,native_size),interpolation=cv2.INTER_AREA if max(width,height)>native_size else cv2.INTER_LINEAR)
        fingerprint=digest(path)[:16]
        variant_id=f'user-{effect_id}-{scale_pct}-{fingerprint}'
        if any(t.get('variant_id')==variant_id for t,_image in self.templates):
            return False
        size=image.shape[0]
        inset=max(2,round(size*5/37))
        metadata=dict(
            effect_id=effect_id,
            variant_id=variant_id,
            path=str(Path(path)),
            buff_scale_pct=scale_pct,
            inset=inset,
            threshold=float(threshold),
            user_template=True,
        )
        self.templates.append((metadata,image))
        return True

    def replace_user_templates(self, specs):
        """Worker-thread transaction: preserve the previous set on decode failure."""
        previous = self.templates
        self.templates = [(t, im) for t, im in previous if not t.get('user_template')]
        try:
            for effect_id, path, scale in specs:
                self.add_user_template(effect_id, path, scale)
        except Exception:
            self.templates = previous
            raise
        self._prepared.clear()
        self._gradient_templates.clear()
        self.reset_row_tracking()

    def _prepare(self,t,template,scale):
        key=(t['variant_id'],scale)
        if key in self._prepared:
            return self._prepared[key]
        cross_scale=t['buff_scale_pct'] != scale
        threshold=t.get('threshold',0.94)
        inset=t.get('inset',4)
        prepared=template
        if cross_scale:
            transfer=self.config.get('scale_transfer')
            if not transfer or str(scale) not in transfer['native_sizes']:
                return None
            source_size=transfer['native_sizes'].get(str(t['buff_scale_pct']))
            if source_size is None:
                return None
            ratio=transfer['native_sizes'][str(scale)]/source_size
            prepared=cv2.resize(template,None,fx=ratio,fy=ratio,interpolation=cv2.INTER_LINEAR)
            inset=max(2,round(inset*ratio))
            threshold=transfer.get('threshold',0.88)
        th,tw=prepared.shape[:2]
        mask=np.zeros((th,tw),np.uint8)
        mask[inset:th-inset,inset:tw-inset]=255
        if not mask.any():
            raise ValueError('Empty template mask')
        value=(prepared,mask,threshold,cross_scale)
        self._prepared[key]=value
        return value

    @staticmethod
    def _gradient(image):
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        gx=cv2.Sobel(gray,cv2.CV_32F,1,0,ksize=3)
        gy=cv2.Sobel(gray,cv2.CV_32F,0,1,ksize=3)
        return cv2.magnitude(gx,gy)

    @staticmethod
    def _correlation_scores(roi, template, mask):
        """Rectangular inset masks are equivalent to cropping both inputs.

        Retain the full-template search domain so bbox coordinates and edge
        eligibility do not change. Other masks keep OpenCV's masked path.
        """
        ix,iy,iw,ih=cv2.boundingRect(mask)
        if iw == 0 or ih == 0 or cv2.countNonZero(mask) != iw*ih:
            return cv2.matchTemplate(roi,template,cv2.TM_CCOEFF_NORMED,mask=mask)
        th,tw=template.shape[:2]
        rh,rw=roi.shape[:2]
        interior=roi[iy:rh-th+iy+ih,ix:rw-tw+ix+iw]
        inner=template[iy:iy+ih,ix:ix+iw]
        if float(cv2.meanStdDev(inner)[1].max()) < 1e-4:
            return np.full((rh-th+1,rw-tw+1),-1,dtype=np.float32)
        return cv2.matchTemplate(interior,inner,cv2.TM_CCOEFF_NORMED)

    @staticmethod
    def _match_inset(local,template,mask):
        """Fast equivalent of a rectangular masked local template search."""
        ix,iy,iw,ih=cv2.boundingRect(mask)
        inner=template[iy:iy+ih,ix:ix+iw]
        if inner.shape[0]>local.shape[0] or inner.shape[1]>local.shape[1]:
            return -1.0,(0,0)
        scores=cv2.matchTemplate(local,inner,cv2.TM_CCOEFF_NORMED)
        scores=np.nan_to_num(scores,nan=-1,posinf=-1,neginf=-1)
        th,tw=template.shape[:2]
        x0,y0=ix,iy
        x1=local.shape[1]-tw+ix+1
        y1=local.shape[0]-th+iy+1
        if x1<=x0 or y1<=y0:
            return -1.0,(0,0)
        restricted=scores[y0:y1,x0:x1]
        _,score,_,loc=cv2.minMaxLoc(restricted)
        return float(score),(loc[0]+x0-ix,loc[1]+y0-iy)

    def _slot_candidates(self,frame,scale,region_bbox=None,calibrated_row=True,row_hint=None):
        from .slots import SlotProposer
        if self._slot_proposer is None:
            self._slot_proposer=SlotProposer()
        region=self.config['region']
        region_bbox=region['bbox'] if region_bbox is None else region_bbox
        row_y=row_hint
        if not calibrated_row and row_hint is None:
            anchors=self.detect(frame,scale,search_mode='native',region_bbox=region_bbox)
            if not anchors:
                return []
            row_y=round(float(np.median([d['bbox'][1] for d in anchors])))-region_bbox[1]
        self._last_anchor_row = row_y
        proposals=self._slot_proposer.propose(frame,region_bbox,scale,max_candidates=17,
                                               calibrated_row=calibrated_row,row_y=row_y)
        candidates=[]
        # Share conservative color bounds across the whole proposed row.
        # Final scores and positions still come from the original matcher.
        from .slot_matching import SlotColorGate
        gate=SlotColorGate(frame,proposals)
        for proposal in proposals:
            px,py,pw,ph=proposal.bbox
            pad=max(3,round(pw*0.14))
            left=max(0,px-pad); top=max(0,py-pad)
            right=min(frame.shape[1],px+pw+pad); bottom=min(frame.shape[0],py+ph+pad)
            local=frame[top:bottom,left:right]
            by_effect={}
            for t,template in self.templates:
                prepared=self._prepare(t,template,scale)
                if prepared is None:
                    continue
                template,mask,threshold,cross_scale=prepared
                th,tw=template.shape[:2]
                if th>local.shape[0] or tw>local.shape[1]:
                    continue
                if not gate.could_match(t['variant_id'],template,mask,threshold,left,top,right,bottom):
                    continue
                score,loc=self._match_inset(local,template,mask)
                if score < threshold:
                    continue
                current=by_effect.get(t['effect_id'])
                if current is None or score>current['icon_score']:
                    xx,yy=loc
                    by_effect[t['effect_id']]=dict(effect_id=t['effect_id'],variant_id=t['variant_id'],
                        region_id=region['region_id'],bbox=[left+xx,top+yy,tw,th],icon_score=float(score),
                        score_method='slot_color_correlation',template_scale=t['buff_scale_pct'],
                        proposal_score=proposal.score,time={'status':'not_implemented','raw_text':None},
                        stack={'status':'not_implemented','raw_text':None})
            if not by_effect and self.config.get('gradient_fallback_threshold') is not None:
                local_gradient=self._gradient(local)
                for t,template in self.templates:
                    if t['buff_scale_pct'] != scale:
                        continue
                    prepared=self._prepare(t,template,scale)
                    template,mask,_,_=prepared
                    gradient=self._gradient_templates.get(t['variant_id'])
                    if gradient is None:
                        gradient=self._gradient(template)
                        self._gradient_templates[t['variant_id']]=gradient
                    score,loc=self._match_inset(local_gradient,gradient,mask)
                    if score < self.config['gradient_fallback_threshold']:
                        continue
                    current=by_effect.get(t['effect_id'])
                    if current is None or score>current['icon_score']:
                        xx,yy=loc; th,tw=template.shape[:2]
                        by_effect[t['effect_id']]=dict(effect_id=t['effect_id'],variant_id=t['variant_id'],
                            region_id=region['region_id'],bbox=[left+xx,top+yy,tw,th],icon_score=float(score),
                            score_method='slot_gradient_fallback',template_scale=t['buff_scale_pct'],
                            proposal_score=proposal.score,time={'status':'not_implemented','raw_text':None},
                            stack={'status':'not_implemented','raw_text':None})
            ranked=sorted(by_effect.values(),key=lambda item:item['icon_score'],reverse=True)
            if ranked and (len(ranked)==1 or ranked[0]['icon_score']-ranked[1]['icon_score']>=self.config.get('min_margin',0.015)):
                candidates.append(ranked[0])
        return candidates

    def detect_region(self,crop,scale,search_mode='slots',absolute_coordinates=False):
        """Recognize an already-captured configured ROI without a full-screen copy."""
        if search_mode not in ('slots', 'slots_anywhere', 'slots_tracked'):
            raise ValueError('cropped ROI detection currently supports slots mode only')
        _,_,width,height=self.config['region']['bbox']
        if search_mode in ('slots_anywhere', 'slots_tracked'):
            if crop.ndim != 3 or crop.shape[2] != 3 or min(crop.shape[:2]) < 1:
                raise ValueError('ROI crop must be a nonempty BGR image')
            height,width=crop.shape[:2]
        if crop.ndim!=3 or crop.shape!=(height,width,3):
            raise ValueError(f'ROI crop must have shape {(height,width,3)}')
        key = (width, height, scale)
        cached = self._row_tracking if search_mode == 'slots_tracked' else None
        candidates = None
        if cached is not None and cached['key'] == key and cached['frames'] < 29:
            candidates = self._slot_candidates(crop, scale, [0,0,width,height],
                calibrated_row=False, row_hint=cached['y'])
            effects = {d['effect_id'] for d in candidates}
            # A moved or obscured row must recover in this same frame. Losing
            # even one previously visible effect triggers the full path.
            if candidates and cached['effects'].issubset(effects):
                cached['frames'] += 1
                cached['effects'] = effects
                self.last_row_search = 'cached'
            else:
                candidates = None
        if candidates is None:
            self._last_anchor_row = None
            candidates = self._slot_candidates(crop,scale,[0,0,width,height],
                                               calibrated_row=search_mode=='slots')
            self.last_row_search = 'full'
            if search_mode == 'slots_tracked':
                self._row_tracking = ({'key': key, 'y': self._last_anchor_row,
                    'frames': 0, 'effects': {d['effect_id'] for d in candidates}}
                    if candidates and self._last_anchor_row is not None else None)
        detections=self._finalize(crop,scale,candidates)
        if absolute_coordinates:
            offset_x,offset_y=self.config['region']['bbox'][:2]
            for detection in detections:
                detection['bbox'][0]+=offset_x; detection['bbox'][1]+=offset_y
                time=detection.get('time',{})
                if time.get('bbox'):
                    time['bbox'][0]+=offset_x; time['bbox'][1]+=offset_y
        return detections

    def _finalize(self,frame,scale,candidates,read_text=True):
        candidates.sort(key=lambda d:d['icon_score'],reverse=True)
        accepted=[]
        for d in candidates:
            if any(iou(d['bbox'],other['bbox'])>0.3 for other in accepted):
                continue
            competitors=[c for c in candidates if c['effect_id']!=d['effect_id'] and iou(c['bbox'],d['bbox'])>0.3]
            if competitors and d['icon_score']-max(c['icon_score'] for c in competitors)<self.config.get('min_margin',0.015):
                continue
            accepted.append(d)
        if self.time_reader is not None and read_text:
            layout=self.config['text']['line_layout'].get(str(scale))
            for d in accepted:
                if layout is None:
                    d['time']={'status':'unreadable','raw_text':None,'reason':'uncalibrated_scale'}
                    continue
                dx,dy,dw,dh=d['bbox']
                top,height=d.get('timer_layout', layout)
                tb=[dx,dy+dh+top,dw,height]
                # The reader already checks a one-pixel vertical alignment.
                # Do not reject an edge ROI before that bounded check can run.
                if not any(box_valid([dx,tb[1]+offset,dw,height],frame.shape[1],frame.shape[0])
                           for offset in (0,-1,1)):
                    d['time']={'status':'unreadable','raw_text':None,'reason':'crop_outside_frame'}
                    continue
                from .border import border_color
                border = border_color(frame,d['bbox'])
                if border and border[1] > border[0]*1.1 and border[1] > border[2]*1.1:
                    d['time']=self.time_reader.read_region(frame,tb,text_color='green')
                    d['time']['text_color']='green'
                else:
                    d['time']=self.time_reader.read_region(frame,tb)
                    if d['time']['status'] != 'read' and self.config['text'].get('fallback_desaturated'):
                        fallback=self.time_reader.read_region(frame,tb,text_color='green')
                        if fallback['status']=='read':
                            fallback['text_color']='green'
                            d['time']=fallback
                if d['time']['status'] != 'read' and self.time_shape_reader is not None:
                    fallback=self.time_shape_reader.read(frame,d['bbox'])
                    if fallback['status']=='read':
                        fallback['method']='support_ink_shadow'
                        d['time']=fallback
                if d['time']['status'] != 'read':
                    fallback=self.time_reader.read_small_region(frame,tb,d['time'].get('text_color','yellow'))
                    if fallback is not None:d['time']=fallback
        if self.stack_reader is not None and read_text:
            for d in accepted:
                d['stack']=self.stack_reader.read_for_icon(frame,d['bbox'],scale)
        from .border import border_color
        for d in accepted:
            d['border_color'] = border_color(frame, d['bbox'])
        if getattr(self, 'merge_marker', None) is not None:
            from .support_marker import read_marker
            for d in accepted:
                if d['effect_id'] == 'support.attack.01':
                    d['merged_emphasis'] = read_marker(frame, d['bbox'], self.merge_marker)
            red = [d for d in accepted if d['effect_id'] == 'support.attack.01']
            if len(red) == 2 and {d.get('merged_emphasis') for d in red} == {True, False}:
                for d in red:
                    if d['merged_emphasis'] is False:
                        # Observed paired slot, not an inferred class/skill identity.
                        d['effect_id'] = 'support.attack.additional'
        return accepted

    def detect(self, frame, scale, search_mode='exhaustive', region_bbox=None, read_text=True):
        if search_mode not in ('exhaustive','adjacent','native','slots'):
            raise ValueError(f'Unknown search mode: {search_mode}')
        if search_mode == 'slots':
            return self._finalize(frame,scale,self._slot_candidates(frame,scale),read_text=read_text)
        region=self.config['region']
        box=region['bbox'] if region_bbox is None else region_bbox
        if not box_valid(box,frame.shape[1],frame.shape[0]):
            raise ValueError('ROI outside image')
        x,y,w,h=box
        roi=frame[y:y+h,x:x+w]
        from .target_matching import ColorSearchFrame
        color_search=ColorSearchFrame(roi)
        candidates=[]
        ordered=sorted(self.templates,key=lambda entry:(entry[0]['buff_scale_pct'] != scale,
            abs(entry[0]['buff_scale_pct']-scale)))
        found_effects=set()
        for t,template in ordered:
            prepared=self._prepare(t,template,scale)
            if prepared is None:
                continue
            template,mask,threshold,cross_scale=prepared
            if cross_scale and search_mode == 'native':
                continue
            if cross_scale and search_mode == 'adjacent' and abs(t['buff_scale_pct']-scale)>10:
                continue
            if cross_scale and t['effect_id'] in found_effects:
                continue
            th,tw=template.shape[:2]
            if th>h or tw>w:
                continue
            # Mean-centered correlation rejects similarly colored flat scenery.
            ix,iy,iw,ih=cv2.boundingRect(mask)
            if (iw and ih and cv2.countNonZero(mask)==iw*ih and
                    not color_search.could_match(template[iy:iy+ih,ix:ix+iw],threshold-.001)):
                continue
            scores=self._correlation_scores(roi,template,mask)
            scores=np.nan_to_num(scores,nan=-1,posinf=-1,neginf=-1)
            for _ in range(17):
                _,score,_,loc=cv2.minMaxLoc(scores)
                if score<threshold:
                    break
                xx,yy=loc
                candidates.append(dict(effect_id=t['effect_id'],variant_id=t['variant_id'],
                    region_id=region['region_id'],bbox=[x+xx,y+yy,tw,th],icon_score=float(score),score_method='color_correlation',template_scale=t['buff_scale_pct'],
                    time={'status':'not_implemented','raw_text':None},stack={'status':'not_implemented','raw_text':None}))
                found_effects.add(t['effect_id'])
                scores[max(0,yy-th//2):yy+th//2+1,max(0,xx-tw//2):xx+tw//2+1]=-1
        # Expiring icons can be washed out by the game's animated white sheen.
        # A native-size edge fallback is opt-in and only runs for effects that
        # raw color correlation did not find.  It is deliberately not applied
        # across scales because resampling makes the edge score unstable.
        gradient_threshold=self.config.get('gradient_fallback_threshold')
        if gradient_threshold is not None:
            roi_gradient=self._gradient(roi)
            for t,template in self.templates:
                if t['buff_scale_pct'] != scale or t['effect_id'] in found_effects:
                    continue
                prepared=self._prepare(t,template,scale)
                template,mask,_,_=prepared
                th,tw=template.shape[:2]
                if th>h or tw>w:
                    continue
                gradient=self._gradient_templates.get(t['variant_id'])
                if gradient is None:
                    gradient=self._gradient(template)
                    self._gradient_templates[t['variant_id']]=gradient
                scores=self._correlation_scores(roi_gradient,gradient,mask)
                scores=np.nan_to_num(scores,nan=-1,posinf=-1,neginf=-1)
                _,score,_,loc=cv2.minMaxLoc(scores)
                if score < gradient_threshold:
                    continue
                xx,yy=loc
                candidates.append(dict(effect_id=t['effect_id'],variant_id=t['variant_id'],
                    region_id=region['region_id'],bbox=[x+xx,y+yy,tw,th],icon_score=float(score),
                    score_method='gradient_fallback',template_scale=t['buff_scale_pct'],
                    time={'status':'not_implemented','raw_text':None},stack={'status':'not_implemented','raw_text':None}))
                found_effects.add(t['effect_id'])
        return self._finalize(frame,scale,candidates,read_text=read_text)

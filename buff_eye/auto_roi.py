"""First-run HUD placement and conservative row confirmation, no full-screen scan."""
from .game_viewport import transform_roi

REFERENCE=(0,0,3440,1440)
SEARCH_REGIONS={'self_buff':(760,1150,860,145), 'target_debuff':(1580,170,780,135)}


def automatic(stored):
    return stored.pixel_rect is None and stored.normalized_rect is None and stored.region_id in SEARCH_REGIONS


def initial_region(viewport, region_id):
    x,y,w,h=transform_roi(SEARCH_REGIONS[region_id],REFERENCE,viewport,region_id)
    vx,vy,vw,vh=viewport
    # HUD anchoring is not perfectly proportional at small window sizes.
    # Keep a physical-pixel margin for the icon, stack and complete timer.
    margin=max(48,round(vh*.045))
    y-=margin
    h+=2*margin
    right,bottom=min(x+w,vx+vw),min(y+h,vy+vh)
    x,y=max(vx,x),max(vy,y)
    return x,y,max(1,right-x),max(1,bottom-y)


class AutoRow:
    def __init__(self, cache_directory=None):
        self.cache_directory=cache_directory
        self.reset()

    def reset(self):
        self.context=None; self.refined=None; self.candidate=None
        self.hits=0; self.seen=None

    def region(self, context, initial, now):
        if context!=self.context:
            self.reset(); self.context=context
            if self.cache_directory is not None:
                import hashlib,json
                from pathlib import Path
                key=hashlib.sha256(repr(context).encode()).hexdigest()
                self.cache_path=Path(self.cache_directory)/(key+'.json')
                try:
                    value=json.loads(self.cache_path.read_text(encoding='utf-8'))
                    rect=value['rect']
                    if (value.get('initial')==list(initial) and isinstance(rect,list) and len(rect)==4
                        and all(type(n) is int for n in rect) and rect[2]>0 and rect[3]>0
                        and rect[0]>=initial[0] and rect[1]>=initial[1]
                        and rect[0]+rect[2]<=initial[0]+initial[2]
                        and rect[1]+rect[3]<=initial[1]+initial[3]):self.refined=tuple(rect)
                except (OSError,ValueError,KeyError,TypeError):pass
        # Empty observations also mean loading, no active buff, or no boss.
        # They are not evidence that the confirmed HUD row moved.
        return self.refined or initial

    def observe(self, detections, capture, initial, now):
        before=self.refined
        self._observe(detections,capture,initial,now)
        if before!=self.refined:self._persist(initial)

    def _persist(self, initial):
        if self.cache_directory is None or not hasattr(self,'cache_path'):return
        import json
        try:
            self.cache_path.parent.mkdir(parents=True,exist_ok=True)
            temp=self.cache_path.with_suffix('.tmp')
            temp.write_text(json.dumps({'initial':list(initial),'rect':self.refined}),encoding='utf-8')
            temp.replace(self.cache_path)
        except OSError:pass

    def forget(self):
        if self.cache_directory is not None and hasattr(self,'cache_path'):
            try:self.cache_path.unlink(missing_ok=True)
            except OSError:pass
        self.reset()

    def _observe(self, detections, capture, initial, now):
        boxes=[d['bbox'] for d in detections if d.get('icon_score',0)>=.90
               and d.get('score_method')!='recent_icon_unoccluded_pixels']
        if not boxes:
            self.candidate=None;self.hits=0
            return
        # Never tighten around competing rows or out-of-frame coordinates.
        if any(len(b)!=4 or min(b[0],b[1])<0 or min(b[2:])<=0 or
               b[0]+b[2]>capture[2] or b[1]+b[3]>capture[3] for b in boxes):return
        ys=[capture[1]+b[1] for b in boxes]; size=max(b[3] for b in boxes)
        if max(ys)-min(ys)>max(3,size*.15):
            self.candidate=None;self.hits=0
            return
        # A visible icon alone does not prove its timer is inside the capture.
        needed_top=min(ys)-max(12,round(size*.8))
        needed_bottom=max(ys)+size+max(24,round(size*.8))
        if needed_top<capture[1] or needed_bottom>capture[1]+capture[3]:
            # Correlation boxes vary by a few pixels with highlights. Grow only
            # the missing margin instead of discarding the confirmed HUD row.
            overflow=max(capture[1]-needed_top,needed_bottom-capture[1]-capture[3])
            if self.refined is not None and overflow<=4:
                top=max(initial[1],min(capture[1],needed_top-4))
                bottom=min(initial[1]+initial[3],max(capture[1]+capture[3],needed_bottom+4))
                self.refined=(initial[0],top,initial[2],bottom-top)
                return
            self.refined=None;self.candidate=None;self.hits=0
            return
        row=(round(sum(ys)/len(ys)),size)
        self.seen=now
        if self.refined is not None:return
        if self.candidate and abs(row[0]-self.candidate[0])<=2 and abs(row[1]-self.candidate[1])<=2:
            self.hits+=1
        else:self.candidate=row;self.hits=1
        if self.hits<3:return
        top=max(initial[1],needed_top-4)
        bottom=min(initial[1]+initial[3],needed_bottom+4)
        self.refined=(initial[0],top,initial[2],bottom-top)

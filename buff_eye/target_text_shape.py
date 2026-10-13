"""Bounded single-digit timer fallback using calibration ink and shadow shapes."""
import json
from pathlib import Path
import cv2
import numpy as np
from .text import foreground, segments


def feature(red):
    value=red.astype(np.float32)
    return value-cv2.GaussianBlur(value,(0,0),2)


class TargetShapeReader:
    def __init__(self, path, *, small_geometry=False):
        self.small_geometry=small_geometry
        self.shapes=json.loads(Path(path).read_text(encoding='utf-8')).get('shapes',[])
        self.cache={}

    def templates(self, width):
        if width not in self.cache:
            result=[]
            seen=set()
            for shape in self.shapes:
                raw=np.array(shape['red_channel'],np.uint8)
                for adjustment in ((.7,.8,.9,1.,1.1) if self.small_geometry and width<=26 else (.9,1.,1.1)):
                    scale=width/shape['icon_width']*adjustment
                    size=(round(raw.shape[1]*scale),round(raw.shape[0]*scale))
                    if min(size)<4: continue
                    template=feature(cv2.resize(raw,size,interpolation=cv2.INTER_AREA))
                    # Identical calibration shapes cannot change the per-character
                    # maximum score; compare each distinct shape only once.
                    key=(shape['character'],template.shape,template.tobytes())
                    if key in seen:
                        continue
                    seen.add(key)
                    result.append((shape['character'],template))
            # Bound cache even for callers outside the current 26..48px detector.
            if len(self.cache)>=24:
                self.cache.clear()
            self.cache[width]=result
        return self.cache[width]

    def read(self, frame, icon):
        x,y,w,h=icon
        # At small raster sizes the first antialiased text row can immediately
        # follow the border. Do not discard it with a fixed two-pixel gap.
        top=y+h if self.small_geometry and w<=26 else y+h+2
        height=max(18,round(w*.55))
        if self.small_geometry:height=min(height,frame.shape[0]-top)
        result={'status':'unreadable','raw_text':None,'value':None,'unit':None,
                'source':'observed','method':'target_ink_shadow','reason':'shape_confidence'}
        if x<0 or top<0 or x+w>frame.shape[1] or height<8 or top+height>frame.shape[0]:
            return result
        crop=frame[top:top+height,x:x+w]
        field=feature(crop[:,:,2])
        ranked={}
        for char,template in self.templates(w):
            th,tw=template.shape
            if th>height or tw>w: continue
            whole=cv2.matchTemplate(field,template,cv2.TM_CCOEFF_NORMED)
            digit=cv2.matchTemplate(field,template[:,:round(tw*.45)],cv2.TM_CCOEFF_NORMED)
            score=(whole+digit[:whole.shape[0],:whole.shape[1]])/2
            # The original timer is centered under the icon. A suffix of 13 or
            # 0.3 must not be accepted as a centered single digit.
            positions=np.arange(score.shape[1])+tw/2
            score[:,np.abs(positions-w/2)>1.5]=-1
            _,best,_,(xx,yy)=cv2.minMaxLoc(score)
            if best>ranked.get(char,(-1,))[0]:
                ranked[char]=(best,float(whole[yy,xx]),float(digit[yy,xx]),xx,yy,tw,th)
        values=sorted(ranked.items(),key=lambda pair:pair[1][0],reverse=True)
        if len(values)<2: return result
        char,(score,whole,digit,xx,yy,tw,th)=values[0]
        margin=score-values[1][1][0]
        result.update(score=score,margin=margin,candidate=char+'초')
        if score<.75 or margin<.07 or digit<.70 or whole<.65:
            return result
        # Retain visible leading digits / decimal marks instead of dropping
        # them as scenery. This fallback is only for complete single integers.
        mask=foreground(crop,'target')
        if any(bx+bw<=xx and bh>=3 for (bx,by,bw,bh),_ in segments(mask[2:])):
            result['reason']='leading_text'
            return result
        result.update(status='read',raw_text=char+'초',value=int(char),unit='s',reason=None,
                      bbox=[x+xx,top+yy,tw,th])
        return result

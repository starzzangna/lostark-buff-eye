"""Bounded subpixel grayscale comparison of existing calibrated glyphs."""
import json
import re
import cv2
import numpy as np


class GrayIntegerReader:
    def __init__(self,path,exclude_asset=None):
        bank=json.loads(path.read_text(encoding='utf-8'))
        if bank['schema_version']!=1:raise ValueError('Unsupported gray glyph bank')
        self.samples=[(g,np.array(g['bitmap'],np.uint8)) for g in bank['glyphs']
                      if g['source_asset_id']!=exclude_asset]
        self.cache={}

    def scores(self,query):
        h,w=query.shape
        if (h,w) not in self.cache:
            templates=[];characters=[]
            for metadata,source in self.samples:
                sw,sh=metadata['source_bbox'][2:]
                if abs(round(sw*h/sh)-w)>1:continue
                for dy in (-.5,0,.5):
                    for dx in (-.5,0,.5):
                        xx,yy=np.meshgrid((np.arange(w)+.5+dx)*sw/w+2.5,
                                          (np.arange(h)+.5+dy)*sh/h+2.5)
                        t=cv2.remap(source,xx.astype('float32'),yy.astype('float32'),cv2.INTER_LINEAR).astype(float).ravel()
                        t-=t.mean();norm=np.linalg.norm(t)
                        if norm==0:continue
                        templates.append(t/norm);characters.append(metadata['character'])
            if len(self.cache)>=32:self.cache.clear()
            self.cache[h,w]=(np.asarray(templates),characters)
        templates,characters=self.cache[h,w]
        q=query.astype(float).ravel();q-=q.mean();norm=np.linalg.norm(q)
        if norm==0 or not len(templates):return {}
        values=np.sum(templates*(q/norm),axis=1)
        return {c:float(max(v for v,k in zip(values,characters) if k==c)) for c in set(characters)}

    def read(self,crop,pieces,threshold,margin,*,min_height=6,allow_single=False):
        # Never infer a missing decimal, unit, or digit. Single digits need
        # the full unit height: a shaved diagonal can become another digit.
        if len(pieces) not in ((2,3,4) if allow_single else (3,4)):return None
        boxes=[b for b,_ in pieces]
        if not all(min_height<=b[3]<=8 for b in boxes):return None
        if len(boxes)==2 and boxes[0][3]!=boxes[1][3]:return None
        if min(b[3] for b in boxes)<max(b[3] for b in boxes)*.8:return None
        if abs((boxes[0][0]+boxes[-1][0]+boxes[-1][2]-crop.shape[1])/2)>2.5:return None
        value=cv2.cvtColor(crop,cv2.COLOR_BGR2HSV)[:,:,2]
        characters=[]
        for box in boxes:
            x,y,w,h=box
            ranks=sorted(self.scores(value[y:y+h,x:x+w]).items(),key=lambda p:-p[1])
            if len(ranks)<2 or ranks[0][1]<threshold or ranks[0][1]-ranks[1][1]<margin:return None
            characters.append(dict(bbox=box,candidate=ranks[0][0],score=ranks[0][1],margin=ranks[0][1]-ranks[1][1]))
        text=''.join(c['candidate'] for c in characters)
        pattern=r'[1-9][0-9]{0,2}초' if allow_single else r'[1-9][0-9]{1,2}초'
        if not re.fullmatch(pattern,text):return None
        return dict(status='read',raw_text=text,value=int(text[:-1]),unit='s',
                    score=min(c['score'] for c in characters),source='observed',
                    method='calibrated_gray_integer',reason=None,characters=characters)

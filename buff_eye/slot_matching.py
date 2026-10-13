"""Frame-local conservative bounds shared across icon slots."""
import cv2
import numpy as np
from .target_matching import ColorSearchFrame


class SlotColorGate:
    def __init__(self,frame,proposals):
        self.maps={}
        self.search=None
        if not proposals:return
        boxes=[]
        for proposal in proposals:
            x,y,w,h=proposal.bbox
            pad=max(3,round(w*.14))
            boxes.append((max(0,x-pad),max(0,y-pad),
                          min(frame.shape[1],x+w+pad),min(frame.shape[0],y+h+pad)))
        self.left=min(b[0] for b in boxes);self.top=min(b[1] for b in boxes)
        right=max(b[2] for b in boxes);bottom=max(b[3] for b in boxes)
        self.search=ColorSearchFrame(frame[self.top:bottom,self.left:right])

    def could_match(self,key,template,mask,threshold,left,top,right,bottom):
        if self.search is None:return True
        ix,iy,iw,ih=cv2.boundingRect(mask)
        if not iw or not ih:return True
        if key not in self.maps:
            self.maps[key]=self.search.possible_positions(
                template[iy:iy+ih,ix:ix+iw],threshold-.001)
        x=left+ix-self.left;y=top+iy-self.top
        h,w=template.shape[:2]
        return bool(np.any(self.maps[key][y:y+bottom-top-h+1,x:x+right-left-w+1]))

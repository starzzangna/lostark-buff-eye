"""Read-only Windows client geometry and conservative symmetric black bars."""
import ctypes
from ctypes import wintypes as wt
import threading
import time
import numpy as np

_lock=threading.Lock()
_cache={}


def content_rect(frame):
    h,w=frame.shape[:2]
    black=frame[:,:,0]<=5
    black &= frame[:,:,1]<=5
    black &= frame[:,:,2]<=5
    mx,my=max(1,w//100),max(1,h//100)
    rows=black[:,mx:w-mx].mean(axis=1)>.998
    cols=black[my:h-my].mean(axis=0)>.998
    def margins(values):
        def edge(run):
            # Some window borders add one or two non-black edge pixels.
            offset=next((i for i,v in enumerate(run[:4]) if v),0)
            return offset+next((i for i,v in enumerate(run[offset:]) if not v),len(run)-offset)
        a=edge(values);b=edge(values[::-1])
        if min(a,b)<2 or abs(a-b)>3 or max(a,b)>len(values)*.3:
            return 0,0
        return a,b
    top,bottom=margins(rows);left,right=margins(cols)
    return (left,top,w-left-right,h-top-bottom)


def game_client_rect():
    u=ctypes.windll.user32
    callback=ctypes.WINFUNCTYPE(wt.BOOL,wt.HWND,wt.LPARAM)
    u.GetWindowTextW.argtypes=[wt.HWND,wt.LPWSTR,ctypes.c_int]
    u.IsWindowVisible.argtypes=[wt.HWND];u.IsIconic.argtypes=[wt.HWND]
    u.GetClientRect.argtypes=[wt.HWND,ctypes.POINTER(wt.RECT)]
    u.ClientToScreen.argtypes=[wt.HWND,ctypes.POINTER(wt.POINT)]
    found=[]
    def visit(hwnd,_):
        title=ctypes.create_unicode_buffer(512)
        u.GetWindowTextW(hwnd,title,512)
        if not title.value.upper().startswith('LOST ARK') or not u.IsWindowVisible(hwnd):return True
        if u.IsIconic(hwnd):return True
        r=wt.RECT();p=wt.POINT()
        if u.GetClientRect(hwnd,ctypes.byref(r)) and u.ClientToScreen(hwnd,ctypes.byref(p)):
            if r.right>=320 and r.bottom>=200:found.append((p.x,p.y,r.right,r.bottom))
        return True
    u.EnumWindows.argtypes=[callback,wt.LPARAM];u.EnumWindows(callback(visit),0)
    return found[0] if len(found)==1 else None


def locate(provider,monitor):
    key=monitor.stable_id
    with _lock:
        now=time.monotonic()
        previous=_cache.get(key)
        if previous and previous[0]>now and previous[2]==monitor.bounds:return previous[1]
        client=game_client_rect()
        result=None
        if client:
            x,y,w,h=client
            x-=monitor.bounds.left;y-=monitor.bounds.top
            if x>=0 and y>=0 and x+w<=monitor.bounds.width and y+h<=monitor.bounds.height:
                from .capture import ScreenCaptureRequest,CaptureTargetKind,CaptureRegion
                source=provider.create_source(ScreenCaptureRequest(CaptureTargetKind.MONITOR,key,CaptureRegion(x,y,w,h)))
                source.open()
                try:frame=source.read()
                finally:source.close()
                left,top,cw,ch=content_rect(frame)
                result=(x+left,y+top,cw,ch)
        _cache[key]=(now+1,result,monitor.bounds)
        return result


def cached(monitor_id):
    value=_cache.get(monitor_id)
    return value[1] if value else None


def transform_roi(rect,reference,current,region_id):
    """HUD uses a horizontal center and either a top or bottom anchor."""
    x,y,w,h=rect;rx,ry,rw,rh=reference;cx,cy,cw,ch=current
    ratio=ch/rh
    left=cx+cw/2+(x-rx-rw/2)*ratio
    if region_id=='self_buff':top=cy+ch+(y-ry-rh)*ratio
    else:top=cy+(y-ry)*ratio
    return (round(left),round(top),max(1,round(w*ratio)),max(1,round(h*ratio)))

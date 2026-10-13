"""Conservative calibrated yellow-text reader; no labels used at inference.

Integer seconds and one-decimal values below one second ending in 초 are valid.
Unknown units, unsupported decimals, merged characters and low margins abstain.
"""
import json
import re
from collections import OrderedDict
from pathlib import Path
import cv2
import numpy as np


def foreground(crop, text_color='yellow'):
    if crop is None or not crop.size:
        return np.zeros((0, 0), np.uint8)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    if text_color == 'target':
        color = cv2.inRange(hsv, np.array([0, 25, 140]), np.array([25, 255, 255]))
    elif text_color == 'target_luminance':
        # Restricted to the target icon's timer crop. Local shadow contrast and
        # calibrated glyph confidence still apply to these desaturated strokes.
        color = cv2.inRange(hsv, np.array([0, 0, 90]), np.array([179, 255, 255]))
    elif text_color == 'green':
        color = cv2.inRange(hsv, np.array([15, 30, 140]), np.array([95, 255, 255]))
    else:
        color = cv2.inRange(hsv, np.array([15, 70, 140]), np.array([45, 255, 255]))
    value = hsv[:, :, 2]
    # The text's dark shadow is local contrast evidence; yellow scenery alone
    # does not qualify. Opening estimates the broad background below thin strokes.
    base = cv2.morphologyEx(value, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    contrast = cv2.subtract(value, base)
    mask = cv2.bitwise_and(color, np.where(contrast >= 35, 255, 0).astype(np.uint8))
    # Single-pixel color noise can otherwise join two adjacent characters.
    original = mask.copy()
    mask[:, np.count_nonzero(mask, axis=0) <= 1] = 0
    # Preserve thin horizontal dots lost by column denoising. Only isolated
    # compact components between surviving glyph columns can be punctuation.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(original, connectivity=8)
    columns = np.any(mask, axis=0)
    for index in range(1, count):
        x,y,w,h,pixels = stats[index]
        if (2 <= pixels <= 8 and w <= 4 and h <= 3
                and y >= original.shape[0] // 2
                and columns[:x].any() and columns[x+w:].any()
                and not columns[max(0,x-1):min(len(columns),x+w+1)].any()):
            mask[labels == index] = 255
    return mask


def segments(mask):
    """Projection segmentation; retain narrow glyphs, reject isolated noise."""
    if not mask.size:
        return []
    columns = np.any(mask > 0, axis=0)
    edges = np.diff(np.pad(columns.astype(np.int8), (1, 1)))
    result = []
    for left, right in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        part = mask[:, left:right]
        ys = np.flatnonzero(np.any(part > 0, axis=1))
        pixels = np.count_nonzero(part)
        is_decimal = len(ys) <= 3 and right-left <= 4 and pixels >= 2
        if (len(ys) < 3 or pixels < 4) and not is_decimal:
            continue
        top, bottom = int(ys[0]), int(ys[-1]) + 1
        result.append(([int(left), top, int(right-left), bottom-top], part[top:bottom].copy()))
    return result


def chromatic_decimal_mask(crop, seed):
    """Keep strong strokes; recover only compact punctuation of their hue."""
    if not crop.size or not np.any(seed):return None
    hsv=cv2.cvtColor(crop,cv2.COLOR_BGR2HSV)
    hues=hsv[:,:,0][seed>0].astype(float)
    hue=float(np.median(hues))
    # Five OpenCV hue steps are ten degrees. Demand a coherent text color.
    if np.mean(np.abs(hues-hue)<=5)<.6:return None
    color=(np.abs(hsv[:,:,0].astype(float)-hue)<=5)&(hsv[:,:,1]>=30)&(hsv[:,:,2]>=140)
    mask=np.where(color&(seed>0),255,0).astype(np.uint8)
    columns=np.any(mask,axis=0)
    count,labels,stats,_=cv2.connectedComponentsWithStats(color.astype(np.uint8),connectivity=8)
    dots=0
    for index in range(1,count):
        x,y,w,h,pixels=stats[index]
        if (2<=pixels<=8 and w<=4 and h<=3 and y>=crop.shape[0]//2
                and columns[:x].any() and columns[x+w:].any()
                and not columns[max(0,x-1):min(len(columns),x+w+1)].any()):
            mask[labels==index]=255
            dots+=1
    if dots!=1:return None
    count,labels,stats,_=cv2.connectedComponentsWithStats(mask,connectivity=8)
    for index in range(1,count):
        if stats[index,cv2.CC_STAT_AREA]<2:mask[labels==index]=0
    return mask


def component_segments(mask):
    """Group connected ink by overlapping x extents, retaining every pixel.

    Adjacent glyph columns need not share a connected stroke. Disconnected
    parts of a unit remain grouped when their horizontal extents overlap.
    """
    if not mask.size:return []
    count,labels,stats,_=cv2.connectedComponentsWithStats(mask,connectivity=8)
    groups=[]
    for index in sorted(range(1,count),key=lambda i:int(stats[i,0])):
        x,y,w,h,_=map(int,stats[index])
        if groups and x<groups[-1][1]:
            groups[-1][1]=max(groups[-1][1],x+w)
            groups[-1][2].append(index)
        else:
            groups.append([x,x+w,[index]])
    result=[]
    for left,right,indices in groups:
        pixels=np.isin(labels[:,left:right],indices)
        ys=np.flatnonzero(np.any(pixels,axis=1))
        top,bottom=int(ys[0]),int(ys[-1])+1
        result.append(([left,top,right-left,bottom-top],
                       np.where(pixels[top:bottom],255,0).astype(np.uint8)))
    return result


def small_decimal_mask(crop, seed, text_color='yellow'):
    """Recover dim punctuation only in the gap between intact small glyphs.

    Window scaling can put a dot below the absolute brightness cutoff. Its
    original hue, local shadow contrast and baseline must still be visible.
    Never repair digit ink or insert a dot into a connected glyph.
    """
    pieces=segments(seed)
    if len(pieces)!=3 or any(not 4<=b[3]<=8 for b,_ in pieces):return None
    first,second,unit=[b for b,_ in pieces]
    left=first[0]+first[2]+1
    right=second[0]-1
    if right<=left:return None
    baseline=max(b[1]+b[3] for b,_ in pieces)
    hsv=cv2.cvtColor(crop,cv2.COLOR_BGR2HSV)
    hues=hsv[:,:,0][seed>0].astype(float)
    hue=float(np.median(hues))
    if np.mean(np.abs(hues-hue)<=5)<.6:return None
    value=hsv[:,:,2]
    base=cv2.morphologyEx(value,cv2.MORPH_OPEN,np.ones((5,5),np.uint8))
    # Dim antialiased ink mixes with the background. Use the existing color
    # category range, rather than the hue of only the brightest strokes.
    upper=95 if text_color=='green' else 45
    saturation=30 if text_color=='green' else 70
    weak=(hsv[:,:,0]>=15)&(hsv[:,:,0]<=upper)&(hsv[:,:,1]>=saturation)&(cv2.subtract(value,base)>=35)
    gap=np.zeros_like(seed)
    gap[max(0,baseline-2):min(crop.shape[0],baseline+1),left:right]=np.where(
        weak[max(0,baseline-2):min(crop.shape[0],baseline+1),left:right],255,0)
    dots=component_segments(gap)
    if len(dots)!=1:return None
    box,glyph=dots[0]
    if not (2<=np.count_nonzero(glyph)<=8 and box[2]<=4 and box[3]<=3):return None
    return seed|gap


def fragmented_integer_mask(crop,seed):
    """Recover a dim diagonal split from a single digit's upright stroke.

    A leading tiny fragment is not a legal decimal. Preserve the measured
    gap before the unit and require a complete, calibrated single digit.
    """
    pieces=segments(seed)
    if len(pieces)!=3:return None
    fragment,stem,unit=[b for b,_ in pieces]
    if not (fragment[2]<=3 and fragment[3]<=3 and
            5<=stem[3]<=8 and stem[2]<=2 and stem[3]==unit[3] and
            stem[1]==unit[1] and unit[2]>=4):return None
    hsv=cv2.cvtColor(crop,cv2.COLOR_BGR2HSV)
    value=hsv[:,:,2]
    contrast=cv2.subtract(value,cv2.morphologyEx(value,cv2.MORPH_OPEN,np.ones((5,5),np.uint8)))
    weak=(hsv[:,:,0]>=15)&(hsv[:,:,0]<=45)&(hsv[:,:,1]>=70)&(contrast>=35)
    mask=np.zeros_like(seed)
    top,bottom=unit[1],unit[1]+unit[3]
    left,right=max(0,fragment[0]-1),min(crop.shape[1],unit[0]+unit[2]+1)
    mask[top:bottom,left:right]=np.where(weak[top:bottom,left:right],255,0)
    # Retain the original empty separator, never split strong observed ink.
    separator=unit[0]-1
    if np.any(seed[:,separator]):return None
    mask[:,separator]=0
    if np.any((seed>0)&(mask==0)):return None
    return mask


def target_chromatic_masks(crop, seed):
    """Separate orange timer ink from differently colored damage behind it."""
    if not crop.size or not np.any(seed):return []
    hsv=cv2.cvtColor(crop,cv2.COLOR_BGR2HSV)
    hues=hsv[:,:,0][seed>0].astype(float)
    hue=float(np.median(hues))
    if np.mean(np.abs(hues-hue)<=5)<.6:return []
    color=(np.abs(hsv[:,:,0].astype(float)-hue)<=5)&(hsv[:,:,1]>=30)&(hsv[:,:,2]>=140)
    result=[]
    for pixels in (color&(seed>0),color):
        if not np.any(pixels):continue
        if np.count_nonzero(pixels&(seed>0))/np.count_nonzero(pixels)<.6:continue
        mask=np.where(pixels,255,0).astype(np.uint8)
        count,labels,stats,_=cv2.connectedComponentsWithStats(mask,connectivity=8)
        for index in range(1,count):
            if stats[index,cv2.CC_STAT_AREA]<2:mask[labels==index]=0
        result.append(mask)
    return result


def remove_target_specks(pieces):
    """Discard only tiny marks above the baseline or after the complete token."""
    tall=[box for box,_ in pieces if box[3]>3]
    if len(tall)<2:return pieces
    baseline=max(b[1]+b[3] for b in tall)
    end=max(b[0]+b[2] for b in tall)
    return [(b,g) for b,g in pieces if not
            (b[2]<=4 and b[3]<=3 and np.count_nonzero(g)<=8
             and (b[1]+b[3]<baseline-3 or b[0]>=end))]


def normalize(glyph, width_delta=0):
    h, w = glyph.shape
    width = max(1, min(28, round(w * 20 / h)+width_delta))
    resized = cv2.resize(glyph, (width, 20), interpolation=cv2.INTER_NEAREST)
    canvas = np.zeros((24, 32), np.uint8)
    x = (32-width)//2
    canvas[2:22, x:x+width] = resized
    return canvas


def similarity(a, b):
    # Small subpixel raster differences cause 1px shifts after normalization.
    best = 0.0
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            shifted = cv2.warpAffine(b, np.float32([[1,0,dx],[0,1,dy]]), (32,24), flags=cv2.INTER_NEAREST)
            aa, bb = a > 0, shifted > 0
            denom = np.count_nonzero(aa) + np.count_nonzero(bb)
            value = 2 * np.count_nonzero(aa & bb) / denom if denom else 0
            best = max(best, value)
    return best


class TimeReader:
    def __init__(self, bank_path, exclude_asset=None, extra_bank_path=None):
        self.path = Path(bank_path)
        bank = json.loads(self.path.read_text(encoding='utf-8'))
        if bank['schema_version'] != 1:
            raise ValueError('Unsupported glyph bank schema')
        if extra_bank_path is not None:
            extra = json.loads(Path(extra_bank_path).read_text(encoding='utf-8'))
            if extra['schema_version'] != 1:
                raise ValueError('Unsupported extra glyph bank schema')
            bank['glyphs'] = bank['glyphs'] + extra['glyphs']
        self.samples = [(g, np.array(g['bitmap'], np.uint8)) for g in bank['glyphs']
                        if g['source_asset_id'] != exclude_asset]
        self.threshold = bank.get('threshold', 0.78)
        self.margin = bank.get('margin', 0.035)
        shifted=[]; characters=[]
        for g,bitmap in self.samples:
            for dy in (-1,0,1):
                for dx in (-1,0,1):
                    shifted.append(cv2.warpAffine(bitmap,np.float32([[1,0,dx],[0,1,dy]]),(32,24),flags=cv2.INTER_NEAREST).reshape(-1)>0)
                    characters.append(g['character'])
        self.shifted=np.array(shifted,dtype=bool).reshape((-1,768))
        self.counts=np.count_nonzero(self.shifted,axis=1)
        self.character_indices={char:np.array([i for i,c in enumerate(characters) if c==char]) for char in set(characters)}
        self.native_cache={}
        self.fitted_native_cache={}
        self.glyph_score_cache=OrderedDict()
        self.gray_reader=None
        gray_path=self.path.with_name('glyph-gray-v1.json')
        if self.path.name=='glyph-bank-v5.json' and gray_path.exists():
            from .gray_text import GrayIntegerReader
            self.gray_reader=GrayIntegerReader(gray_path,exclude_asset)

    def _glyph_scores(self, normalized):
        # Only identical normalized ink reuses scores. No detection, timer,
        # frame or temporal decision is cached here.
        candidate=normalized.reshape(-1)>0
        key=np.packbits(candidate).tobytes()
        cached=self.glyph_score_cache.get(key)
        if cached is not None:
            self.glyph_score_cache.move_to_end(key)
            return cached
        denominators=self.counts+np.count_nonzero(candidate)
        values=np.divide(2*np.count_nonzero(self.shifted & candidate,axis=1),denominators,
                         out=np.zeros(len(denominators),dtype=float),where=denominators!=0)
        scores={char:float(np.max(values[indices])) for char,indices in self.character_indices.items()}
        self.glyph_score_cache[key]=scores
        if len(self.glyph_score_cache)>512:self.glyph_score_cache.popitem(last=False)
        return scores

    def _native_scores(self, glyph, fitted=False):
        """Compare at observed pixel size without magnifying raster offsets."""
        h,w=glyph.shape
        if fitted:
            # A one-pixel width rounding difference matters greatly at 6px.
            # Fit only aspect-compatible calibration glyphs to observed bounds.
            key=(h,w)
            if key not in self.fitted_native_cache:
                templates=[]
                for metadata,bitmap in self.samples:
                    yy,xx=np.where(bitmap>0)
                    if not len(yy):continue
                    source=bitmap[yy.min():yy.max()+1,xx.min():xx.max()+1]
                    if abs(round(source.shape[1]*h/source.shape[0])-w)>1:continue
                    templates.append((metadata['character'],cv2.resize(source,(w,h),interpolation=cv2.INTER_AREA)>=128))
                if len(self.fitted_native_cache)>=32:self.fitted_native_cache.clear()
                self.fitted_native_cache[key]=templates
            scores={}
            for char,template in self.fitted_native_cache[key]:
                score=2*np.count_nonzero((glyph>0)&template)/(np.count_nonzero(glyph)+np.count_nonzero(template))
                scores[char]=max(scores.get(char,0),float(score))
            return scores
        key=(h,w)
        if key not in self.native_cache:
            templates=[]
            for metadata,bitmap in self.samples:
                yy,xx=np.where(bitmap>0)
                if not len(yy):continue
                source=bitmap[yy.min():yy.max()+1,xx.min():xx.max()+1]
                width=max(1,round(source.shape[1]*h/source.shape[0]))
                if abs(width-w)>1:continue
                template=cv2.resize(source,(width,h),interpolation=cv2.INTER_AREA)>=128
                templates.append((metadata['character'],template))
            if len(self.native_cache)>=32:self.native_cache.clear()
            self.native_cache[key]=templates
        canvas=np.pad(glyph>0,((1,1),(2,2)))
        pixels=np.count_nonzero(glyph)
        scores={}
        for char,template in self.native_cache[key]:
            width=template.shape[1]
            denominator=pixels+np.count_nonzero(template)
            best=max(2*np.count_nonzero(canvas[dy:dy+h,dx:dx+width]&template)/denominator
                     for dy in range(3) for dx in range(canvas.shape[1]-width+1))
            scores[char]=max(scores.get(char,0),best)
        return scores

    def read(self, crop, text_color='yellow'):
        seed=foreground(crop,text_color)
        pieces = segments(seed)
        result = self._read_pieces(pieces, text_color)
        if result['reason']=='invalid_decimal_position' and text_color=='yellow':
            repaired=fragmented_integer_mask(crop,seed)
            if repaired is not None:
                restored=segments(repaired)
                attempt=self._read_pieces(restored,text_color,native='fitted')
                if attempt['status']=='read' and re.fullmatch(r'[1-9]초',attempt['raw_text']):
                    first,last=restored[0][0],restored[-1][0]
                    if abs((first[0]+last[0]+last[2]-crop.shape[1])/2)<=2.5:
                        attempt['method']='calibrated_fragmented_integer'
                        result=attempt
        if result['status']!='read' and text_color in ('yellow','green'):
            # Fit intact small integers at native size. Do not remove marks:
            # an absent decimal in 07 must still fail the normal grammar.
            if 2<=len(pieces)<=4 and all(7<=b[3]<=8 for b,_ in pieces):
                attempt=self._read_pieces(pieces,text_color,native='fitted')
                first,last=pieces[0][0],pieces[-1][0]
                if (attempt['status']=='read' and re.fullmatch(r'(?:0|00|[1-9][0-9]{0,2})초',attempt['raw_text'])
                        and abs((first[0]+last[0]+last[2]-crop.shape[1])/2)<=2.5
                        and min(b[3] for b,_ in pieces)>=max(b[3] for b,_ in pieces)*.8):
                    attempt['method']='calibrated_small_integer'
                    result=attempt
        if result['status']!='read' and text_color in ('yellow','green'):
            mask=small_decimal_mask(crop,seed,text_color)
            if mask is not None:
                recovered=segments(mask)
                attempt=self._read_pieces(recovered,text_color,native='fitted')
                if attempt['status']=='read' and re.fullmatch(r'0\.[0-9]초',attempt['raw_text']):
                    first,last=recovered[0][0],recovered[-1][0]
                    if abs((first[0]+last[0]+last[2]-crop.shape[1])/2)<=2.5:
                        attempt['method']='calibrated_small_decimal'
                        return attempt
        if result['reason']=='invalid_decimal_position' and len(pieces)==3:
            # A visible decimal followed by adjacent digit/unit columns can
            # fail projection segmentation. Never insert a dot or cut ink.
            separated=component_segments(seed)
            if len(separated)==4:
                attempt=self._read_pieces(separated,text_color)
                if attempt['reason']=='low_confidence':
                    attempt=self._read_pieces(separated,text_color,native=True)
                if attempt['status']=='read' and re.fullmatch(r'0\.[0-9]초',attempt['raw_text']):
                    first,last=separated[0][0],separated[-1][0]
                    if abs((first[0]+last[0]+last[2]-crop.shape[1])/2)<=2.5:
                        attempt['method']='calibrated_component_decimal'
                        return attempt
        if text_color=='target' and result['reason']=='invalid_decimal_position':
            cleaned=remove_target_specks(pieces)
            if len(cleaned)!=len(pieces):
                attempt=self._read_pieces(cleaned,text_color)
                if attempt['status']=='read':
                    first,last=cleaned[0][0],cleaned[-1][0]
                    if abs((first[0]+last[0]+last[2]-crop.shape[1])/2)<=2.5:
                        result=attempt
                        result['method']='calibrated_target_speck_filter'
        if result['reason']=='low_confidence':
            # One source pixel at a 10px font becomes two at normalized height.
            # Require both narrower and wider whole-token readings to agree;
            # a single favorable width must never select a different number.
            variants=[self._read_pieces(pieces,text_color,width_delta=d) for d in (-2,2)]
            if all(r['status']=='read' for r in variants) and variants[0]['raw_text']==variants[1]['raw_text']:
                result=min(variants,key=lambda r:r['score'])
                result['method']='calibrated_glyphs_width_consensus'
        if result['status']!='read' and text_color=='green':
            mask=chromatic_decimal_mask(crop,seed)
            if mask is not None:
                decimal=self._read_pieces(segments(mask),text_color)
                if decimal['status']=='read' and re.fullmatch(r'0\.[0-9]초',decimal['raw_text']):
                    boxes=[c['bbox'] for c in decimal['characters']]
                    center=(boxes[0][0]+boxes[-1][0]+boxes[-1][2])/2
                    if abs(center-crop.shape[1]/2)<=2.5:
                        decimal['method']='calibrated_chromatic_decimal'
                        return decimal
        if result['status']!='read' and text_color=='target':
            accepted=[]
            for mask in target_chromatic_masks(crop,seed):
                pieces2=segments(mask)
                attempt=self._read_pieces(pieces2,text_color)
                if attempt['status']!='read' or not re.fullmatch(r'(?:0|00|[1-9][0-9]{0,2})초',attempt['raw_text']):continue
                first,last=pieces2[0][0],pieces2[-1][0]
                if abs((first[0]+last[0]+last[2]-crop.shape[1])/2)>2.5:continue
                if any(b[3]<last[3]*.65 or np.count_nonzero(seed[b[1]:b[1]+b[3],b[0]:b[0]+b[2]])<2 for b,_ in pieces2):continue
                accepted.append(attempt)
            if accepted and len({r['raw_text'] for r in accepted})==1:
                result=max(accepted,key=lambda r:r['score'])
                result['method']='calibrated_target_chromatic'
        if result['status'] == 'read' or text_color.startswith('target'):
            return result
        candidates = []
        if len(pieces) >= 3:
            first, last = pieces[0][0], pieces[-1][0]
            # Only isolated edge specks: never erase a full-height leading
            # digit or a decimal point inside the token.
            if first[2] <= 2 and first[3] <= 3:
                candidates.append(pieces[1:])
            if last[2] == 1:
                candidates.append(pieces[:-1])
        reads = []
        for candidate in candidates:
            left = candidate[0][0][0]
            end = candidate[-1][0]
            if abs((left+end[0]+end[2]-crop.shape[1])/2) > 2.5:
                continue
            attempt = self._read_pieces(candidate, text_color)
            if attempt['status'] == 'read':
                reads.append(attempt)
        if reads and len({r['raw_text'] for r in reads}) == 1:
            return max(reads, key=lambda r:r['score'])
        if result['status']!='read' and text_color in ('yellow','green') and self.gray_reader is not None:
            attempt=self.gray_reader.read(crop,segments(seed),self.threshold,self.margin)
            if attempt is not None:return attempt
        return result

    def _small_ink_read(self,crop,seed,text_color):
        """Require binary and original-intensity agreement on small complete ink.

        Color thresholds can erase a dim 7's top stroke. Restore local shadow
        contrast only inside the token, and never accept binary evidence alone.
        """
        if not crop.size or crop.shape[1]>36:return None
        pieces=segments(foreground(crop,'target_luminance'))
        if not 2<=len(pieces)<=4:return None
        boxes=[b for b,_ in pieces]
        if not all(5<=b[3]<=8 for b in boxes):return None
        if min(b[3] for b in boxes)<max(b[3] for b in boxes)*.7:return None
        if abs((boxes[0][0]+boxes[-1][0]+boxes[-1][2]-crop.shape[1])/2)>2:return None
        # Every character must have colored ink from the original mask.
        if any(np.count_nonzero(seed[y:y+h,x:x+w])<2 for x,y,w,h in boxes):return None
        binary=self._read_pieces(pieces,'green',native='fitted')
        if binary['status']!='read' or not re.fullmatch(r'(?:0|00|[1-9][0-9]{0,2})초',binary['raw_text']):return None
        value=cv2.cvtColor(crop,cv2.COLOR_BGR2HSV)[:,:,2]
        for char,(x,y,w,h) in zip(binary['characters'],boxes):
            ranked=sorted(self.gray_reader.scores(value[y:y+h,x:x+w]).items(),key=lambda p:-p[1])
            if not ranked or ranked[0][0]!=char['candidate'] or ranked[0][1]<self.threshold:return None
        binary['method']='calibrated_small_ink_consensus'
        return binary

    def read_small_region(self,frame,box,text_color='yellow'):
        if self.gray_reader is None:return None
        x,y,w,h=box
        reads=[]
        for offset,extra in ((0,0),(-1,0),(1,0),(-1,2),(-2,4)):
            yy=y+offset
            end=min(yy+h+extra,frame.shape[0])
            if x<0 or yy<0 or x+w>frame.shape[1] or end-yy<h:continue
            crop=frame[yy:end,x:x+w]
            attempt=self._small_ink_read(crop,foreground(crop,text_color),text_color)
            if attempt is None:attempt=self._gray_decimal_read(crop,text_color)
            if attempt is None:
                attempt=self.gray_reader.read(crop,segments(foreground(crop,text_color)),
                                              self.threshold,self.margin,min_height=5,allow_single=text_color!='target')
            if attempt is None and text_color in ('yellow','green'):
                attempt=self._gray_seeded_integer(crop,text_color)
            if attempt is None and text_color=='green':
                attempt=self._bright_green_integer(crop)
            if attempt is None and text_color=='yellow':
                attempt=self._yellow_chroma_read(crop)
            if attempt is None and text_color=='yellow':
                attempt=self._yellow_edge_integer(crop)
            if attempt:
                attempt['bbox']=[x,yy,w,end-yy]
                reads.append(attempt)
        if reads and len({r['raw_text'] for r in reads})==1:
            return max(reads,key=lambda r:r['score'])
        return None

    def read_tiny_target_chroma(self,frame,icon):
        """Read a complete tiny single-second token with two color masks.

        Equal full glyph heights and wide unit strokes prevent clipped 6/5
        and decimal suffixes from being accepted as single integers.
        """
        x,y,w,h=icon
        top=y+h
        if not 18<=w<=20 or x<0 or top<0 or x+w>frame.shape[1] or top+12>frame.shape[0]:return None
        crop=frame[top:top+12,x:x+w]
        b,g,r=cv2.split(crop.astype(np.float32))
        chroma=r-np.maximum(g,b)
        reads=[]
        for cutoff in (30,35):
            pieces=segments(np.where(chroma>=cutoff,255,0).astype(np.uint8))
            if len(pieces)!=2:return None
            (dx,dy,dw,dh),(ux,uy,uw,uh)=[box for box,_ in pieces]
            if not 5<=dh<=7 or dh!=uh or dy!=uy or min(dw,uw)<3:return None
            if abs((dx+ux+uw-w)/2)>1.5:return None
            chars=[];scores=[]
            for (xx,yy,ww,hh),_ in pieces:
                ranks=sorted(self.gray_reader.scores(chroma[yy:yy+hh,xx:xx+ww]).items(),key=lambda t:-t[1])
                if len(ranks)<2 or ranks[0][1]<.80 or ranks[0][1]-ranks[1][1]<.10:return None
                chars.append(ranks[0][0]);scores.append(ranks[0][1])
            raw=''.join(chars)
            if not re.fullmatch(r'[1-9]초',raw) or scores[0]<.90:return None
            reads.append((raw,min(scores)))
        if reads[0][0]!=reads[1][0]:return None
        raw=reads[0][0]
        return dict(status='read',raw_text=raw,value=int(raw[0]),unit='s',source='observed',
                    score=min(s for _,s in reads),method='tiny_target_chroma_consensus',reason=None,
                    bbox=[x,top,w,12])

    def _yellow_chroma_read(self,crop):
        """Recover yellow ink joined by similarly bright, less yellow scenery.

        Both masks must contain complete tokens and the same observed value.
        Punctuation is an actual compact component, never inferred from 0x.
        """
        if not crop.size or crop.shape[1]>40:return None
        b,g,r=cv2.split(crop.astype(np.float32))
        chroma=np.minimum(g,r)-b
        reads=[]
        for cutoff in (80,100):
            mask=np.where(chroma>=cutoff,255,0).astype(np.uint8)
            pieces=segments(mask)
            if not 2<=len(pieces)<=4:return None
            boxes=[box for box,_ in pieces]
            tall=[box for box in boxes if box[3]>3]
            if not tall or any(not 5<=box[3]<=12 for box in tall):return None
            if min(box[3] for box in tall)<max(box[3] for box in tall)*.8:return None
            if abs((boxes[0][0]+boxes[-1][0]+boxes[-1][2]-crop.shape[1])/2)>2:return None
            baseline=max(box[1]+box[3] for box in tall)
            chars=[];scores=[]
            for index,((x,y,w,h),glyph) in enumerate(pieces):
                if h<=3:
                    if (len(pieces)!=4 or index!=1 or w>2 or np.count_nonzero(glyph)>6
                            or abs(y+h-baseline)>2):return None
                    chars.append('.');continue
                ranks=sorted(self.gray_reader.scores(chroma[y:y+h,x:x+w]).items(),key=lambda t:-t[1])
                if not ranks or ranks[0][1]<self.threshold:return None
                char,score=ranks[0]
                if char=='초':
                    binary=sorted(self._native_scores(glyph,fitted=True).items(),key=lambda t:-t[1])
                    if index!=len(pieces)-1 or not binary or binary[0][0]!='초' or binary[0][1]<self.threshold:return None
                elif len(ranks)<2 or ranks[0][1]-ranks[1][1]<self.margin:return None
                chars.append(char);scores.append(score)
            raw=''.join(chars)
            if not re.fullmatch(r'(?:[1-9][0-9]{0,2}|0\.[0-9])초',raw):return None
            reads.append((raw,min(scores)))
        if reads[0][0]!=reads[1][0]:return None
        raw=reads[0][0]
        return dict(status='read',raw_text=raw,value=float(raw[:-1]) if '.' in raw else int(raw[:-1]),
                    unit='s',score=min(s for _,s in reads),source='observed',
                    method='calibrated_yellow_chroma_consensus',reason=None)

    def _yellow_edge_integer(self,crop):
        """Restore one clipped integer glyph only when both side crops agree.

        This path needs two or three digits and a verified unit. It cannot
        interpret decimal tokens, missing units, or a leading zero.
        """
        if not crop.size or not 28<=crop.shape[1]<=40:return None
        b,g,r=cv2.split(crop.astype(np.float32))
        chroma=np.minimum(g,r)-b
        pieces=segments(np.where(chroma>=100,255,0).astype(np.uint8))
        if len(pieces) not in (3,4):return None
        boxes=[box for box,_ in pieces]
        if any(not 8<=h<=12 for x,y,w,h in boxes):return None
        if max(y+h for x,y,w,h in boxes)-min(y+h for x,y,w,h in boxes)>1:return None
        if abs((boxes[0][0]+boxes[-1][0]+boxes[-1][2]-crop.shape[1])/2)>2:return None
        chars=[];scores=[];restored=0
        for index,((x,y,w,h),glyph) in enumerate(pieces):
            ranks=sorted(self.gray_reader.scores(chroma[y:y+h,x:x+w]).items(),key=lambda t:-t[1])
            if index==len(pieces)-1:
                binary=sorted(self._native_scores(glyph,fitted=True).items(),key=lambda t:-t[1])
                if (not ranks or ranks[0][0]!='초' or ranks[0][1]<self.threshold
                        or not binary or binary[0][0]!='초' or binary[0][1]<self.threshold):return None
                chars.append('초');scores.append(ranks[0][1]);continue
            valid=(len(ranks)>=2 and ranks[0][0].isdigit() and ranks[0][1]>=self.threshold
                   and ranks[0][1]-ranks[1][1]>=self.margin)
            if valid:
                chars.append(ranks[0][0]);scores.append(ranks[0][1]);continue
            # Need free space on BOTH sides; never borrow an adjacent glyph.
            left=boxes[index-1][0]+boxes[index-1][2] if index else 0
            right=boxes[index+1][0]
            if x-1<=left or x+w+1>=right:return None
            votes=[]
            for start in (x-1,x):
                wide=sorted(self.gray_reader.scores(chroma[y:y+h,start:start+w+1]).items(),key=lambda t:-t[1])
                if (len(wide)<2 or not wide[0][0].isdigit() or wide[0][1]<.90
                        or wide[0][1]-wide[1][1]<self.margin):return None
                votes.append(wide[0])
            if votes[0][0]!=votes[1][0]:return None
            restored+=1;chars.append(votes[0][0]);scores.append(min(v[1] for v in votes))
        raw=''.join(chars)
        if restored!=1 or not re.fullmatch(r'[1-9][0-9]{1,2}초',raw):return None
        return dict(status='read',raw_text=raw,value=int(raw[:-1]),unit='s',score=min(scores),
                    source='observed',method='calibrated_yellow_edge_consensus',reason=None)

    def _bright_green_integer(self,crop):
        """Separate dark green timer ink from a bright neutral backdrop.

        Brightness contrast cannot recover strokes darker than the scenery.
        Require two chromatic masks to agree and verify their glyphs against
        both binary and continuous chroma templates. Never insert punctuation.
        """
        if not crop.size or crop.shape[1]>36:return None
        b,g,r=cv2.split(crop.astype(np.float32))
        neutral=(np.maximum.reduce([b,g,r])-np.minimum.reduce([b,g,r])<60)&((b+g+r)/3>140)
        if np.mean(neutral)<.35:return None
        chroma=g-(b+r)/2
        reads=[]
        for cutoff in (20,30):
            pieces=segments(np.where(chroma>=cutoff,255,0).astype(np.uint8))
            if not 2<=len(pieces)<=4:return None
            boxes=[box for box,_ in pieces]
            if any(not 5<=box[3]<=8 for box in boxes):return None
            if min(box[3] for box in boxes)<max(box[3] for box in boxes)*.8:return None
            if abs((boxes[0][0]+boxes[-1][0]+boxes[-1][2]-crop.shape[1])/2)>2:return None
            attempt=self._read_pieces(pieces,'green',native='fitted')
            if attempt['status']!='read' or not re.fullmatch(r'[1-9][0-9]{0,2}초',attempt['raw_text']):return None
            for char,(x,y,w,h) in zip(attempt['characters'],boxes):
                ranked=sorted(self.gray_reader.scores(chroma[y:y+h,x:x+w]).items(),key=lambda item:-item[1])
                if not ranked or ranked[0][0]!=char['candidate'] or ranked[0][1]<self.threshold:return None
                if char['candidate']!='초' and (len(ranked)<2 or ranked[0][1]-ranked[1][1]<self.margin):return None
            reads.append(attempt)
        if len({r['raw_text'] for r in reads})!=1:return None
        result=min(reads,key=lambda r:r['score'])
        result['method']='calibrated_bright_green_consensus'
        return result

    def _gray_seeded_integer(self,crop,text_color):
        """Use the observed unit baseline to recover dim digit edges.

        Restrict luminance segmentation to the unit's line, not scenery above
        or below. Narrow 1s otherwise lose their second antialiased column.
        All complete glyphs still need the original score and margin.
        """
        if crop.shape[1]>36:return None
        seed=foreground(crop,text_color);pieces=segments(seed)
        if not 2<=len(pieces)<=4:return None
        # One-column intensity vectors cannot distinguish a 1 from a resized
        # 7. Include its actual adjacent dim ink instead of reducing margin.
        luminance=foreground(crop,'target_luminance')
        widened=[];changed=False
        for (x,y,w,h),glyph in pieces:
            if w==1 and 5<=h<=8 and x+1<crop.shape[1] and np.count_nonzero(luminance[y:y+h,x+1])>=2:
                widened.append(([x,y,2,h],luminance[y:y+h,x:x+2]));changed=True
            else:widened.append(([x,y,w,h],glyph))
        if changed:
            attempt=self.gray_reader.read(crop,widened,self.threshold,self.margin,min_height=5,allow_single=True)
            if attempt:
                attempt['method']='calibrated_gray_narrow_ink'
                return attempt
        ux,uy,uw,uh=pieces[-1][0]
        if not 5<=uh<=8:return None
        line=crop[uy:uy+uh]
        restored=segments(foreground(line,'target_luminance'))
        if not 2<=len(restored)<=4:return None
        if any(np.count_nonzero(seed[uy+y:uy+y+h,x:x+w])<2 for (x,y,w,h),_ in restored):return None
        attempt=self.gray_reader.read(line,restored,self.threshold,self.margin,min_height=5,allow_single=True)
        if attempt:
            attempt['method']='calibrated_gray_unit_baseline'
            for char in attempt['characters']:char['bbox'][1]+=uy
        return attempt

    def _gray_decimal_read(self,crop,text_color):
        """Read complete 0.x ink; require an actual colored dot in the gap.

        Column denoising may erase a one-pixel decimal. It must be verified
        against the original pixels, never supplied from a predicted timer.
        """
        if text_color not in ('yellow','green') or crop.shape[1]>40:return None
        seed=foreground(crop,text_color)
        pieces=segments(seed)
        if len(pieces)==4 and pieces[1][0][2]<=2 and pieces[1][0][3]<=2:
            pieces=[pieces[0],pieces[2],pieces[3]]
        if len(pieces)!=3:return None
        # Saturation clipping can remove the top/bottom of a dim digit (4).
        # Recover only a complete, equally tall 0/digit/unit from luminance;
        # the original colored seed still supplies hue and real-dot evidence.
        if min(b[3] for b,_ in pieces)<max(b[3] for b,_ in pieces)*.8:
            restored=segments(foreground(crop,'target_luminance'))
            if len(restored)==4 and restored[1][0][2]<=2 and restored[1][0][3]<=2:
                restored=[restored[0],restored[2],restored[3]]
            if len(restored)!=3:return None
            if any(np.count_nonzero(seed[y:y+h,x:x+w])<2 for (x,y,w,h),_ in restored):return None
            full_height=max(b[3] for b,_ in pieces)
            # Keep intact colored glyph bounds; broadening a complete unit
            # introduces its halo and can erase the comparison alternatives.
            pieces=[old if old[0][3]>=full_height*.8 else new
                    for old,new in zip(pieces,restored)]
        boxes=[b for b,_ in pieces]
        if not all(6<=b[3]<=10 for b in boxes):return None
        if min(b[3] for b in boxes)<max(b[3] for b in boxes)*.8:return None
        if abs((boxes[0][0]+boxes[-1][0]+boxes[-1][2]-crop.shape[1])/2)>2:return None
        hsv=cv2.cvtColor(crop,cv2.COLOR_BGR2HSV);value=hsv[:,:,2]
        chars=[];scores=[]
        for x,y,w,h in boxes:
            ranked=sorted(self.gray_reader.scores(value[y:y+h,x:x+w]).items(),key=lambda p:-p[1])
            if len(ranked)<2 or ranked[0][1]<self.threshold or ranked[0][1]-ranked[1][1]<self.margin:return None
            chars.append(ranked[0][0]);scores.append(ranked[0][1])
        if chars[0]!='0' or chars[2]!='초' or chars[1] not in '0123456789':return None
        left=boxes[0][0]+boxes[0][2];right=boxes[1][0]
        baseline=max(b[1]+b[3] for b in boxes)
        if right-left<2:return None
        hues=hsv[:,:,0][seed>0].astype(float);hue=float(np.median(hues))
        if np.mean(np.abs(hues-hue)<=5)<.6:return None
        contrast=cv2.subtract(value,cv2.morphologyEx(value,cv2.MORPH_OPEN,np.ones((5,5),np.uint8)))
        ink=(np.abs(hsv[:,:,0].astype(float)-hue)<=5)&(hsv[:,:,1]>=30)&(value>=90)&(contrast>=35)
        # Check whole components before restricting their x extent: clipping
        # a glyph to the gap can turn its edge into a false dot. A disconnected
        # dot may touch either extent without needing an extra empty column.
        count,_,stats,_=cv2.connectedComponentsWithStats(ink.astype(np.uint8),8)
        interior=[(x,y,w,h,n) for x,y,w,h,n in stats[1:] if x>=left and x+w<=right]
        dots=[b for b in interior if 1<=b[4]<=6 and b[2]<=2 and b[3]<=2
              and abs(b[1]+b[3]-baseline)<=1]
        if len(dots)!=1 or len(interior)!=1:return None
        raw='0.'+chars[1]+'초'
        return dict(status='read',raw_text=raw,value=int(chars[1])/10,unit='s',score=min(scores),
                    source='observed',method='calibrated_gray_visible_decimal',reason=None)

    def _read_pieces(self, pieces, text_color, width_delta=0, native=False):
        result = {'status':'unreadable','raw_text':None,'value':None,'unit':None,
                  'score':None,'source':'observed','method':'calibrated_glyphs',
                  'reason':'segmentation','characters':[]}
        if not 2 <= len(pieces) <= 5:
            return result
        if text_color == 'green':
            # A hue-clipped 7 can resemble a short 1. Require full-height
            # numeric strokes relative to the unit; tiny decimal dots exempt.
            unit_height = pieces[-1][0][3]
            if any(3 < box[3] < unit_height * 0.65 for box, _ in pieces[:-1]):
                result['reason'] = 'incomplete_strokes'
                return result
        text = []
        for index, (box, glyph) in enumerate(pieces):
            if box[2] <= 4 and box[3] <= 3 and np.count_nonzero(glyph) <= 8:
                baseline = max((b[1]+b[3] for b,_ in pieces if b[3]>3), default=-100)
                if len(pieces) != 4 or index != 1 or abs(box[1]+box[3]-baseline)>3:
                    result['reason'] = 'invalid_decimal_position'
                    return result
                text.append('.')
                result['characters'].append({'bbox':box,'candidate':'.','score':1.0,'margin':1.0})
                continue
            scores=(self._native_scores(glyph,fitted=native=='fitted') if native else
                    self._glyph_scores(normalize(glyph,width_delta)))
            ranked = sorted(scores.items(), key=lambda p:p[1], reverse=True)
            if not ranked:
                result['reason'] = 'empty_bank'
                return result
            char, score = ranked[0]
            gap = score - (ranked[1][1] if len(ranked)>1 else 0)
            result['characters'].append({'bbox':box,'candidate':char,'score':score,'margin':gap})
            if score < self.threshold or gap < self.margin:
                result['reason'] = 'low_confidence'
                return result
            text.append(char)
        raw = ''.join(text)
        # 07 can be a decimal with a lost dot; never turn it into seven seconds.
        if not re.fullmatch(r'(?:0|00|[1-9][0-9]{0,2}|0\.[0-9])초', raw):
            result['reason'] = 'unsupported_format'
            return result
        numeric = raw[:-1]
        value = float(numeric) if '.' in numeric else int(numeric)
        result.update(status='read',raw_text=raw,value=value,unit='s',
                      score=min(c['score'] for c in result['characters']),reason=None)
        return result

    def read_region(self, frame, box, text_color='yellow'):
        """Bounded vertical alignment search; disagreeing accepted reads abstain."""
        x,y,w,h=box
        results=[]
        for dy in (0,-1,1):
            yy=y+dy
            if x<0 or yy<0 or x+w>frame.shape[1] or yy+h>frame.shape[0]: continue
            result=self.read(frame[yy:yy+h,x:x+w], text_color)
            result['bbox']=[x,yy,w,h]
            results.append(result)
            if dy==0 and result['status']=='read':
                return result
        accepted=[r for r in results if r['status']=='read']
        if not accepted:
            # A fixed-height timer box can clip antialiased top/bottom strokes.
            # Only after the ordinary alignments fail, include bounded margins;
            # preserve the same glyph/decimal rules and require agreement.
            for dy,extra in ((-1,2),(-2,4),(0,2)):
                yy=y+dy
                if x<0 or yy<0 or x+w>frame.shape[1] or yy+h+extra>frame.shape[0]:continue
                result=self.read(frame[yy:yy+h+extra,x:x+w],text_color)
                result['bbox']=[x,yy,w,h+extra]
                if result['status']=='read':accepted.append(result)
        if accepted:
            values={r['raw_text'] for r in accepted}
            if len(values)>1:
                return {'status':'unreadable','raw_text':None,'value':None,'unit':None,
                        'score':None,'reason':'alignment_disagreement','bbox':box,'method':'calibrated_glyphs'}
            return max(accepted,key=lambda r:r['score'])
        return results[0] if results else {
            'status':'unreadable','raw_text':None,'reason':'crop_outside_frame','bbox':box}


"""Maximum cardinality one-to-one icon evaluation; no greedy matching."""
from .core import iou

def match(truth,predictions):
    edges={i:[j for j,p in enumerate(predictions) if p['effect_id']==g['effect_id']
             and p.get('region_id')==g.get('region_id') and iou(g['bbox'],p['bbox'])>=0.5]
           for i,g in enumerate(truth)}
    owners={}
    def augment(i,seen):
        for j in edges[i]:
            if j in seen:
                continue
            seen.add(j)
            if j not in owners or augment(owners[j],seen):
                owners[j]=i
                return True
        return False
    for i in edges:
        augment(i,set())
    return {i:j for j,i in owners.items()}

def score_frame(truth,predictions):
    pairs=match(truth,predictions)
    result={'tp':len(pairs),'fp':len(predictions)-len(pairs),'fn':len(truth)-len(pairs)}
    for field in ('time','stack'):
        eligible=[i for i,g in enumerate(truth) if g.get(field,{}).get('readability')=='readable']
        answered=correct=0
        for i in eligible:
            p=predictions[pairs[i]].get(field,{}) if i in pairs else {}
            value=p.get('raw_text')
            if value is not None:
                answered+=1
                correct+=value==truth[i][field]['raw_text']
        result[field]={'total':len(eligible),'answered':answered,'correct':correct}
    return result

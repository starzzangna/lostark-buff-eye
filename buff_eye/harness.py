"""Offline CLI: validate, replay, evaluate and compare."""
import argparse
import html
import json
import platform
import sys
import time
from pathlib import Path
import cv2
import numpy as np
from .core import Detector,read_json,validate_manifest,safe_path,image_read,digest,box_valid
from .evaluation import score_frame,match

def save(path,obj):
    Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')

def jsonlines(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8-sig').splitlines() if line.strip()]

def validate_labels(path,manifest):
    assets={a['asset_id']:a for a in manifest['assets']}
    rows=jsonlines(path)
    seen=set()
    for row in rows:
        if row['frame_id'] in seen:
            raise ValueError('Duplicate label frame')
        seen.add(row['frame_id'])
        a=assets[row['asset_id']]
        if row['frame_index']!=0 or row['timestamp_us']!=0:
            raise ValueError('Still images require index and timestamp zero')
        regions={r['region_id']:r for r in row['regions']}
        if len(regions)!=len(row['regions']):
            raise ValueError('Duplicate region ID')
        for r in row['regions']:
            if not box_valid(r['bbox'],a['width'],a['height']):
                raise ValueError('Invalid region bbox')
        for g in row['instances']:
            if g['region_id'] not in regions or not box_valid(g['bbox'],a['width'],a['height']):
                raise ValueError('Invalid instance region or bbox')
            rx,ry,rw,rh=regions[g['region_id']]['bbox']
            gx,gy,gw,gh=g['bbox']
            if gx<rx or gy<ry or gx+gw>rx+rw or gy+gh>ry+rh:
                raise ValueError('Instance outside its ROI')
            for field in ('time','stack'):
                value=g[field]
                if value.get('readability') not in {'readable','unreadable','not_shown','occluded'}:
                    raise ValueError('Invalid text readability')
                if value['readability']=='readable' and not isinstance(value.get('raw_text'),str):
                    raise ValueError('Readable text requires a ground-truth string')
                if value.get('bbox') is not None and not box_valid(value['bbox'],a['width'],a['height']):
                    raise ValueError('Invalid text bbox')
    return rows

def replay(args):
    manifest=validate_manifest(args.manifest)
    detector=Detector(args.config)
    out=Path(args.out)
    out.mkdir(parents=True,exist_ok=False)
    config=detector.config
    records=[]
    cards=[]
    for a in manifest['assets']:
        if args.exclude_glyph_source and config.get('text'):
            from .text import TimeReader
            detector.time_reader=TimeReader(safe_path(Path(args.config).parent,config['text']['bank_path']),exclude_asset=a['asset_id'])
        frame=image_read(safe_path(Path(args.manifest).parent,a['path']))
        scale=a['buff_scale_pct'] if a['buff_scale_pct'] is not None else args.assume_scale
        if scale is None:
            raise ValueError(f"Unknown buff scale for {a['asset_id']}; pass --assume-scale after visual calibration")
        started=time.perf_counter()
        detections=detector.detect(frame,scale,search_mode=args.mode)
        elapsed=(time.perf_counter()-started)*1000
        record=dict(frame_id=a['asset_id']+':0',asset_id=a['asset_id'],timestamp_us=0,
                    buff_scale_pct=scale,buff_scale_provenance='manifest' if a['buff_scale_pct'] is not None else 'cli_assumption',split=a['split'],detections=detections,
                    processing_ms=elapsed,output_state=[])
        records.append(record)
        x,y,w,h=config['region']['bbox']
        preview=frame[y:y+h,x:x+w].copy()
        for d in detections:
            dx,dy,dw,dh=d['bbox']
            cv2.rectangle(preview,(dx-x,dy-y),(dx-x+dw,dy-y+dh),(0,255,0),1)
            cv2.putText(preview,d['effect_id'].split('.')[1],(dx-x,dy-y-3),cv2.FONT_HERSHEY_SIMPLEX,0.3,(255,255,255),1)
        filename=a['asset_id']+'.png'
        cv2.imencode('.png',preview)[1].tofile(out/filename)
        detail=html.escape(json.dumps(detections,ensure_ascii=False,indent=2))
        crops=[]
        for index,d in enumerate(detections):
            dx,dy,dw,dh=d['bbox']
            patch=frame[max(0,dy-20):min(frame.shape[0],dy+dh+20),dx:dx+dw]
            cropname=f'{a["asset_id"]}-icon-{index}.png'
            cv2.imencode('.png',patch)[1].tofile(out/cropname)
            reading=html.escape(d['time'].get('raw_text') or '?')
            reason=html.escape(d['time'].get('reason') or d['time']['status'])
            crops.append(f'<figure><img src="{cropname}" style="height:180px"><figcaption>{html.escape(d["effect_id"])}<br>score {d["icon_score"]:.3f}<br><strong style="font-size:28px">{reading}</strong><br>{reason}</figcaption></figure>')
        cards.append(f'<section><h2>{scale}% · {len(detections)} detections</h2><img src="{filename}"><div style="display:flex;flex-wrap:wrap">'+''.join(crops)+f'</div><details><summary>Recognition details / stack OCR pending</summary><pre>{detail}</pre></details></section>')
    (out/'predictions.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records),encoding='utf-8')
    source_hashes={p.name:digest(p) for p in Path(__file__).parent.glob('*.py')}
    save(out/'run.json',dict(schema_version=1,mode=args.mode+'-stills',status='CALIBRATION_ONLY',
        manifest=manifest,manifest_sha256=digest(args.manifest),config=config,config_sha256=digest(args.config),
        template_hashes={t['path']:digest(safe_path(Path(args.config).parent,t['path'])) for t in config['templates']},
        glyph_bank_sha256=digest(safe_path(Path(args.config).parent,config['text']['bank_path'])) if config.get('text') else None,
        exclude_glyph_source=args.exclude_glyph_source,
        assumed_scale=args.assume_scale,
        source_hashes=source_hashes,python=sys.version,opencv=cv2.__version__,numpy=np.__version__,os=platform.platform(),
        command=sys.argv,limitations=['Calibrated integer and 0.x-second OCR only' if config.get('text') else 'No OCR','No stack OCR','No live capture','No temporal evaluation','Not independent test data']))
    (out/'preview.html').write_text('<!doctype html><meta charset="utf-8"><title>Buff Eye calibration</title><style>body{background:#171b24;color:#eee;font:16px sans-serif;margin:30px}section{padding:15px;border:1px solid #456;margin-bottom:20px}img{max-width:100%;image-rendering:pixelated}pre{white-space:pre-wrap}</style><h1>Calibration preview — not an accuracy claim</h1>'+''.join(cards),encoding='utf-8')
    print(f'Replayed {len(records)} images: {out}')

def evaluate(args):
    run=read_json(Path(args.run)/'run.json')
    labels=validate_labels(args.labels,run['manifest'])
    predictions=jsonlines(Path(args.run)/'predictions.jsonl')
    pred={p['frame_id']:p for p in predictions}
    if len(pred)!=len(predictions):
        raise ValueError('Duplicate prediction frame')
    expected_ids={a['asset_id']+':0' for a in run['manifest']['assets']}
    if set(pred)-expected_ids:
        raise ValueError('Predictions contain frames outside manifest')
    totals=dict(tp=0,fp=0,fn=0)
    breakdown=[]
    failures=[]
    for row in labels:
        if row.get('review_status') not in ('self_reviewed','reviewed'):
            continue
        valid={r['region_id'] for r in row['regions'] if r['visibility']=='valid' and r['annotation_complete']}
        if not valid or row['capture_status']!='valid':
            continue
        scope=row['tracked_effect_ids']
        truth=[g for g in row['instances'] if g['region_id'] in valid and g['effect_id'] in scope and g['identity_readability']=='readable']
        found=[p for p in pred.get(row['frame_id'],{}).get('detections',[]) if p['region_id'] in valid and p['effect_id'] in scope]
        result=score_frame(truth,found)
        pairs=match(truth,found)
        for i,g in enumerate(truth):
            if i not in pairs:
                failures.append(dict(frame_id=row['frame_id'],effect_id=g['effect_id'],type='ICON_FN',expected=g['bbox'],actual=None))
            for field in ('time','stack'):
                if g.get(field,{}).get('readability')!='readable': continue
                observed=found[pairs[i]].get(field,{}) if i in pairs else {}
                actual=observed.get('raw_text')
                if actual!=g[field]['raw_text']:
                    failures.append(dict(frame_id=row['frame_id'],effect_id=g['effect_id'],
                        type=field.upper()+('_ABSTAIN' if actual is None else '_WRONG'),
                        expected=g[field]['raw_text'],actual=actual,reason=observed.get('reason','icon_missing')))
        for j,p in enumerate(found):
            if j not in pairs.values():
                failures.append(dict(frame_id=row['frame_id'],effect_id=p['effect_id'],type='ICON_FP',expected=None,actual=p['bbox']))
        for key in totals:
            totals[key]+=result[key]
        breakdown.append(dict(frame_id=row['frame_id'],**result))
    tp,fp,fn=totals.values()
    numeric={}
    for field in ('time','stack'):
        counts={k:sum(r[field][k] for r in breakdown) for k in ('total','answered','correct')}
        n,answered,correct=counts['total'],counts['answered'],counts['correct']
        numeric[field]={**counts,'overall_accuracy':correct/n if n else None,
            'conditional_accuracy':correct/answered if answered else None,
            'abstention_rate':(n-answered)/n if n else None,'implementation_status':'CALIBRATED_PROTOTYPE' if field=='time' and run['config'].get('text') else 'NOT_IMPLEMENTED'}
    metrics=dict(**totals,precision=tp/(tp+fp) if tp+fp else None,recall=tp/(tp+fn) if tp+fn else None,
        numeric=numeric,
        evaluated_frames=len(breakdown),breakdown=breakdown,status='NOT_EVALUATED',
        reason='Calibration-only still-image check; independent sessions and temporal tests required',
        labels_sha256=digest(args.labels))
    save(Path(args.run)/'metrics.json',metrics)
    (Path(args.run)/'failures.jsonl').write_text(''.join(json.dumps(dict(failure_id=f'F{i+1:04}',**f),ensure_ascii=False)+'\n' for i,f in enumerate(failures)),encoding='utf-8')
    (Path(args.run)/'labels.snapshot.jsonl').write_bytes(Path(args.labels).read_bytes())
    table=''.join('<tr>'+''.join('<td>'+html.escape(str(f.get(k,'')))+'</td>' for k in ('frame_id','effect_id','type','expected','actual','reason'))+'</tr>' for f in failures)
    (Path(args.run)/'failures.html').write_text('<!doctype html><meta charset="utf-8"><title>Recognition failures</title><style>body{font:16px sans-serif;background:#18202c;color:white}td,th{padding:12px;border:1px solid #567}</style><h1>Failure review</h1><a href="preview.html">Source and recognition preview</a><table><tr><th>Frame</th><th>Effect</th><th>Type</th><th>Expected</th><th>Actual</th><th>Reason</th></tr>'+table+'</table>',encoding='utf-8')
    (Path(args.run)/'report.md').write_text('# Calibration check\n\nRelease status: NOT_EVALUATED\n\n```json\n'+json.dumps(metrics,indent=2)+'\n```\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in metrics.items() if k!='breakdown'},indent=2))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('validate'); p.add_argument('--manifest',required=True); p.add_argument('--labels')
    p=sub.add_parser('replay'); p.add_argument('--manifest',required=True); p.add_argument('--config',required=True); p.add_argument('--out',required=True); p.add_argument('--mode',choices=['exhaustive','adjacent','native','slots'],default='exhaustive'); p.add_argument('--exclude-glyph-source',action='store_true'); p.add_argument('--assume-scale',type=int,choices=[80,90,100,110,120])
    p=sub.add_parser('evaluate'); p.add_argument('--run',required=True); p.add_argument('--labels',required=True)
    p=sub.add_parser('compare'); p.add_argument('--baseline',required=True); p.add_argument('--candidate',required=True)
    args=parser.parse_args()
    try:
        if args.command=='validate':
            m=validate_manifest(args.manifest)
            if args.labels: validate_labels(args.labels,m)
            print(f'VALID: {len(m["assets"])} assets')
        elif args.command=='replay': replay(args)
        elif args.command=='evaluate': evaluate(args)
        else:
            a,b=[read_json(Path(p)/'run.json') for p in (args.baseline,args.candidate)]
            if a['manifest_sha256']!=b['manifest_sha256']: raise ValueError('Different datasets cannot be compared')
            am,bm=[read_json(Path(p)/'metrics.json') for p in (args.baseline,args.candidate)]
            if am['labels_sha256']!=bm['labels_sha256']: raise ValueError('Different labels cannot be compared')
            print(json.dumps({k:{'baseline':am[k],'candidate':bm[k]} for k in ('tp','fp','fn','precision','recall','numeric')},indent=2))
    except (ValueError,KeyError,OSError) as error:
        parser.exit(2,f'ERROR: {error}\n')

if __name__=='__main__': main()

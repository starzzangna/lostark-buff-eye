from pathlib import Path
import json

root=Path(SPECPATH)
base=root/'datasets/calibration-3440'
config=json.loads((base/'detector-ocr-v6.json').read_text(encoding='utf-8'))
paths={'detector-ocr-v6.json','target-glyphs.json',config['text']['bank_path'],config['stack']['bank_path']}
paths.add('glyph-gray-v1.json')
paths.update(('target-native/dark.png','target-native/light_dark.png'))
paths.update(t['path'] for t in config['templates'])
paths.update(config['overlay_icons'].values())
paths.add(config['merge_marker_path'])
paths.update('skills/'+p for p in ('battle_item_01_47.png','battle_item_01_87.png','buff_60.png','buff_67.png','buff_498.png'))
datas=[]
for name in sorted(paths):
    source=(base/name).resolve()
    if not source.is_relative_to(base.resolve()) or not source.is_file():
        raise ValueError('Missing or invalid runtime asset: '+name)
    datas.append((str(source),str(Path('datasets/calibration-3440')/Path(name).parent)))
a=Analysis([str(root/'scripts/packaged_entry.py')],pathex=[str(root),str(root/'.deps')],
           binaries=[],datas=datas,hiddenimports=[],hookspath=[],hooksconfig={},runtime_hooks=[],
           excludes=['tkinter','matplotlib','pandas','scipy','torch','IPython'],noarchive=False)
pyz=PYZ(a.pure)
# Qt uses Windows' ICU API. A developer PATH may contain Poppler's incompatible
# icuuc.dll; never shadow the system DLL with that unrelated distribution.
a.binaries=[entry for entry in a.binaries if Path(entry[0]).name.lower() not in ('icuuc.dll','icudt78.dll')]
exe=EXE(pyz,a.scripts,a.binaries,a.datas,[],name='LostArkBuffEye',debug=False,
        bootloader_ignore_signals=False,strip=False,upx=False,console=False)

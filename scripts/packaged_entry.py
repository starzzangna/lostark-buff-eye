"""Desktop entry point and read-only bundle validation."""
import json
import sys
from pathlib import Path


def self_check(destination):
    import cv2
    import numpy as np
    from PySide6.QtWidgets import QApplication, QLabel
    from buff_eye.app import DEFAULT_CONFIG
    from buff_eye.support_buffs import SupportDetector as Detector
    from buff_eye.target_buffs import TargetDetector, ASSETS
    from buff_eye.settings import user_data_directory
    support=Detector(DEFAULT_CONFIG)
    target=TargetDetector(DEFAULT_CONFIG.parent/'skills',DEFAULT_CONFIG.parent/'glyph-bank-v5.json')
    support_found=[]
    from buff_eye.effect_groups import MERGED_COMMON
    for key in MERGED_COMMON:
        frame=np.zeros((80,100,3),np.uint8)
        frame[12:31,12:31]=cv2.resize(support.overlay_icons[key],(19,19),interpolation=cv2.INTER_AREA)
        support.reset_row_tracking()
        assert key in [d['effect_id'] for d in support.detect_region(frame)],key
        support_found.append(key)
    found=[]
    for key in ASSETS:
        frame=np.zeros((100,100,3),np.uint8)
        frame[22:51,22:51]=cv2.resize(target.overlay_icons[key],(29,29),interpolation=cv2.INTER_AREA)
        target.reset_row_tracking()
        assert [d['effect_id'] for d in target.detect_region(frame)]==[key],key
        found.append(key)
    app=QApplication.instance() or QApplication([])
    widget=QLabel('LostArk Buff Eye')
    widget.resize(200,50)
    assert not widget.grab().isNull()
    widget.close()
    app.processEvents()
    report={'frozen':bool(getattr(sys,'frozen',False)),
            'support_templates':len(support.templates),'support_assets':support_found,'target_assets':found,
            'qt_render':True,'settings_directory':str(user_data_directory()),
            'settings_written':False,'executable':sys.executable}
    Path(destination).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__':
    if len(sys.argv)==3 and sys.argv[1]=='--self-check':
        self_check(sys.argv[2])
    else:
        from buff_eye.app import main
        raise SystemExit(main())

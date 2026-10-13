"""Run with run.ps1; build dependencies stay separate from app dependencies."""
import sys
from pathlib import Path
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'.build-deps'))
from PyInstaller.__main__ import run
run(['--noconfirm','--distpath',str(root/'dist'),
     '--workpath',str(root/'build/pyinstaller'),str(root/'LostArkBuffEye.spec')])

# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['gui.py'],
    pathex=[],
    binaries=[],
    datas=[('C:\\Users\\User\\PycharmProjects\\face\\video-face-swap\\models\\GFPGANv1.4.pth', 'models'), ('C:\\Users\\User\\PycharmProjects\\face\\video-face-swap\\models\\inswapper_128.onnx', 'models'), ('C:/Users/User/PycharmProjects/face/Lib/site-packages/gfpgan', 'gfpgan'), ('C:/Users/User/PycharmProjects/face/Lib/site-packages/basicsr', 'basicsr'), ('C:\\Users\\User\\PycharmProjects\\face\\video-face-swap', '.')],
    hiddenimports=['cv2', 'insightface', 'moviepy', 'numpy', 'gfpgan', 'tqdm', 'torch', 'torchvision', 'PySide6', 'torchvision.transforms.functional_tensor'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='gui',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

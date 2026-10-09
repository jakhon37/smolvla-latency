"""Time offscreen MuJoCo rendering with one GL backend: python mujoco_render_check.py egl|osmesa."""

import os
import sys
import time

backend = sys.argv[1]
os.environ["MUJOCO_GL"] = backend
os.environ["PYOPENGL_PLATFORM"] = backend

try:
    import mujoco

    model = mujoco.MjModel.from_xml_string(
        "<mujoco><worldbody><light pos='0 0 3'/><geom type='plane' size='1 1 .1'/>"
        "<body pos='0 0 .3'><freejoint/><geom type='box' size='.05 .05 .05' rgba='.8 .2 .2 1'/></body>"
        "</worldbody></mujoco>"
    )
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, 256, 256)
    n = 200
    t = time.perf_counter()
    for _ in range(n):
        mujoco.mj_step(model, data)
        renderer.update_scene(data)
        pixels = renderer.render()
    rate = n / (time.perf_counter() - t)
    print(f"{backend}: OK {rate:.0f} renders/s at 256x256 (mean pixel {pixels.mean():.1f})")
except Exception as e:
    print(f"{backend}: FAILED {type(e).__name__}: {str(e).splitlines()[0][:160]}")

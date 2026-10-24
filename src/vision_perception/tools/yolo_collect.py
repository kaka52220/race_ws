#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
yolo_collect.py —— 采集 YOLO 训练数据（网页推流 + 按目标分类存图）

和 lap_collect.py 的区别：
    lap_collect 是「按圈存图」，用来采车道线数据。
    这个按【目标类别】存图，用来训练红绿灯 / 挡板的检测模型。

两个类别（按比赛规则）：
    red_light  红绿灯（一张红纸，放在人行道旁）
    barrier    挡板（PP 中空挡板，标记点 3-4 之间随机放 1~2 个）
    negative   负样本（画面里没有目标的其他位置）

为什么需要 negative：
    防止模型把「地上任何红色/蓝色的东西」都当目标。YOLO 必须有背景图。

三个关键设计：
    1. 固定曝光 —— 摄像头断电重插会复位成 brightness=50、blc=121，
       画面亮度 242、82% 过曝，采出来的红纸和蓝板都发白，标了也没用。
    2. 去重 —— 4fps 存图时相邻两张只差 5cm，95% 是重复的。
       和上一张差异太小就跳过，直接省掉一半标注量。
    3. manifest.csv —— 记录每张图的目标、亮度、过曝、场景标签，
       采完能查出哪批有问题。

用法：
    bash ~/ka.sh                                  # 先停掉所有 ROS 节点，摄像头是独占的
    python3 ~/yolo_collect.py                     # 默认存到 ~/data/yolo_<时间>
    python3 ~/yolo_collect.py --scene 挡板-左前   # 给这一批打标签，写进 manifest
    python3 ~/yolo_collect.py --fps 3 --dedup 3.0

然后浏览器打开 http://<香橙派IP>:8080
"""
import argparse
import csv
import glob
import os
import re
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2

W, H = 1280, 720

# 类别目录名 → 网页上显示的名字
TARGETS = {
    "red_light": "红绿灯",
    "barrier": "挡板",
    "negative": "负样本",
}

# 固定曝光：和 cam_tune.py --fix / pos_sweep.py 用同一套值。
# 摄像头断电重插后会复位成 brightness=50 + blc=121（亮度 242、82% 过曝），
# 这个状态下红纸和蓝板都发白，标出来的框是废的。
CAM_CTRLS = (
    "backlight_compensation=0",
    "brightness=0",
    "auto_exposure=1",
    "exposure_time_absolute=15",
)


def fix_camera(dev=None):
    """把摄像头固定到已知良好的曝光参数。失败不致命（可能已经是对的）。"""
    dev = dev or detect_camera_index()
    for c in CAM_CTRLS:
        subprocess.run(["v4l2-ctl", "-d", dev, "--set-ctrl=%s" % c],
                       capture_output=True)
    try:
        out = subprocess.check_output(
            ["v4l2-ctl", "-d", dev, "--get-ctrl=exposure_time_absolute"],
            text=True, stderr=subprocess.DEVNULL).strip()
        print("  曝光设置: %s" % out)
    except Exception:
        pass


def detect_camera_index():
    try:
        out = subprocess.check_output(["v4l2-ctl", "--list-devices"],
                                      text=True, stderr=subprocess.DEVNULL)
        for line in out.splitlines():
            m = re.match(r"\s*(/dev/video\d+)", line)
            if m:
                return m.group(1)
    except Exception:
        pass
    for p in sorted(glob.glob("/dev/video*")):
        if p[-1].isdigit():
            return p
    return "/dev/video0"


# ═══════════════════════════════════════════════════════════
#  摄像头
# ═══════════════════════════════════════════════════════════
class Camera:
    def __init__(self, device):
        self.device = device
        self.cap = None
        self.frame = None
        self.seq = 0
        self.lock = threading.Lock()
        self.running = True
        self.fps = 0.0
        self.bright = 0.0
        self.blown = 0.0
        self.error = ""
        self._open()

    def _open(self):
        idx = int(re.sub(r"\D", "", self.device) or 0)
        cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            self.error = "打不开摄像头 %s" % self.device
            return
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
        self.cap = cap

    def start(self):
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        cnt, t0 = 0, time.time()
        while self.running:
            if self.cap is None:
                time.sleep(0.5)
                self._open()
                continue
            ok, f = self.cap.read()
            if not ok:
                time.sleep(0.05)
                continue
            with self.lock:
                self.frame = f
                self.seq += 1
            cnt += 1
            el = time.time() - t0
            if el >= 1.0:
                self.fps = cnt / el
                cnt, t0 = 0, time.time()
                g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
                self.bright = float(g.mean())
                self.blown = float(100.0 * (g > 240).mean())

    def get(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy(), self.seq


# ═══════════════════════════════════════════════════════════
#  录制（按目标分类 + 去重 + 清单）
# ═══════════════════════════════════════════════════════════
class Recorder:
    def __init__(self, outroot, save_fps, dedup_thr, scene):
        self.outroot = outroot
        self.scene = scene
        self.interval = 1.0 / max(save_fps, 0.1)
        self.dedup_thr = dedup_thr
        self.target = None            # None = 未录制
        self.counts = {k: 0 for k in TARGETS}
        self.last_save = 0.0
        self.prev_small = None
        self.skipped = 0              # 被去重跳过的张数，用来判断阈值合不合适
        self.burst_left = 0
        self.lock = threading.Lock()
        for k in TARGETS:
            os.makedirs(os.path.join(outroot, k), exist_ok=True)
        self.manifest = os.path.join(outroot, "manifest.csv")
        if not os.path.exists(self.manifest):
            with open(self.manifest, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(
                    ["target", "file", "scene", "time", "bright", "blown", "note"])

    def start(self, target):
        if target not in TARGETS:
            return False, "未知目标 %s" % target
        with self.lock:
            self.target = target
            self.last_save = 0.0
            self.prev_small = None    # 换目标后重新开始比，避免跨目标误判重复
        return True, "开始采集「%s」" % TARGETS[target]

    def stop(self):
        with self.lock:
            old = self.target
            self.target = None
            self.burst_left = 0
        return old

    def burst(self, n=5):
        """带内连拍：忽略去重，按间隔连存 n 张（用来在稍微不同的位置各留一张）。"""
        with self.lock:
            self.burst_left = int(n)
        return int(n)

    def maybe_save(self, frame, now):
        with self.lock:
            if self.target is None:
                return
            if now - self.last_save < self.interval:
                return
            burst = self.burst_left > 0
            if burst:
                self.burst_left -= 1
            target = self.target
            self.last_save = now

        # ── 去重：缩到 160x90 灰度比平均绝对差 ──
        #   4fps 时相邻两张只差约 5cm，差异值通常在 0.5~2；
        #   车真的移动了会有 5 以上。阈值 3.0 相当于「肉眼能看出变了」。
        small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (160, 90))
        if not burst and self.prev_small is not None and self.dedup_thr > 0:
            diff = float(cv2.absdiff(small, self.prev_small).mean())
            if diff < self.dedup_thr:
                with self.lock:
                    self.skipped += 1
                return
        self.prev_small = small

        with self.lock:
            self.counts[target] += 1
            n = self.counts[target]
        p = os.path.join(self.outroot, target, "%04d.jpg" % n)
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if not ok:
            return
        with open(p, "wb") as f:
            f.write(buf.tobytes())

        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        with open(self.manifest, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                target, "%s/%04d.jpg" % (target, n), self.scene,
                time.strftime("%H:%M:%S"), "%.0f" % g.mean(),
                "%.1f" % (100.0 * (g > 240).mean()),
                "burst" if burst else "",
            ])

    def status(self):
        with self.lock:
            return {"target": self.target, "counts": dict(self.counts),
                    "skipped": self.skipped}


# ═══════════════════════════════════════════════════════════
#  HTTP
# ═══════════════════════════════════════════════════════════
PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>YOLO 采集</title>
<style>
 body{margin:0;background:#111;color:#ddd;font-family:system-ui,sans-serif}
 .wrap{display:flex;gap:14px;padding:14px;flex-wrap:wrap}
 img{width:min(72vw,1000px);border-radius:8px;background:#000}
 .panel{min-width:280px;flex:1}
 h2{margin:4px 0 10px;font-size:17px;color:#8fd}
 button{font-size:16px;padding:11px 14px;margin:5px 4px;border:0;border-radius:8px;
        background:#2a6;color:#fff;cursor:pointer}
 button.t2{background:#268}
 button.t3{background:#666}
 button.stop{background:#b33}
 button.snap{background:#a70}
 button.cur{outline:3px solid #ff0}
 .st{font-size:15px;line-height:1.9;margin-top:12px;padding:10px;
     background:#1c1c1c;border-radius:8px}
 .big{font-size:24px;color:#ff0;font-weight:700}
 .muted{color:#888;font-size:13px;line-height:1.7}
</style></head><body>
<div class="wrap">
  <div><img id="v" src="/stream"></div>
  <div class="panel">
    <h2>选目标类别</h2>
    <div id="btns"></div>
    <button class="stop" onclick="act('/api/stop')">■ 停止</button>
    <button class="snap" onclick="act('/api/burst?n=5')">📷 连拍 5 张</button>
    <div class="st">
      当前目标：<span class="big" id="tg">—</span><br>
      本类已存：<span class="big" id="cur">0</span> 张<br>
      去重跳过：<span id="skip">0</span> 张<br>
      各累总计：<span id="all">—</span><br>
      摄像头：<span id="cam">—</span><br>
      亮度/过曝：<span class="big" id="lux">—</span>
    </div>
    <p class="muted">
      <b>采集顺序建议</b><br>
      1. 挡板摆 8 个位置，每个位置从 3 个距离各采一次<br>
      2. 两块挡板同时摆 6 种组合<br>
      3. 红绿灯：0.4/0.8/1.2/2/3/4 m，各 3 个横向位置<br>
      4. 负样本：赛道其他位置各采十几张<br>
      <br>
      <b>每采一批按 <code>Ctrl+C</code> 重开，换 --scene 标签</b><br>
      去重会把「车没动」的重复帧自动跳过，跳过数少说明车动得太慢。
    </p>
  </div>
</div>
<script>
const T = %TARGETS%;
const K = Object.keys(T);
const CLS = ['','t2','t3'];
for(let i=0;i<K.length;i++){
  const b=document.createElement('button');
  b.textContent=T[K[i]]; b.id='b'+K[i]; b.className=CLS[i]||'';
  b.onclick=()=>act('/api/start?target='+K[i]);
  document.getElementById('btns').appendChild(b);
}
function act(u){fetch(u).then(r=>r.json()).then(refresh);}
function refresh(){
  fetch('/api/status').then(r=>r.json()).then(s=>{
    document.getElementById('tg').textContent = s.target? T[s.target] : '未记录';
    document.getElementById('cur').textContent = s.target? s.counts[s.target] : 0;
    document.getElementById('skip').textContent = s.skipped;
    document.getElementById('all').textContent =
      K.map(k=>T[k]+':'+s.counts[k]).join('  ');
    document.getElementById('cam').textContent = s.fps.toFixed(1)+' fps';
    var lx = document.getElementById('lux');
    var bad = (s.bright>170 || (s.bright>0 && s.bright<80) || s.blown>12);
    lx.textContent = (s.bright?s.bright.toFixed(0):'—') + ' / '
                   + (s.blown?s.blown.toFixed(1):'0') + '%' + (s.bright?(bad?'  ⚠ 过曝':'  ✅'):'');
    lx.style.color = bad ? '#f66' : '#7f7';
    K.forEach(k=>{ document.getElementById('b'+k).className = (s.target==k)?'cur':(CLS[K.indexOf(k)]||''); });
  });
}
setInterval(refresh,600); refresh();
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    cam = None
    rec = None

    def log_message(self, *a):
        pass

    def do_GET(self):
        p = self.path.split("?")[0]
        if p == "/":
            b = PAGE.replace("%TARGETS%", _targets_js()).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)
        elif p == "/stream":
            self._stream()
        elif p == "/api/start":
            tg = ""
            if "target=" in self.path:
                tg = self.path.split("target=")[1].split("&")[0]
            ok, msg = self.rec.start(tg)
            self._json({"ok": ok, "msg": msg})
        elif p == "/api/stop":
            old = self.rec.stop()
            self._json({"ok": True, "stopped": old})
        elif p == "/api/burst":
            n = 5
            if "n=" in self.path:
                try:
                    n = int(self.path.split("n=")[1].split("&")[0])
                except ValueError:
                    pass
            self._json({"ok": True, "burst": self.rec.burst(n)})
        elif p == "/api/status":
            s = self.rec.status()
            s["fps"] = self.cam.fps
            s["bright"] = self.cam.bright
            s["blown"] = self.cam.blown
            s["error"] = self.cam.error
            self._json(s)
        else:
            self.send_error(404)

    def _json(self, o):
        import json
        b = json.dumps(o).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def _stream(self):
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()
        try:
            while True:
                f, seq = self.cam.get()
                if f is None:
                    time.sleep(0.05)
                    continue
                self.rec.maybe_save(f, time.time())
                ok, buf = cv2.imencode(".jpg", f, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                if not ok:
                    time.sleep(0.02)
                    continue
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: ")
                self.wfile.write(str(len(buf)).encode())
                self.wfile.write(b"\r\n\r\n")
                self.wfile.write(buf.tobytes())
                self.wfile.write(b"\r\n")
                time.sleep(0.033)
        except (BrokenPipeError, ConnectionResetError):
            pass


def _targets_js():
    """把 TARGETS 转成 JS 对象字面量，避免网页里和 Python 两边各写一份。"""
    items = ", ".join('%s: "%s"' % (k, v) for k, v in TARGETS.items())
    return "{" + items + "}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--fps", type=float, default=4.0, help="存图频率，默认 4")
    ap.add_argument("--dedup", type=float, default=3.0,
                    help="去重阈值（160x90 灰度的平均绝对差），0=关闭，默认 3.0")
    ap.add_argument("--scene", default="", help="本批场景标签，写进 manifest.csv")
    ap.add_argument("--camera", default="")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--no-fix", action="store_true", help="不自动设置曝光")
    a = ap.parse_args()

    dev = a.camera or detect_camera_index()
    print("=" * 60)
    print("YOLO 数据采集")
    print("=" * 60)
    print("摄像头:", dev)
    if not a.no_fix:
        print("固定曝光参数（防止断电重插后过曝）...")
        fix_camera(dev)
    else:
        print("跳过曝光设置（--no-fix）")

    out = a.out or os.path.join(os.path.expanduser("~/data"),
                                "yolo_" + time.strftime("%Y%m%d_%H%M%S"))
    out = os.path.expanduser(out)
    os.makedirs(out, exist_ok=True)

    cam = Camera(dev)
    if cam.error:
        print("错误:", cam.error)
        sys.exit(1)
    cam.start()

    rec = Recorder(out, a.fps, a.dedup, a.scene)
    Handler.cam = cam
    Handler.rec = rec

    print("输出目录:", out)
    print("场景标签:", a.scene or "(未设置，建议用 --scene 标注这一批是什么摆位)")
    print("存图频率: %.1f fps   去重阈值: %.1f" % (a.fps, a.dedup))
    print("类别:", "  ".join("%s(%s)" % (v, k) for k, v in TARGETS.items()))
    print()
    print("浏览器打开:  http://%s:%d" % (_local_ip(), a.port))
    print("按 Ctrl+C 结束")
    print("=" * 60)

    srv = ThreadingHTTPServer(("0.0.0.0", a.port), Handler)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        cam.running = False
        srv.server_close()
        st = rec.status()
        print()
        print("采集结束：", "  ".join("%s %d 张" % (TARGETS[k], v)
                                      for k, v in st["counts"].items()))
        print("去重跳过 %d 张   清单: %s" % (st["skipped"], rec.manifest))
        print("目录:", out)


def _local_ip():
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


if __name__ == "__main__":
    main()

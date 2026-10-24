#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
debug_view.py —— 调试用网页：看拟合线 + 实时看 e/theta/κ + 在线改控制参数

为什么要单独写一个，而不是直接在 vision_node 里加：
    摄像头是独占的，调试脚本再开一次会抢摄像头。
    这个脚本**不碰摄像头**，只订阅视觉节点已经发出来的东西，
    所以不影响帧率、不影响控制，边跑车边看。

订阅三个话题（都是现成的，不用改任何节点）：
    /debug/track_image   sensor_msgs/Image   视觉画的调试图（原图叠加 + 鸟瞰图）
    /track               custom_msgs/TrackInfo   e / theta / kappa / 宽度 / 质量 / 模式 / 丢线
    /cmd_vel             geometry_msgs/Twist    大脑实际发出去的速度和舵角

还能在线改 /brain_node 的参数（网页上填数字回车即可），
底层调的是 ROS 参数服务，和 ros2 param set 等价。

用法：
    # 终端 1：正常发车，但把调试图打开
    ros2 launch bringup start_race.launch.py dry_run:=false publish_debug:=true

    # 终端 2：
    python3 ~/debug_view.py

    浏览器打开 http://<香橙派IP>:8090

⚠ publish_debug:=true 会让视觉每帧多花 10~15ms 画图，但只在 3Hz 时画，
  实测帧率还是 30Hz。如果发现掉帧，把 debug_image_hz 调成 2。
"""
import argparse
import io
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image

from rcl_interfaces.srv import GetParameters, SetParameters
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue

try:
    from custom_msgs.msg import TrackInfo
except ImportError:                       # 消息包没编译时也不至于整个脚本崩
    TrackInfo = None

MODE_TXT = {0: "双边", 1: "仅左", 2: "仅右", 3: "丢线"}

# 网页上能改的参数：名字 → (显示名, 建议范围提示)
TUNABLE = [
    ("max_speed", "最高速 m/s", "0.21"),
    ("min_speed", "最低速 m/s", "0.21"),
    ("a_lat", "侧向加速度上限", "0.8"),
    ("stanley_k", "e 增益 stanley_k", "0.7"),
    ("kappa_gain", "曲率前馈权重", "0.5"),
    ("theta_gain", "航向角权重", "0.0"),
    ("e_clip_m", "e 夹取 m", "0.35"),
    ("v_stanley_min", "stanley 分母下限", "0.50"),
    ("steer_max_deg", "最大舵角 度", "25.0"),
    ("steer_trim_deg", "直线微调 度", "0.0"),
    ("lost_speed", "丢线爬行 m/s", "0.07"),
    ("speed_scale", "速度倍率", "1.0"),
]


# ═══════════════════════════════════════════════════════════
#  ROS 节点
# ═══════════════════════════════════════════════════════════
class DebugView(Node):
    def __init__(self, target):
        super().__init__("debug_view")
        self.target = target            # 被调参数的节点名，默认 /brain_node
        self.frame = None               # 最新调试图（BGR）
        self.frame_seq = 0
        self.track = {}
        self.cmd = {"v": 0.0, "steer_deg": 0.0, "t": 0.0}
        self.params = {}
        self.hz_img = 0.0
        self.hz_track = 0.0
        self._lock = threading.Lock()
        self._set_queue = []            # 网页上的改参数请求，由 ROS 线程执行
        self._img_cnt = [0, time.time()]
        self._trk_cnt = [0, time.time()]

        qos = QoSProfile(depth=2,
                         reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST)
        self.create_subscription(Image, "/debug/track_image", self.on_img, qos)
        if TrackInfo is not None:
            self.create_subscription(TrackInfo, "/track", self.on_track, 10)
        self.create_subscription(Twist, "/cmd_vel", self.on_cmd, 10)

        self.cli_get = self.create_client(
            GetParameters, "%s/get_parameters" % self.target)
        self.cli_set = self.create_client(
            SetParameters, "%s/set_parameters" % self.target)

        self.create_timer(1.0, self.tick_params)
        self.create_timer(0.2, self.tick_rates)
        self.get_logger().info(
            "调试页已启动，等 /debug/track_image ... "
            "如果一直没画面，检查 launch 是不是 publish_debug:=true")

    # ── 话题回调 ──
    def on_img(self, msg):
        try:
            buf = np.frombuffer(msg.data, dtype=np.uint8)
            if msg.encoding in ("bgr8", "8UC3"):
                img = buf.reshape(msg.height, msg.width, 3)
            elif msg.encoding == "rgb8":
                img = buf.reshape(msg.height, msg.width, 3)[:, :, ::-1]
            elif msg.encoding in ("mono8", "8UC1"):
                img = cv2.cvtColor(buf.reshape(msg.height, msg.width),
                                   cv2.COLOR_GRAY2BGR)
            else:
                return
            with self._lock:
                self.frame = np.ascontiguousarray(img)
                self.frame_seq += 1
            self._img_cnt[0] += 1
        except Exception as ex:                       # 坏帧不要让节点挂掉
            self.get_logger().warn(f"解码调试图失败: {ex}", once=True)

    def on_track(self, msg):
        with self._lock:
            self.track = {
                "e_cm": float(msg.lateral_error) * 100.0,
                "theta_deg": float(np.degrees(msg.heading_error)),
                "kappa": float(msg.curvature),
                "width_m": float(msg.lane_width),
                "quality": float(msg.quality),
                "mode": int(msg.mode),
                "mode_txt": MODE_TXT.get(int(msg.mode), "?"),
                "is_lost": bool(msg.is_lost),
                "reason": msg.reject_reason,
                "n_pts": int(msg.n_points),
            }
        self._trk_cnt[0] += 1

    def on_cmd(self, msg):
        with self._lock:
            self.cmd = {
                "v": float(msg.linear.x),
                "steer_deg": float(np.degrees(msg.angular.z)),
                "t": time.time(),
            }

    # ── 定时任务 ──
    def tick_rates(self):
        now = time.time()
        with self._lock:
            if now - self._img_cnt[1] >= 1.0:
                self.hz_img = self._img_cnt[0] / (now - self._img_cnt[1])
                self._img_cnt[0] = 0
                self._img_cnt[1] = now
            if now - self._trk_cnt[1] >= 1.0:
                self.hz_track = self._trk_cnt[0] / (now - self._trk_cnt[1])
                self._trk_cnt[0] = 0
                self._trk_cnt[1] = now

    def tick_params(self):
        # 1) 处理网页上的改参数请求
        with self._lock:
            pending = self._set_queue[:]
            self._set_queue.clear()
        for name, value in pending:
            self.do_set(name, value)
        # 2) 读回当前值
        self.do_get()

    def do_get(self):
        if not self.cli_get.service_is_ready():
            if not self.cli_get.wait_for_service(timeout_sec=0.05):
                return
        req = GetParameters.Request()
        req.names = [n for n, _, _ in TUNABLE]
        fut = self.cli_get.call_async(req)
        fut.add_done_callback(self._on_get)

    def _on_get(self, fut):
        try:
            res = fut.result()
        except Exception:
            return
        vals = {}
        for name, v in zip([n for n, _, _ in TUNABLE], res.values):
            if v.type == ParameterType.PARAMETER_DOUBLE:
                vals[name] = round(float(v.double_value), 4)
            elif v.type == ParameterType.PARAMETER_INTEGER:
                vals[name] = int(v.integer_value)
            elif v.type == ParameterType.PARAMETER_NOT_SET:
                vals[name] = None
        with self._lock:
            self.params = vals

    def do_set(self, name, value):
        if not self.cli_set.service_is_ready():
            if not self.cli_set.wait_for_service(timeout_sec=0.2):
                self.get_logger().warn(f"改参数失败：{self.target} 不在线")
                return
        # 必须用 Parameter(name=..., value=...) 包一层，只传 ParameterValue 会被忽略
        req = SetParameters.Request()
        req.parameters = [Parameter(
            name=name,
            value=ParameterValue(type=ParameterType.PARAMETER_DOUBLE,
                                 double_value=float(value)))]
        fut = self.cli_set.call_async(req)
        fut.add_done_callback(
            lambda f, n=name, v=value: self._on_set(f, n, v))

    def _on_set(self, fut, name, value):
        try:
            ok = fut.result().results[0].successful
        except Exception:
            ok = False
        self.get_logger().info(
            f"改参数 {name} = {value}  {'成功' if ok else '失败'}")

    # ── 给 HTTP 线程用 ──
    def request_set(self, name, value):
        with self._lock:
            self._set_queue.append((name, float(value)))

    def snapshot(self):
        with self._lock:
            d = {
                "track": dict(self.track),
                "cmd": dict(self.cmd),
                "params": dict(self.params),
                "hz_img": round(self.hz_img, 1),
                "hz_track": round(self.hz_track, 1),
                "has_image": self.frame is not None,
            }
            f = None if self.frame is None else self.frame.copy()
        return d, f


# ═══════════════════════════════════════════════════════════
#  网页
# ═══════════════════════════════════════════════════════════
PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>调参调试页</title>
<style>
 body{margin:0;background:#0e0e10;color:#ddd;font-family:system-ui,sans-serif;font-size:14px}
 .wrap{display:flex;gap:12px;padding:12px;align-items:flex-start}
 .left{flex:1 1 auto;min-width:520px}
 .right{width:330px;flex:0 0 auto}
 img{width:100%;border-radius:8px;background:#000;display:block}
 .card{background:#17171a;border-radius:8px;padding:10px 12px;margin-bottom:10px}
 h2{margin:0 0 8px;font-size:13px;color:#7fd;font-weight:700;letter-spacing:.5px}
 table{width:100%;border-collapse:collapse}
 td{padding:3px 0;vertical-align:middle}
 td.k{color:#8a8f98;width:46%}
 td.v{font-family:ui-monospace,Consolas,monospace;font-size:15px;color:#fff;text-align:right}
 .big{font-size:20px;font-weight:700}
 .ok{color:#5f5}.warn{color:#fc4}.bad{color:#f55}
 input{width:92px;background:#0b0b0d;border:1px solid #333;color:#fff;
       border-radius:5px;padding:4px 6px;font-family:ui-monospace,monospace;font-size:13px}
 input:focus{outline:1px solid #2a6}
 .prow{display:flex;justify-content:space-between;align-items:center;padding:3px 0}
 .prow span{color:#9aa0a8;font-size:12.5px}
 button{background:#2a6;color:#fff;border:0;border-radius:6px;padding:6px 10px;
        cursor:pointer;font-size:13px;margin-top:6px}
 .muted{color:#666;font-size:11.5px;margin-top:6px;line-height:1.5}
</style></head><body>
<div class="wrap">
  <div class="left">
    <img id="v" src="/stream">
    <div class="card" style="margin-top:10px">
      <h2>画面</h2>
      <table>
        <tr><td class="k">调试图帧率</td><td class="v" id="hzimg">—</td></tr>
        <tr><td class="k">/track 帧率</td><td class="v" id="hztrk">—</td></tr>
      </table>
      <div class="muted">调试图默认 3 Hz（画图要 10~15 ms，画太勤会拖慢视觉）。
        没画面就检查 launch 是否 <code>publish_debug:=true</code>。</div>
    </div>
  </div>
  <div class="right">
    <div class="card">
      <h2>视觉 /track</h2>
      <table>
        <tr><td class="k">e（横向偏差）</td><td class="v big" id="e">—</td></tr>
        <tr><td class="k">theta（航向）</td><td class="v" id="th">—</td></tr>
        <tr><td class="k">kappa（曲率）</td><td class="v" id="k">—</td></tr>
        <tr><td class="k">车道宽</td><td class="v" id="w">—</td></tr>
        <tr><td class="k">质量</td><td class="v" id="q">—</td></tr>
        <tr><td class="k">模式</td><td class="v" id="mode">—</td></tr>
        <tr><td class="k">丢线</td><td class="v" id="lost">—</td></tr>
      </table>
      <div class="muted" id="reason"></div>
    </div>
    <div class="card">
      <h2>控制 /cmd_vel</h2>
      <table>
        <tr><td class="k">速度</td><td class="v big" id="v">—</td></tr>
        <tr><td class="k">舵角</td><td class="v big" id="steer">—</td></tr>
        <tr><td class="k">指令时间</td><td class="v" id="cmdt">—</td></tr>
      </table>
    </div>
    <div class="card">
      <h2>参数（回车即生效）</h2>
      <div id="params"></div>
      <div class="muted">改的是 /brain_node 的 ROS 参数，和
        <code>ros2 param set</code> 等价。一次只改一个，跑一段再看。</div>
    </div>
  </div>
</div>
<script>
const TUNABLE = %TUNABLE%;
let built = false;
function fmt(x, d){ return (x===null||x===undefined)? '—' : Number(x).toFixed(d); }
function refresh(){
  fetch('/api/status').then(r=>r.json()).then(s=>{
    const t = s.track || {}, c = s.cmd || {};
    document.getElementById('hzimg').textContent = s.hz_img.toFixed(1)+' Hz';
    document.getElementById('hztrk').textContent = s.hz_track.toFixed(1)+' Hz';
    if(t.e_cm !== undefined){
      const e = document.getElementById('e');
      e.textContent = (t.e_cm>=0?'+':'') + t.e_cm.toFixed(1) + ' cm';
      e.className = 'v big ' + (Math.abs(t.e_cm) < 8 ? 'ok' : (Math.abs(t.e_cm) < 18 ? 'warn' : 'bad'));
      document.getElementById('th').textContent = (t.theta_deg>=0?'+':'') + t.theta_deg.toFixed(1) + '°';
      document.getElementById('k').textContent = fmt(t.kappa, 3);
      document.getElementById('w').textContent = fmt(t.width_m, 2) + ' m';
      document.getElementById('q').textContent = fmt(t.quality, 2);
      document.getElementById('mode').textContent = t.mode_txt;
      const l = document.getElementById('lost');
      l.textContent = t.is_lost ? '是' : '否';
      l.className = 'v ' + (t.is_lost ? 'bad' : 'ok');
      document.getElementById('reason').textContent = t.is_lost ? ('拒因：'+t.reason) : '';
    }
    document.getElementById('v').textContent = fmt(c.v, 3) + ' m/s';
    document.getElementById('steer').textContent = (c.steer_deg>=0?'+':'') + fmt(c.steer_deg,1) + '°';
    document.getElementById('cmdt').textContent = c.t ? (Math.max(0,(s.now-c.t)).toFixed(2)+' s 前') : '—';
    if(!built && s.params && Object.keys(s.params).length){
      const box = document.getElementById('params');
      TUNABLE.forEach(([n, label, hint])=>{
        const row = document.createElement('div');
        row.className = 'prow';
        row.innerHTML = '<span title="'+hint+'">'+label+'</span>';
        const inp = document.createElement('input');
        inp.id = 'p_'+n; inp.type = 'number'; inp.step = 'any';
        inp.value = (s.params[n]===null||s.params[n]===undefined)?'':s.params[n];
        inp.onkeydown = (ev)=>{ if(ev.key==='Enter'){
            fetch('/api/set?name='+n+'&value='+encodeURIComponent(inp.value))
              .then(()=>setTimeout(refresh,400));
        }};
        row.appendChild(inp);
        box.appendChild(row);
      });
      built = true;
    } else if(built && s.params){
      TUNABLE.forEach(([n])=>{
        const el = document.getElementById('p_'+n);
        if(el && document.activeElement !== el && s.params[n]!==null && s.params[n]!==undefined){
          el.value = s.params[n];
        }
      });
    }
  }).catch(()=>{});
}
setInterval(refresh, 400); refresh();
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    node = None

    def log_message(self, *a):
        pass

    def do_GET(self):
        p = self.path.split("?")[0]
        if p == "/":
            body = PAGE.replace(
                "%TUNABLE%",
                json.dumps([[n, l, h] for n, l, h in TUNABLE])).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif p == "/stream":
            self._stream()
        elif p == "/api/status":
            d, _ = self.node.snapshot()
            d["now"] = time.time()
            self._json(d)
        elif p == "/api/set":
            name = self.path.split("name=")[1].split("&")[0]
            value = self.path.split("value=")[1].split("&")[0]
            try:
                self.node.request_set(name, float(value))
                self._json({"ok": True})
            except Exception as ex:
                self._json({"ok": False, "err": str(ex)})
        elif p == "/snap":
            _, f = self.node.snapshot()
            if f is None:
                self.send_error(503)
                return
            ok, buf = cv2.imencode(".jpg", f, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(buf)))
            self.end_headers()
            self.wfile.write(buf.tobytes())
        else:
            self.send_error(404)

    def _json(self, o):
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
        last_seq = -1
        try:
            while True:
                # 只有真的有新帧才发，避免把旧图重复推 30 次/秒
                with self.node._lock:
                    seq = self.node.frame_seq
                    f = None if self.node.frame is None else self.node.frame
                if f is None or seq == last_seq:
                    time.sleep(0.02)
                    continue
                last_seq = seq
                ok, buf = cv2.imencode(".jpg", f, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                if not ok:
                    continue
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: ")
                self.wfile.write(str(len(buf)).encode())
                self.wfile.write(b"\r\n\r\n")
                self.wfile.write(buf.tobytes())
                self.wfile.write(b"\r\n")
        except (BrokenPipeError, ConnectionResetError):
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--target", default="/brain_node",
                    help="被调参数的节点，默认 /brain_node")
    a = ap.parse_args()

    rclpy.init()
    node = DebugView(a.target)
    Handler.node = node

    srv = ThreadingHTTPServer(("0.0.0.0", a.port), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()

    print("=" * 62)
    print("调参调试页")
    print("=" * 62)
    print("  浏览器打开:  http://%s:%d" % (_local_ip(), a.port))
    print("  改参数目标:  %s" % a.target)
    print("  订阅:        /debug/track_image  /track  /cmd_vel")
    print()
    print("  ⚠ 视觉那边要带 publish_debug:=true，否则没有画面：")
    print("    ros2 launch bringup start_race.launch.py "
          "dry_run:=false publish_debug:=true")
    print("=" * 62)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        srv.shutdown()
        node.destroy_node()
        rclpy.shutdown()


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

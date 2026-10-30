"""andy 实时推流：浏览器打开 http://<Pi>:8080 看
画面 = 检测图（YOLO 框）+ 右黄线拟合（绿）+ 避障目标路径（紫）+ 状态文字；下方小图 = 巡线调试图。
只订阅低频调试图（3~5 Hz），不碰 30 Hz 原图，对主循环影响很小。"""
import threading, time, math
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import cv2, numpy as np, rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import String, Float32MultiArray
from custom_msgs.msg import TrackInfo
from cv_bridge import CvBridge
from .lane_tracker import LaneTracker

LATEST = {"jpg": None}


class StreamNode(Node):
    def __init__(self):
        super().__init__("andy_stream")
        self.declare_parameter("port", 8080)
        self.br = CvBridge(); self.lt = LaneTracker()
        self.det = None; self.trk = None; self.rl = None; self.state = ""; self.track = None
        self.raw = None; self.t_det = 0.0
        # detect_node 另发一份 640x360 JPEG（原 1280x720 bgr8 每帧 2.7 MB，避障时 CPU 紧，DDS 丢帧 → 黑屏）
        # 注意：不要订阅 /camera/image_raw/compressed，会逼 usb_cam 每帧 30 Hz 压 JPEG，抢 CPU
        self.create_subscription(CompressedImage, "/debug/detect_image/jpeg", self.on_detj, 2)
        self.create_subscription(Image, "/debug/track_image", lambda m: (setattr(self, "trk", m), setattr(self, "t_trk", time.time())), 2)
        self.create_subscription(Float32MultiArray, "/lane_right", self.on_rl, 5)
        self.create_subscription(String, "/mission_state", lambda m: setattr(self, "state", m.data), 5)
        self.create_subscription(TrackInfo, "/track", lambda m: setattr(self, "track", m), 5)
        self.create_timer(0.2, self.compose)

    def on_detj(self, m):
        self.raw = m; self.t_raw = time.time()

    def on_det(self, m):
        self.det = m; self.t_det = time.time()

    def on_rl(self, m):
        self.rl = (list(m.data), time.time())

    def proj(self, d, x):
        try:
            p = self.lt.ground_to_image(d, x)
            return None if p is None else (int(p[0] / 2), int(p[1] / 2))
        except Exception:
            return None

    def compose(self):
        try:
            self._compose()
        except Exception as ex:   # 推流出错不能把节点弄死
            self.get_logger().error(f"compose: {ex!r}", throttle_duration_sec=5.0)

    def _compose(self):
        img = None
        if self.det is not None and time.time() - self.t_det < 1.0:
            img = cv2.resize(self.br.imgmsg_to_cv2(self.det, "bgr8"), (640, 360))
        elif self.raw is not None and time.time() - getattr(self, "t_raw", 0) < 3.0:
            im = cv2.imdecode(np.frombuffer(bytes(self.raw.data), np.uint8), cv2.IMREAD_COLOR)
            if im is not None:
                img = cv2.resize(im, (640, 360))
        if img is None and getattr(self, "last_img", None) is not None and time.time() - self.t_last < 5.0:
            img = self.last_img.copy()
            cv2.putText(img, f"STALE {time.time() - self.t_last:.1f}s", (240, 180), 0, 1.0, (0, 0, 255), 2)
        elif img is not None:
            self.last_img = img.copy(); self.t_last = time.time()
        if img is None:
            img = np.zeros((360, 640, 3), np.uint8)
            cv2.putText(img, "waiting for camera / launch ...", (120, 190), 0, 0.8, (0, 200, 255), 2)
            cv2.putText(img, time.strftime("%H:%M:%S"), (270, 230), 0, 0.7, (200, 200, 200), 1)
        st = self.state
        D = None
        if " D=" in st:
            try:
                D = float(st.split(" D=")[1].split()[0])
            except Exception:
                D = None
        if self.rl and time.time() - self.rl[1] < 0.5 and self.rl[0][2] >= 6:
            a, b, n = self.rl[0]
            for dd, col, off in ((None, (0, 255, 0), 0.0), (D, (255, 0, 255), D or 0.0)):
                if dd is None and col != (0, 255, 0):
                    continue
                pts = [self.proj(d, a + b * d - off) for d in np.arange(0.25, 1.2, 0.05)]
                pts = [p for p in pts if p is not None]
                for p, q in zip(pts, pts[1:]):
                    cv2.line(img, p, q, col, 3)
            cv2.putText(img, f"dR={a:.2f}m", (8, 330), 0, 0.6, (0, 255, 0), 2)
        else:
            cv2.putText(img, "right line: --", (8, 330), 0, 0.6, (0, 0, 255), 2)
        cv2.rectangle(img, (0, 0), (640, 26), (0, 0, 0), -1)
        cv2.putText(img, st[:70], (6, 18), 0, 0.5, (255, 255, 255), 1)
        if self.track is not None:
            t = self.track
            cv2.putText(img, f"e={t.lateral_error*100:+.0f}cm th={math.degrees(t.heading_error):+.0f} m{t.mode} q{t.quality:.2f}",
                        (330, 330), 0, 0.5, (255, 255, 0), 1)
        if self.trk is not None and time.time() - getattr(self, "t_trk", 0) < 1.0:
            tr = self.br.imgmsg_to_cv2(self.trk, "bgr8")
            h = int(640 * tr.shape[0] / tr.shape[1])
            img = np.vstack([img, cv2.resize(tr, (640, h))])
        ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if ok:
            LATEST["jpg"] = jpg.tobytes()


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path.startswith("/stream"):
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=f")
            self.end_headers()
            try:
                last = None
                while True:
                    j = LATEST["jpg"]
                    if j is not None and j is not last:
                        last = j
                        self.wfile.write(b"--f\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(j) + j + b"\r\n")
                    time.sleep(0.1)
            except Exception:
                return
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write("<html><body style='margin:0;background:#111'><img src='/stream' style='width:100%;max-width:960px'></body></html>".encode())


def main():
    rclpy.init()
    n = StreamNode()
    srv = ThreadingHTTPServer(("0.0.0.0", int(n.get_parameter("port").value)), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    n.get_logger().info(f"推流: http://<Pi IP>:{n.get_parameter('port').value}")
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

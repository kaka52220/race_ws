#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rosmaster 扩展板 (YB-ERF01) 串口自检 / 调试工具
=================================================

用法:
    python rosmaster_test.py            # 自动找 CH340 串口
    python rosmaster_test.py COM7       # 指定串口

状态行(每 0.3 秒刷新):
    V=12.4V  Vx=  +52 Vy=  +0 Vz=  +0  |  M2:  +18 M4:  +17

    M2/M4 后面是这个采样周期(约 40ms)的编码器增量。
    用手推车 / 架空转轮子时看它的正负号, 就能判断编码器方向对不对:
        往前转 -> 增量为正  = 方向正确
        往前转 -> 增量为负  = 方向反了
        怎么转都是 0      = 编码器没信号

命令(直接敲, 回车执行):
    v              读固件版本
    c 5            设车型  1=X3 2=X3PLUS 4=X1 5=R2阿克曼 6=旭日派
    l 10           左转 10 度 (车不动, 只打方向)
    r 10           右转 10 度
    m              方向回中
    f 50           前进 50 mm/s (闭环, 走 PID)
    b 50           后退 50 mm/s
    o 5            开环占空比 5% (绕过 PID, 测方向用; 轮子会明显转动)
    s              停车
    a on / a off   自动上报 开 / 关   (关掉后就没有实时数据了)
    q              退出

安全:
    * 第一次测 f/b 之前, 一定先架空确认方向和编码器符号
    * 车要失控时按 Ctrl+C, 程序会自动发停止帧
"""

import sys
import time
import threading

try:
    import serial
    from serial.tools import list_ports
except ImportError:
    print("缺少 pyserial, 先运行:  pip install pyserial")
    sys.exit(1)

HEAD_HOST = 0xFC          # 上位机 -> 板子
HEAD_BOARD = 0xFB         # 板子 -> 上位机 (PTO_DEVICE_ID - 1)

F_AUTO_REPORT = 0x01
F_REPORT_SPEED = 0x0A
F_REPORT_ENC = 0x0D
F_MOTOR = 0x10
F_MOTION = 0x12
F_CAR_TYPE = 0x15
F_AKM_STEER = 0x31
F_REQUEST = 0x50
F_VERSION = 0x51

SAVE = 0x5F


def s16(v):
    v = int(v)
    return bytes([v & 0xFF, (v >> 8) & 0xFF])


def build(func, payload=b""):
    """组一帧上位机命令 (返回 bytes)"""
    n = len(payload) + 3
    f = bytes([0xFF, HEAD_HOST, n, func]) + bytes(payload)
    return f + bytes([sum(f[2:]) & 0xFF])


class Parser:
    """解析板子回来的帧 (帧头 FF FB), 容忍开机的 ASCII 日志"""

    def __init__(self):
        self.buf = bytearray()

    def feed(self, data):
        self.buf += data
        out = []
        while True:
            i = self.buf.find(b"\xff\xfb")
            if i < 0:
                if len(self.buf) > 128:
                    del self.buf[:-2]
                return out
            if i:
                del self.buf[:i]
            if len(self.buf) < 3:
                return out
            n = self.buf[2]
            if n < 3 or n > 40:
                del self.buf[:2]
                continue
            if len(self.buf) < n + 2:
                return out
            body = bytes(self.buf[3:n + 2])      # func + 参数 + 校验 (n-1 字节)
            if (n + sum(body[:-1])) & 0xFF == body[-1]:
                out.append((body[0], body[1:-1]))
                del self.buf[:n + 2]
            else:
                del self.buf[:2]


class Rosmaster:
    def __init__(self, port, baud=115200):
        self.port = port
        self.ser = serial.Serial(port, baud, timeout=0.02)
        self.parser = Parser()
        self.lock = threading.Lock()
        self.running = True
        self.typing = False

        self.voltage = 0.0
        self.vx = self.vy = self.vz = 0
        self.enc = [0, 0, 0, 0]
        self.enc_prev = [0, 0, 0, 0]
        self.enc_delta = [0, 0, 0, 0]
        self.enc_seen = False

    # ---------------- 底层 ----------------
    def send(self, func, payload=b""):
        with self.lock:
            self.ser.write(build(func, payload))

    def _rx_loop(self):
        while self.running:
            try:
                data = self.ser.read(256)
            except Exception:
                break
            if data:
                for func, pl in self.parser.feed(data):
                    self._handle(func, pl)

    def _handle(self, func, pl):
        if func == F_REPORT_SPEED and len(pl) >= 7:
            self.vx = int.from_bytes(pl[0:2], "little", signed=True)
            self.vy = int.from_bytes(pl[2:4], "little", signed=True)
            self.vz = int.from_bytes(pl[4:6], "little", signed=True)
            self.voltage = pl[6] / 10.0
        elif func == F_REPORT_ENC and len(pl) >= 16:
            now = [int.from_bytes(pl[i * 4:i * 4 + 4], "little", signed=True) for i in range(4)]
            if self.enc_seen:
                self.enc_delta = [now[i] - self.enc_prev[i] for i in range(4)]
            self.enc = now
            self.enc_prev = now
            self.enc_seen = True

    # ---------------- 命令 ----------------
    def stop(self):
        self.send(F_MOTION, b"\x00" + s16(0) + s16(0) + s16(0))

    def version(self):
        self.send(F_REQUEST, bytes([F_VERSION, 0]))

    def set_car_type(self, t):
        self.send(F_CAR_TYPE, bytes([t & 0xFF, SAVE]))

    def steer(self, deg):
        self.send(F_AKM_STEER, bytes([1, int(deg) & 0xFF]))

    def motion(self, vx):
        self.send(F_MOTION, b"\x00" + s16(vx) + s16(0) + s16(0))

    def open_loop(self, pct):
        self.send(F_MOTOR, bytes([0, int(pct) & 0xFF, 0, int(pct) & 0xFF]))

    def auto_report(self, on):
        self.send(F_AUTO_REPORT, bytes([1 if on else 0, 0]))

    def status_line(self):
        if not self.enc_seen:
            enc = "M2 n/a (自动上报关了?)"
        else:
            enc = "M2 %+5d/%-7d M4 %+5d/%-7d" % (
                self.enc_delta[1], self.enc[1], self.enc_delta[3], self.enc[3])
        return "V=%4.1fV Vx=%+5d Vy=%+4d Vz=%+4d | %s" % (
            self.voltage, self.vx, self.vy, self.vz, enc)

    # ---------------- 线程 ----------------
    def start(self):
        threading.Thread(target=self._rx_loop, daemon=True).start()
        threading.Thread(target=self._status_loop, daemon=True).start()

    def _status_loop(self):
        while self.running:
            if not self.typing:
                sys.stdout.write("\r" + self.status_line().ljust(74))
                sys.stdout.flush()
            time.sleep(0.3)

    def close(self):
        self.running = False
        time.sleep(0.1)
        try:
            self.ser.close()
        except Exception:
            pass


HELP = """
命令:  v=读版本    c 5=设车型    l 10 / r 10=转向    m=回中
       f 50=前进    b 50=后退     o 5=开环5%          s=停车
       a on|off=自动上报   e=编码器累计值        q=退出

状态行:  M2 +18/1204  前面的 +18 是本次采样(40ms)的增量,
                   后面的 1204 是开机以来的累计值
"""


def pick_port():
    ports = list(list_ports.comports())
    for p in ports:
        d = (p.description or "").upper()
        if "CH340" in d or "CH34" in d or "USB-SERIAL" in d or "USB2.0-SERIAL" in d:
            return p.device
    if len(ports) == 1:
        return ports[0].device
    return None


def process(rm, line):
    parts = line.split()
    if not parts:
        return True
    cmd, arg = parts[0], (parts[1] if len(parts) > 1 else None)
    try:
        if cmd == "q":
            return False
        elif cmd == "v":
            rm.version()
            return "读版本"
        elif cmd == "e":
            return "累计计数  M1=%d   M2=%d   M3=%d   M4=%d" % (
                rm.enc[0], rm.enc[1], rm.enc[2], rm.enc[3])
        elif cmd == "c" and arg:
            rm.set_car_type(int(arg))
            return "车型 = %s" % arg
        elif cmd in ("l", "r") and arg:
            rm.steer(int(arg) if cmd == "r" else -int(arg))
            return "转向 %s%s 度" % (cmd.upper(), arg)
        elif cmd == "m":
            rm.steer(0)
            return "方向回中"
        elif cmd in ("f", "b") and arg:
            v = int(arg) * (1 if cmd == "f" else -1)
            rm.motion(v)
            return "%s %s mm/s" % ("前进" if v > 0 else "后退", abs(v))
        elif cmd == "o" and arg:
            rm.open_loop(int(arg))
            return "开环 %s%% (发 s 停)" % arg
        elif cmd == "s":
            rm.stop()
            rm.open_loop(0)
            return "停车"
        elif cmd == "a" and arg:
            rm.auto_report(arg == "on")
            return "自动上报 %s" % arg
    except ValueError:
        return "参数不对"
    return "?? 不认识的命令"


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    port = args[0] if args else pick_port()
    if not port:
        print("没找到串口, 请指定, 例如:  python rosmaster_test.py COM7")
        print("当前可用串口:")
        for p in list_ports.comports():
            print("   %-8s %s" % (p.device, p.description))
        return 1

    print("打开 %s ..." % port)
    try:
        rm = Rosmaster(port)
    except Exception as e:
        print("打开失败: %s" % e)
        return 1
    rm.start()
    time.sleep(0.4)
    print(HELP)

    try:
        import msvcrt
        buf = ""
        while rm.running:
            ch = msvcrt.getwch()
            if ch in ("\x00", "\xe0"):          # 方向键等, 吞掉第二个字节
                msvcrt.getwch()
                continue
            if ch == "\x03":                    # Ctrl+C
                raise KeyboardInterrupt
            if ch in ("\r", "\n"):
                print()
                msg = process(rm, buf.strip().lower())
                if msg is False:
                    break
                if msg:
                    print("  " + str(msg))
                buf = ""
                rm.typing = False
            elif ch == "\x08":                  # 退格
                if buf:
                    buf = buf[:-1]
                    sys.stdout.write("\b \b")
                    sys.stdout.flush()
            else:
                if not rm.typing:
                    sys.stdout.write("\n> ")
                    rm.typing = True
                buf += ch
                sys.stdout.write(ch)
                sys.stdout.flush()
    except KeyboardInterrupt:
        print("\n中断")
    except ImportError:
        # 非 Windows 控制台的兜底
        while True:
            try:
                line = input("> ")
            except (EOFError, KeyboardInterrupt):
                break
            if not process(rm, line.strip().lower()):
                break
    finally:
        print("停车, 关闭串口")
        try:
            rm.stop()
            rm.open_loop(0)
            time.sleep(0.05)
        except Exception:
            pass
        rm.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

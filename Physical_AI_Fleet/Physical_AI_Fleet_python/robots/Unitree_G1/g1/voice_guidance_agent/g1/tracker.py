import sys, time, json, socket, urllib.request
import numpy as np, cv2, gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst
Gst.init(None)
DET = sys.argv[1]
p = Gst.parse_launch(
 "udpsrc address=230.1.1.1 port=1720 multicast-iface=eth0 ! application/x-rtp,encoding-name=H264 "
 "! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! video/x-raw,format=BGR "
 "! appsink name=sink emit-signals=false drop=true max-buffers=1 sync=false")
sink = p.get_by_name("sink"); p.set_state(Gst.State.PLAYING)
u = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
first = True
while True:
    s = sink.emit("try-pull-sample", 500_000_000)
    if s is None: continue
    st = s.get_caps().get_structure(0); w, h = st.get_value("width"), st.get_value("height")
    ok, m = s.get_buffer().map(Gst.MapFlags.READ)
    f = np.frombuffer(m.data, np.uint8).reshape(h, w, 3).copy(); s.get_buffer().unmap(m)
    if f.std() < 12: continue
    f = cv2.resize(f, (640, 360))
    jpg = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 70])[1].tobytes()
    try:
        req = urllib.request.Request(DET + ("?lock=1" if first else ""), data=jpg)
        r = json.loads(urllib.request.urlopen(req, timeout=1.5).read())
        if r.get("found"): first = False
    except Exception:
        r = {"found": False}
    r["t"] = time.time()
    u.sendto(json.dumps(r).encode(), ("127.0.0.1", 9100))

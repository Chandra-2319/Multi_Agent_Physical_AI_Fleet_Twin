import sys, time
import numpy as np, cv2, gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst
Gst.init(None)
p = Gst.parse_launch(
 "udpsrc address=230.1.1.1 port=1720 multicast-iface=eth0 ! application/x-rtp,encoding-name=H264 "
 "! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! video/x-raw,format=BGR "
 "! appsink name=sink emit-signals=false drop=true max-buffers=1 sync=false")
sink = p.get_by_name("sink")
p.set_state(Gst.State.PLAYING)
out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/view.jpg"
good, t0 = 0, time.time()
while time.time() - t0 < 20:
    s = sink.emit("try-pull-sample", 500_000_000)
    if s is None: continue
    st = s.get_caps().get_structure(0)
    w, h = st.get_value("width"), st.get_value("height")
    ok, m = s.get_buffer().map(Gst.MapFlags.READ)
    f = np.frombuffer(m.data, np.uint8).reshape(h, w, 3).copy()
    s.get_buffer().unmap(m)
    if f.std() > 12: good += 1
    else: good = 0
    if good >= 5:
        cv2.imwrite(out, f); print("saved", out, w, h); break
else:
    print("no good frame, check the stream")
p.set_state(Gst.State.NULL)

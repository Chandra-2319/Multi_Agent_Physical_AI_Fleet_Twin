import socket, json, math, omni.kit.app, omni.usd
from pxr import UsdGeom, Gf

PRIM_PATH = "/World/mir"   # change this

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("0.0.0.0", 5005))
sock.setblocking(False)

stage = omni.usd.get_context().get_stage()
api = UsdGeom.XformCommonAPI(stage.GetPrimAtPath(PRIM_PATH))

def on_update(e):
    data = None
    try:
        while True:
            data, _ = sock.recvfrom(4096)
    except BlockingIOError:
        pass
    if data:
        d = json.loads(data)
        api.SetTranslate(Gf.Vec3d(d["x"], d["y"], 0.0))
        api.SetRotate(Gf.Vec3f(0, 0, math.degrees(d["yaw"])))

sub = omni.kit.app.get_app().get_update_event_stream().create_subscription_to_pop(on_update)

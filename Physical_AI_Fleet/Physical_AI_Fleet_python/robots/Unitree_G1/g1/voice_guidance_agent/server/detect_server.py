import numpy as np, cv2
from fastapi import FastAPI, Request
from ultralytics import YOLO

app = FastAPI()
model = YOLO("yolov8n.pt")
target = {"hist": None}

def torso_hist(img, b):
    x1, y1, x2, y2 = [int(v) for v in b]
    h = y2 - y1
    crop = img[max(0, y1): y2, max(0, x1): x2]
    if crop.size == 0: return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [18, 8], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist

@app.post("/detect")
async def detect(request: Request, lock: int = 0):
    img = cv2.imdecode(np.frombuffer(await request.body(), np.uint8), cv2.IMREAD_COLOR)
    H, W = img.shape[:2]
    r = model(img, classes=[0], conf=0.25, verbose=False)[0]
    boxes = r.boxes.xyxy.cpu().numpy()
    if len(boxes) == 0: return {"found": False}
    if lock or target["hist"] is None:
        chosen = max(boxes, key=lambda b: (b[3] - b[1]) - 0.5 * abs((b[0] + b[2]) / 2 - W / 2))
        target["hist"] = torso_hist(img, chosen); score = 1.0
    else:
        sc = []
        for b in boxes:
            hh = torso_hist(img, b)
            sc.append(cv2.compareHist(target["hist"], hh, cv2.HISTCMP_CORREL) if hh is not None else -1)
        i = int(np.argmax(sc)); chosen, score = boxes[i], float(sc[i])
        if score < 0.2: return {"found": False, "score": score}
    x1, y1, x2, y2 = chosen
    return {"found": True, "cx": float((x1 + x2) / 2 / W), "h": float((y2 - y1) / H), "yb": float(y2 / H), "w": float((x2 - x1) / W),
            "score": float(score), "n": int(len(boxes))}

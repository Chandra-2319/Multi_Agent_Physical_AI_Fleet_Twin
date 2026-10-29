# Physical_AI_Fleet v0.5.3 - the vision model finds the gauge in the image (bounding box), then reads the crop

## Install
Replace Physical_AI_Fleet_python/, config/ and tools/ with the ones in this zip. Copy inputs/cosmos.json over yours (new
"detect" block and crop.source). Other inputs are not touched. Restart Isaac Sim.

## 1. First check which box format your model follows (2 minutes, no Isaac Sim)
  cd <extension folder>
  python3 tools/test_gauge_bbox.py ~/physical_ai_fleet_logs/evidence/<a full evidence frame>.jpg --all-formats
Open the *_norm1000_box.jpg, *_norm1_box.jpg, *_pixels_box.jpg: keep the format whose green box fits the round dial and put it
in inputs/cosmos.json "detect": {"format": ...}. The script also reads each crop, so you see the end result.
If no format fits, try --think (thinking for the box step too) and set "thinking": true.

## 2. How a reading works now
frame -> "is there a pressure gauge, where?" (quick call, thinking off, ~2-5 s) -> crop to 1.8 x the box, 768 px ->
"what is the pressure?" (thinking on). If the model finds no gauge (or confidence < detect.min_confidence 0.5), the
projected crop is used in the simulation; on a real robot the whole frame is read, and the reader can still ask to turn.
Audit: gauge_detect (box, confidence, iou_with_projection in the simulation), gauge_crop (source vlm / projection / full).
To compare: "crop": {"source": ["projection", "vlm"]} = the v0.5.2 behaviour with the model's box as fallback.

## 3. Real quadruped
The same session runs for a Digital Twin robot: its camera JPEG goes through the box step, no positions are used.

## Tests
python -m pytest -q tests -> 347 passed (325 existing + 22 new). Not tested: whether your model follows the box format
(step 1 checks that), Isaac Sim itself.

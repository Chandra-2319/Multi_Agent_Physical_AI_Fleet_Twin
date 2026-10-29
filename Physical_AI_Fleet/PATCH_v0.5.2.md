# Physical_AI_Fleet v0.5.2 - the gauge is cropped before the vision model reads it

## Install
Replace Physical_AI_Fleet_python/ and config/ with the ones in this zip. Copy inputs/cosmos.json over yours (new timing +
"crop" block). Your other inputs (asset_types, gauges, layout, site map, robots_actions) are not touched. Restart Isaac Sim.

## What changed
1. Crop: at the asset the frame is cut to a square around the dial (crop.margin x the dial diameter, default 1.8) and enlarged
   to 768 px. The dial position comes from the needle pivot in your Motor_with_gauge.usd, projected with Isaac Sim's camera
   (get_image_coords_from_world_points); if that is not available, from the robot pose and the camera mount. The dial radius is
   estimated from the needle size; set gauges.json "dial_radius_m" (world metres, after usd_scale) to override.
   Evidence: ..._a1.jpg (full frame) and ..._a1_crop.jpg (what the model saw) + ..._a1_crop.cosmos.json.
   If the gauge cannot be found in the image the full frame is sent (nothing is lost) and gauge_crop says why.
2. Prompt: HIGH = the needle points into the red sector, even if its tip stops short of the arc.
3. Timing (your 8-15 s per image with thinking on): call_timeout_s 45, total_time_s 120, max_tokens 8192. Thinking on by default.

## Check in Isaac Sim
- Audit log "gauge_crop": method "camera" is expected; "centre_px_camera" and "centre_px_pose" should be close.
- Open the newest *_crop.jpg: the dial should be in the middle and fill about half of it. If it is cut off, raise crop.margin;
  if it is too small, set dial_radius_m.
- TEST VLM on a *_crop.jpg gives the same result the robot gets.

## Tests
python -m pytest -q tests  -> 325 passed (316 existing + 9 new). Not tested: Isaac Sim's camera projection itself.

# SPDX-License-Identifier: Apache-2.0
"""UI for the Physical_AI_Fleet extension (structure follows the Loaded_scenario template).
Frames:
  World Controls   Environment USD (browse), ground plane / obstacle options, LOAD / RESET
  AI Models        v0.5.1: Provider (NVIDIA / Nebius); Nemotron URL / model / key; VLM URL / model / key (thinking always on);
                   Profile File; LOAD / SAVE / FORGET / APPLY; TEST NEMOTRON; gauge test image + TEST VLM; ASK NOW
  Fleet Builder    Go2 / Spot / H1 / G1 / AMR, Simulation or Digital Twin; policy .pt / env.yaml shown
  Assets & Site    equipment type + ID + X / Y / yaw -> ADD ASSET; REMOVE; layout file LOAD / SAVE;
                   site map file LOAD (walkable area + no-go zones), show / hide the site map
  Run Fleet        RUN / STOP, routine inspections, camera capture, camera light
  Manual Control   operator skills (still gated), stop, home, reset robot
  Demo Events      pick the asset + fault + priority and INJECT: a quadruped is sent to inspect and read the gauge
  Approvals        approve / reject high-risk assignments
  Fleet Status     physics rate, environment, robots, incidents, Nemotron status
  Audit Log        latest events, model calls, gate decisions
"""
from __future__ import annotations

import os
import traceback

import omni.timeline
import omni.ui as ui
import omni.usd
from isaacsim.examples.extension.core_connectors import LoadButton, ResetButton
from isaacsim.gui.components.element_wrappers import (
    Button, CheckBox, CollapsableFrame, DropDown, FloatField, StateButton, StringField, TextBlock,
)
from isaacsim.gui.components.ui_utils import get_style
from omni.usd import StageEventType

from .common.audit_log import AuditLog
from .fleet_config.catalog import MODELS, SKILLS
from .fleet_config.settings import get_settings
from .perception.frame_capture import load_operator_photo
from .fleet_config import ai_profiles
from .fleet_config.paths import inputs_dir
from .scenario import USD_EXTS, FleetScenario

MODEL_LABELS = {m.label: m.key for m in MODELS.values()}
MODE_LABELS = {"Simulation": "simulation", "Digital Twin": "digital_twin"}
FAULT_KINDS = ["overpressure", "leak"]
DEMO_PRIORITIES = ["high", "medium", "low"]  # first item = what the drop-down shows by default
NO_ITEMS = "(none)"
PROVIDER_LABELS = {"NVIDIA": "nvidia", "Nebius": "nebius"}
PROVIDER_KEYS = {v: k for k, v in PROVIDER_LABELS.items()}
# The little "copy" icon next to a text block uses pyperclip, which needs xclip / xsel on Linux. Without it Kit logs
# "AttributeError: module 'pyperclip' has no attribute 'EXCEPT_MSG'". The full log is in the JSONL file instead.
COPY_BUTTONS = False


def _ext_filter(exts):
    def accept(path: str) -> bool:
        return os.path.splitext(path.lower())[1] in exts
    return accept


class UIBuilder:
    def __init__(self) -> None:
        self.frames = []
        self.wrapped_ui_elements = []
        self._timeline = omni.timeline.get_timeline_interface()
        self._on_init()

    # ------------------------------------------------------------------ called by extension.py
    def on_menu_callback(self) -> None:
        pass

    def on_timeline_event(self, event) -> None:
        play = getattr(omni.timeline, "GLOBAL_EVENT_PLAY", None)
        if play is not None and event.event_name == play:
            note = self._scenario.start_run()
            if note:
                self._run_note = note
                self._refresh()
        elif event.event_name == omni.timeline.GLOBAL_EVENT_STOP:
            self._scenario.on_timeline_stop()
            self._run_note = "Stopped: robots are back home and every pending task was cleared."
            if hasattr(self, "_run_btn"):
                self._run_btn.reset()
                self._run_btn.enabled = self._scenario.coordinator is not None
            self._refresh_asset_widgets()
            self._refresh()

    def on_physics_step(self, step: float) -> None:
        pass

    def on_stage_event(self, event) -> None:
        if event.event_name == omni.usd.get_context().stage_event_name(StageEventType.OPENED):
            if not self._scenario.owns_current_stage():
                self._reset_extension()

    def cleanup(self) -> None:
        self._ping_sub = None
        self._cosmos_sub = None
        for ui_elem in self.wrapped_ui_elements:
            ui_elem.cleanup()

    def shutdown(self) -> None:
        self.cleanup()
        self._scenario.teardown()

    def build_ui(self) -> None:
        self.wrapped_ui_elements = []
        self._build_world_frame()
        self._build_brain_frame()
        self._build_fleet_builder_frame()
        self._build_assets_frame()
        self._build_run_frame()
        self._build_manual_frame()
        self._build_demo_frame()
        self._build_approval_frame()
        self._build_status_frame()
        loaded = self._scenario.coordinator is not None
        self._reset_btn.enabled = self._run_btn.enabled = self._add_btn.enabled = self._remove_btn.enabled = loaded
        self._brain_now_btn.enabled = loaded
        self._fill_policy_fields()
        problems = self._scenario.settings.load_problems
        if problems:
            self._asset_note = "Problems when loading the JSON files: " + " | ".join(problems)
        if self._scenario.inputs_created:
            self._asset_note = (self._asset_note + "\n" if self._asset_note else "") + \
                f"Created {inputs_dir()} with: {', '.join(self._scenario.inputs_created)} - edit the files there."
        self._refresh_asset_widgets()
        self._refresh()

    # ------------------------------------------------------------------ frames
    def _build_world_frame(self) -> None:
        s = get_settings()
        with CollapsableFrame("World Controls", collapsed=False):
            with ui.VStack(style=get_style(), spacing=5, height=0):
                self._env_field = StringField(
                    "Environment USD", default_value=self._env_usd, use_folder_picker=True,
                    item_filter_fn=_ext_filter(USD_EXTS), on_value_changed_fn=self._set("_env_usd"),
                    tooltip="Optional scene (warehouse, plant...) referenced under /World/Environment on LOAD. "
                            "Browse with the folder icon, or type a /Isaac/... path. Leave empty for a flat floor.")
                self._ground_cb = CheckBox("Add Ground Plane", default_value=self._add_ground,
                                           on_click_fn=self._set("_add_ground"),
                                           tooltip="Turn off if the environment already has a floor with collision")
                self._env_obs_cb = CheckBox("Environment Obstacles", default_value=self._env_obstacles,
                                            on_click_fn=self._set("_env_obstacles"),
                                            tooltip="Walking robots plan around the environment's collision geometry")
                self._load_btn = LoadButton(
                    "Load Scene", "LOAD",
                    setup_scene_fn=lambda: self._scenario.setup_scene(self._env_usd, self._add_ground, self._env_obstacles),
                    setup_post_load_fn=self._on_loaded)
                self._load_btn.set_world_settings(physics_dt=s.physics_dt, rendering_dt=s.rendering_dt)
                self._reset_btn = ResetButton("Reset Scene", "RESET", pre_reset_fn=None, post_reset_fn=self._on_reset)
                self.wrapped_ui_elements += [self._env_field, self._ground_cb, self._env_obs_cb, self._load_btn, self._reset_btn]

    def _key_row(self, label: str, tooltip: str):
        with ui.HStack(height=22):
            ui.Label(label, width=150, tooltip=tooltip)
            model = ui.SimpleStringModel("")
            ui.StringField(model=model, password_mode=True)
        return model

    def _build_brain_frame(self) -> None:
        with CollapsableFrame("AI Models", collapsed=False):
            with ui.VStack(style=get_style(), spacing=5, height=0):
                self._provider_dd = DropDown("Provider", populate_fn=lambda: sorted(PROVIDER_LABELS, key=lambda l: PROVIDER_LABELS[l] != self._provider),
                                             on_selection_fn=self._on_provider_selected,
                                             tooltip="NVIDIA or Nebius. Selecting one fills the six fields below from the Profile File.")
                self._brain_url = StringField("Nemotron URL", default_value="", tooltip="Fleet-brain endpoint (OpenAI-compatible)")
                self._brain_model = StringField("Nemotron Model", default_value="", tooltip="Fleet-brain model id (text)")
                self._brain_key = self._key_row("Nemotron API Key", "Hidden. Blank on SAVE keeps the stored key.")
                self._vlm_url = StringField("VLM URL", default_value="",
                                            tooltip="Vision model that reads the gauge. A full invoke URL (.../chat/completions, "
                                                    ".../vlm/..., .../cosmos/...) is used as-is.")
                self._vlm_model = StringField("VLM Model", default_value="", tooltip="e.g. nvidia/nemotron-3.5-super-vl-120b-a12b")
                self._vlm_key = self._key_row("VLM API Key", "Hidden. Blank on SAVE keeps the stored key.")
                self._profile_field = StringField("Profile File", default_value=str(ai_profiles.profiles_path()),
                                                  use_folder_picker=True, item_filter_fn=_ext_filter((".json",)),
                                                  tooltip="ai_profiles.json holding both providers (with keys). Kept outside "
                                                          "the extension folder by default.")
                self._ai_load_btn = Button("Load", "LOAD", on_click_fn=self._on_load_ai,
                                           tooltip="Fill all six fields from the Profile File and use them")
                self._ai_save_btn = Button("Save", "SAVE", on_click_fn=self._on_save_ai,
                                           tooltip="Use the six fields and write them as this provider's profile")
                self._ai_forget_btn = Button("Forget", "FORGET", on_click_fn=self._on_forget_ai,
                                             tooltip="Delete this provider's profile from the Profile File")
                self._brain_apply_btn = Button("Apply", "APPLY", on_click_fn=self._on_apply_brain,
                                               tooltip="Use the six fields now, for this session only")
                self._brain_test_btn = Button("Test Nemotron", "TEST NEMOTRON", on_click_fn=self._on_test_brain,
                                              tooltip="One tiny request to the fleet-brain model")
                self._cosmos_img_field = StringField("Gauge Test Image", default_value="", use_folder_picker=True,
                                                     item_filter_fn=_ext_filter((".jpg", ".jpeg", ".png")),
                                                     tooltip="A picture of the gauge (e.g. a saved evidence image)")
                self._cosmos_test_btn = Button("Test VLM", "TEST VLM", on_click_fn=self._on_test_cosmos,
                                               tooltip="Ask the vision model about the test image (in the background)")
                self._brain_now_btn = Button("Ask Now", "ASK NEMOTRON NOW", on_click_fn=self._on_ask_now,
                                             tooltip="Builds a fleet snapshot and asks for a decision right away")
                self._brain_msg = TextBlock("AI", num_lines=8, include_copy_button=COPY_BUTTONS)
                self.wrapped_ui_elements += [self._provider_dd, self._brain_url, self._brain_model, self._vlm_url,
                                             self._vlm_model, self._profile_field, self._ai_load_btn,
                                             self._ai_save_btn, self._ai_forget_btn, self._brain_apply_btn, self._brain_test_btn,
                                             self._cosmos_img_field, self._cosmos_test_btn, self._brain_now_btn]
                self._brain_now_btn.enabled = False
                self._provider_dd.repopulate()
                self._fill_ai_fields(self._scenario.profile_fields(self._provider, self._profile_field.get_value()),
                                     from_settings=True)

    def _build_fleet_builder_frame(self) -> None:
        with CollapsableFrame("Fleet Builder", collapsed=False):
            with ui.VStack(style=get_style(), spacing=5, height=0):
                self._model_dd = DropDown("Robot Model", tooltip="Go2 / Spot quadrupeds, H1 / G1 humanoids, AMR",
                                          populate_fn=lambda: list(MODEL_LABELS), on_selection_fn=self._on_model_selected)
                self._mode_dd = DropDown("Mode", tooltip="Simulation: walks with its RL policy. "
                                                         "Digital Twin: a real robot drives the sim pose.",
                                         populate_fn=lambda: list(MODE_LABELS), on_selection_fn=self._on_mode_selected)
                self._walk_cb = CheckBox("Walking Policy", default_value=True, on_click_fn=self._set("_walking"),
                                         tooltip="ON = RL walking policy (physics). OFF = kinematic glide fallback")
                self._id_field = StringField("Robot ID", default_value="go2-1")
                self._usd_field = StringField("USD Path", default_value=MODELS["go2"].usd,
                                              tooltip="Leave the default for walking robots (the policy spawns its own USD). "
                                                      "Real Go2 mirror: a Go2 USD, e.g. Assets/Go2_twin.usd (any ROS graph in it is switched off)")
                self._policy_field = StringField(
                    "Policy (.pt)", default_value="", use_folder_picker=True,
                    item_filter_fn=_ext_filter((".pt", ".jit", ".onnx")),
                    tooltip="Filled with the default policy of the selected model. Change it to use your own policy.")
                self._env_yaml_field = StringField(
                    "Policy env.yaml", default_value="", use_folder_picker=True,
                    item_filter_fn=_ext_filter((".yaml", ".yml")),
                    tooltip="The env.yaml that belongs to the policy above (joint order, gains, dt, decimation).")
                self._conn_field = StringField("Connection", default_value="",
                                               tooltip="Digital Twin only. Real Go2 mirror: udp:9870 (sent by go2_udp_sender.py "
                                                       "on the Go2; no ROS on the PC), add ;jog=<robot ip>:9999 for the JOG buttons. "
                                                       "Go2 bridge: host:port. MiR: IP or http URL.")
                self._frame_field = StringField("Map -> World", default_value="",
                                                tooltip="Digital Twin only. 'tx,ty,theta_deg' from the robot map to the world.")
                self._x_field = FloatField("Home X (m)", default_value=0.0, step=0.5)
                self._y_field = FloatField("Home Y (m)", default_value=-3.0, step=0.5)
                self._yaw_field = FloatField("Home Yaw (deg)", default_value=0.0, step=15.0,
                                             tooltip="Heading of the robot at its home: 0 = facing +X, 90 = facing +Y")
                self._add_btn = Button("Add Robot", "ADD ROBOT", on_click_fn=self._on_add_robot)
                self._remove_btn = Button("Remove Selected", "REMOVE", on_click_fn=self._on_remove_robot,
                                          tooltip="Removes the robot selected in Manual Control")
                self._builder_msg = TextBlock("Result", num_lines=5, include_copy_button=COPY_BUTTONS)
                self.wrapped_ui_elements += [self._model_dd, self._mode_dd, self._walk_cb, self._id_field, self._usd_field,
                                             self._policy_field, self._env_yaml_field, self._conn_field, self._frame_field,
                                             self._x_field, self._y_field, self._yaw_field, self._add_btn, self._remove_btn]
                self._model_dd.repopulate()
                self._mode_dd.repopulate()

    def _build_assets_frame(self) -> None:
        sc = self._scenario
        with CollapsableFrame("Assets & Site", collapsed=False):
            with ui.VStack(style=get_style(), spacing=5, height=0):
                self._a_type_dd = DropDown("Equipment Type", populate_fn=self._type_labels, on_selection_fn=self._on_type_selected,
                                           tooltip="Types come from fleet_config/asset_types.json (usd payload, footprint, readings)")
                self._a_id = StringField("Asset ID", default_value=sc.next_asset_id(self._a_type),
                                         tooltip="Letters, digits, - or _ ; e.g. pump-02")
                self._a_x = FloatField("X (m)", default_value=4.0, step=0.5)
                self._a_y = FloatField("Y (m)", default_value=-4.0, step=0.5)
                self._a_yaw = FloatField("Yaw (deg)", default_value=0.0, step=15.0,
                                         tooltip="Rotation about Z. The approach side of the type turns with it. "
                                                 "Roll / pitch / scale: set them in the layout JSON.")
                self._a_add_btn = Button("Add Asset", "ADD ASSET", on_click_fn=self._on_add_asset,
                                         tooltip="Creates the asset (on the stage at once if the scene is loaded) and saves the layout")
                self._a_dd = DropDown("Asset", populate_fn=self._asset_ids, on_selection_fn=self._set("_a_sel"))
                self._a_remove_btn = Button("Remove Asset", "REMOVE ASSET", on_click_fn=self._on_remove_asset,
                                            tooltip="Not possible while the asset has an open incident or a robot working on it")
                self._layout_field = StringField("Floor Layout File", default_value=sc.settings.layout_path, use_folder_picker=True,
                                                 item_filter_fn=_ext_filter((".json",)),
                                                 tooltip="floor_layout.json: every asset (type, position, orientation) and every robot (model, mode, home, yaw)")
                self._layout_load_btn = Button("Load Floor Layout", "LOAD FLOOR LAYOUT", on_click_fn=self._on_load_layout,
                                               tooltip="Replaces all assets with the ones in the file and adds its robots that are not in the scene yet "
                                                       "(not while robots work on incidents)")
                self._layout_save_btn = Button("Save Floor Layout", "SAVE FLOOR LAYOUT", on_click_fn=self._on_save_layout,
                                               tooltip="Writes the current assets AND robots (with home position and yaw) to the file")
                self._site_field = StringField("Site Map File", default_value=sc.settings.site.path, use_folder_picker=True,
                                               item_filter_fn=_ext_filter((".json",)),
                                               tooltip="site_map.json: walkable_area.polygon and no_go_zones")
                self._site_load_btn = Button("Load Site Map", "LOAD SITE MAP", on_click_fn=self._on_load_site,
                                             tooltip="Robots then plan only inside the walkable area and around no-go zones")
                self._site_cb = CheckBox("Show Site Map", default_value=sc.settings.site.show, on_click_fn=self._on_show_site,
                                         tooltip="Green outline = walkable area, red = no-go zones (display only)")
                self._a_msg = TextBlock("Result", num_lines=5, include_copy_button=COPY_BUTTONS)
                self._a_list = TextBlock("Assets", num_lines=6, include_copy_button=COPY_BUTTONS)
                self.wrapped_ui_elements += [self._a_type_dd, self._a_id, self._a_x, self._a_y, self._a_yaw, self._a_add_btn,
                                             self._a_dd, self._a_remove_btn, self._layout_field, self._layout_load_btn,
                                             self._layout_save_btn, self._site_field, self._site_load_btn, self._site_cb]
                self._a_type_dd.repopulate()

    def _build_run_frame(self) -> None:
        s = get_settings()
        with CollapsableFrame("Run Fleet", collapsed=False):
            with ui.VStack(style=get_style(), spacing=5, height=0):
                self._run_btn = StateButton("Run Fleet", "RUN", "STOP", on_a_click_fn=self._on_run,
                                            on_b_click_fn=self._timeline.pause, physics_callback_fn=self._on_physics)
                self._auto_cb = CheckBox("Auto Dispatch", default_value=True, on_click_fn=self._on_auto)
                self._nemo_cb = CheckBox("Use Nemotron", default_value=True, on_click_fn=self._on_nemotron,
                                         tooltip="Off: deterministic rule-based assigner (same contract)")
                self._frames_cb = CheckBox("Capture Quadruped Camera", default_value=True, on_click_fn=self._on_frames,
                                           tooltip="A quadruped reads its own camera when it reaches an asset; the frame is saved as evidence")
                self._light_cb = CheckBox("Camera Light", default_value=s.camera_light, on_click_fn=self._on_camera_light,
                                          tooltip="A small spot light on each robot camera so the inspected face of the asset is lit")
                self._idle_cb = CheckBox("Skip Policy When Idle", default_value=self._scenario.idle_hold, on_click_fn=self._on_idle_hold,
                                         tooltip="Faster simulation: an idle walking robot keeps its standing pose without running "
                                                 "its walking network; it resumes at once for the next task. Robot cameras "
                                                 "render only when a capture is near.")
                self._routine_cb = CheckBox("Routine Inspections", default_value=self._scenario.routine_enabled,
                                            on_click_fn=self._on_routine,
                                            tooltip="OFF (default): robots only react to alarms and events. ON: every asset is also "
                                                    "inspected on a timer that counts running time only.")
                self._routine_min = FloatField("Routine Every (min)", default_value=self._scenario.routine_minutes, step=5.0,
                                               tooltip="Minutes of running time between routine inspections of an asset (1 to 1440).")
                self._run_msg = TextBlock("Run", num_lines=4, include_copy_button=COPY_BUTTONS)
                self.wrapped_ui_elements += [self._run_btn, self._auto_cb, self._nemo_cb, self._frames_cb, self._light_cb,
                                             self._idle_cb, self._routine_cb, self._routine_min]

    def _build_manual_frame(self) -> None:
        with CollapsableFrame("Manual Control", collapsed=True):
            with ui.VStack(style=get_style(), spacing=5, height=0):
                self._robot_dd = DropDown("Robot", populate_fn=self._robot_ids, on_selection_fn=self._set("_sel_robot"))
                self._skill_dd = DropDown("Skill", populate_fn=lambda: list(SKILLS), on_selection_fn=self._set("_sel_skill"))
                self._asset_dd = DropDown("Target Asset", populate_fn=self._asset_ids, on_selection_fn=self._set("_sel_asset"))
                self._dispatch_btn = Button("Dispatch", "DISPATCH", on_click_fn=self._on_dispatch)
                self._stop_btn = Button("Stop All", "STOP ALL", on_click_fn=self._on_stop_all)
                self._home_btn = Button("All Home", "RETURN HOME", on_click_fn=self._on_all_home)
                self._reset_robot_btn = Button("Recover Robot", "RESET ROBOT", on_click_fn=self._on_reset_robot,
                                               tooltip="Fallen / stuck simulated robot: back to home in the standing pose")
                self._align_btn = Button("Align Twin", "ALIGN TWIN", on_click_fn=self._on_align,
                                         tooltip="Real robot (mirror): stand it on its home spot, facing its home yaw, then press. "
                                                 "Computes Map -> World and saves it to the floor layout.")
                self._jog_fwd = Button("Jog Real Robot", "FORWARD 15 cm", on_click_fn=lambda: self._on_jog("forward"),
                                       tooltip="0.25 m/s for 0.6 s, then stop. Refused if the end point is not walkable.")
                self._jog_back = Button("", "BACK 15 cm", on_click_fn=lambda: self._on_jog("back"))
                self._jog_left = Button("", "TURN LEFT 15 deg", on_click_fn=lambda: self._on_jog("turn_left"))
                self._jog_right = Button("", "TURN RIGHT 15 deg", on_click_fn=lambda: self._on_jog("turn_right"))
                self._real_stop = Button("Real Robot", "STOP REAL ROBOT", on_click_fn=self._on_real_stop,
                                         tooltip="Zero velocity to the real robot now. The hand remote always overrides.")
                self.wrapped_ui_elements += [self._robot_dd, self._skill_dd, self._asset_dd, self._dispatch_btn,
                                             self._stop_btn, self._home_btn, self._reset_robot_btn, self._align_btn,
                                             self._jog_fwd, self._jog_back, self._jog_left, self._jog_right, self._real_stop]
                self._robot_dd.repopulate()
                self._skill_dd.repopulate()
                self._asset_dd.repopulate()

    def _build_demo_frame(self) -> None:
        with CollapsableFrame("Demo Events", collapsed=False):
            with ui.VStack(style=get_style(), spacing=5, height=0):
                self._demo_asset_dd = DropDown("Asset", populate_fn=self._asset_ids, on_selection_fn=self._set("_demo_asset"),
                                               tooltip="The asset the fault is injected on")
                self._fault_dd = DropDown("Fault Type", populate_fn=lambda: FAULT_KINDS, on_selection_fn=self._set("_sel_fault"))
                self._prio_dd = DropDown("Priority", populate_fn=lambda: DEMO_PRIORITIES, on_selection_fn=self._set("_sel_prio"),
                                         tooltip="Incident priority: Nemotron serves high before medium before low")
                self._inject_btn = Button("Inject Fault", "INJECT FAULT", on_click_fn=self._on_inject,
                                          tooltip="Raises a sensor alarm on the asset's primary reading. A quadruped is sent to "
                                                  "read it; that reading confirms (or clears) the fault. Needs RUN.")
                self._repair_btn = Button("Technician Done", "MARK REPAIRED", on_click_fn=self._on_repaired,
                                          tooltip="Clears the fault; Nemotron then assigns a re-inspection (a different robot if possible)")
                self._tools_btn = Button("Tools Loaded", "TOOLS LOADED (MANUAL)", on_click_fn=self._on_tools_loaded,
                                         tooltip="Stand-in for the G1 pick-up while its policy is not ready: the AMR then delivers the tools")
                self._photo_field = StringField("Operator Photo", default_value="", use_folder_picker=True,
                                                item_filter_fn=_ext_filter((".jpg", ".jpeg", ".png")),
                                                tooltip="Fallback only: labelled provenance=operator_phone")
                self._photo_btn = Button("Submit Photo", "SUBMIT PHOTO", on_click_fn=self._on_photo)
                self._demo_msg = TextBlock("Last Result", num_lines=5, include_copy_button=COPY_BUTTONS)
                self.wrapped_ui_elements += [self._demo_asset_dd, self._fault_dd, self._prio_dd, self._inject_btn,
                                             self._repair_btn, self._tools_btn, self._photo_field, self._photo_btn]
                self._demo_asset_dd.repopulate()
                self._fault_dd.repopulate()
                self._prio_dd.repopulate()

    def _build_approval_frame(self) -> None:
        with CollapsableFrame("Approvals", collapsed=False):
            with ui.VStack(style=get_style(), spacing=5, height=0):
                self._approval_text = TextBlock("Pending", num_lines=3)
                self._approve_btn = Button("Approve Oldest", "APPROVE", on_click_fn=lambda: self._on_approval(True))
                self._reject_btn = Button("Reject Oldest", "REJECT", on_click_fn=lambda: self._on_approval(False))
                self.wrapped_ui_elements += [self._approve_btn, self._reject_btn]

    def _build_status_frame(self) -> None:
        with CollapsableFrame("Fleet Status", collapsed=False):
            with ui.VStack(style=get_style(), spacing=5, height=0):
                self._status_text = TextBlock("Fleet", num_lines=12, include_copy_button=COPY_BUTTONS)
        with CollapsableFrame("Audit Log", collapsed=False):
            with ui.VStack(style=get_style(), spacing=5, height=0):
                self._log_text = TextBlock("Latest", num_lines=12, include_copy_button=COPY_BUTTONS)

    # ------------------------------------------------------------------ callbacks
    def _on_init(self) -> None:
        s = get_settings()
        self._scenario = FleetScenario()
        self._scenario.on_refresh = self._refresh
        self._sel_model, self._sel_mode, self._walking = "go2", "simulation", True
        self._sel_robot = self._sel_skill = self._sel_asset = self._sel_fault = None
        self._demo_asset = self._sel_prio = self._a_sel = None
        types = self._scenario.asset_types()
        self._a_type = next(iter(types.values()), "")
        self._brain_note = self._demo_note = self._asset_note = self._run_note = ""
        self._ping_sub = None
        self._ping_shown = True
        self._cosmos_sub = None
        self._cosmos_shown = True
        self._provider = self._scenario.settings.ai_provider or ai_profiles.load_file()["active"]
        self._env_usd, self._add_ground, self._env_obstacles = s.environment_usd, s.add_ground_plane, s.environment_obstacles
        self._default_policy = ("", "")

    def _set(self, attr: str):
        return lambda value: setattr(self, attr, value)

    def _robot_ids(self) -> list:
        c = self._scenario.coordinator
        return list(c.fleet.robots) if c and c.fleet.robots else [NO_ITEMS]

    def _asset_ids(self) -> list:
        return self._scenario.asset_ids() or [NO_ITEMS]

    def _type_labels(self) -> list:
        return list(self._scenario.asset_types()) or [NO_ITEMS]

    @staticmethod
    def _pick(selected, items: list):
        return selected if selected in items else (items[0] if items else None)

    def _next_id(self) -> str:
        prefix = self._sel_model
        n = 1 + sum(1 for rid in self._robot_ids() if rid.startswith(prefix + "-"))
        return f"{prefix}-{n}"

    def _fill_policy_fields(self) -> None:
        if not hasattr(self, "_policy_field"):
            return
        info = self._scenario.default_policy_info(self._sel_model)
        self._default_policy = (info["policy"], info["env"])
        self._policy_field.set_value(info["policy"] or "(not used)")
        self._env_yaml_field.set_value(info["env"] or "(not used)")
        self._builder_msg.set_text(f"{MODELS[self._sel_model].label}: {info['note']}")

    def _refresh_asset_widgets(self) -> None:
        for dd, attr in ((getattr(self, "_asset_dd", None), "_sel_asset"), (getattr(self, "_demo_asset_dd", None), "_demo_asset"),
                         (getattr(self, "_a_dd", None), "_a_sel")):
            if dd is not None:
                dd.repopulate()
                setattr(self, attr, None)
        self._render_assets()

    def _render_assets(self) -> None:
        if not hasattr(self, "_a_list"):
            return
        c = self._scenario.coordinator
        lines = []
        for aid, a in self._scenario.settings.assets.items():
            inc = c.fleet.incident_for(aid) if c else None
            tag = f"  [{inc.status}{', ' + inc.priority if inc and inc.priority else ''}]" if inc else ""
            lines.append(f"{aid}  {a.label}  ({a.position[0]:.1f}, {a.position[1]:.1f}) yaw {a.orientation_deg[2]:g}  "
                         f"{a.primary or 'reading'} {a.normal_low:g}-{a.normal_high:g} {a.unit}{tag}")
        self._a_list.set_text("\n".join(lines) or "No assets - add one above or LOAD LAYOUT.")
        sc = self._scenario
        extra = [f"Gauges: {p}" for p in sc.gauge_problems[-3:]]
        self._a_msg.set_text("\n".join(([self._asset_note] if self._asset_note else []) + extra
                                       + [f"Site: {sc.work_area()}", f"Input files: {inputs_dir()}"]))

    def _on_loaded(self) -> None:
        self._scenario.setup_post_load()
        self._reset_btn.enabled = self._run_btn.enabled = self._add_btn.enabled = self._remove_btn.enabled = True
        self._brain_now_btn.enabled = True
        self._run_btn.reset()
        self._robot_dd.repopulate()
        self._refresh_asset_widgets()
        robots = getattr(self._scenario, "layout_robots_note", "")
        self._robot_dd.repopulate()
        self._builder_msg.set_text(f"Scene loaded. Environment: {self._scenario.env_info}. "
                                   f"{self._scenario.physics_status()}. " + (robots + " " if robots else "Add robots, then ") +
                                   "press RUN.")
        self._refresh()

    def _on_reset(self) -> None:
        self._scenario.reset()
        self._run_note = "Reset: robots are back home and every pending task was cleared."
        self._run_btn.reset()
        self._run_btn.enabled = True
        self._refresh_asset_widgets()
        self._refresh()

    def _on_physics(self, step: float, context=None) -> None:
        self._scenario.update(step)

    # --- run
    def _on_run(self) -> None:
        msg = self._scenario.apply_routine(self._scenario.routine_enabled, self._routine_min.get_value())
        if msg.startswith("Routine interval"):
            self._run_note = msg
        note = self._scenario.start_run()
        if note:
            self._run_note = note
        self._timeline.play()
        self._refresh()

    def _on_routine(self, value: bool) -> None:
        self._run_note = self._scenario.apply_routine(value, self._routine_min.get_value())
        self._refresh()

    def _on_idle_hold(self, value: bool) -> None:
        self._run_note = self._scenario.set_idle_hold(value)
        self._refresh()

    def _on_camera_light(self, value: bool) -> None:
        self._run_note = self._scenario.set_camera_light(value)
        self._refresh()

    def _render_run(self) -> None:
        sc = self._scenario
        lines = ([self._run_note] if self._run_note else [])
        state = (f"ON, every {sc.routine_minutes:g} min of running time" if sc.routine_enabled
                 else "OFF (robots react to alarms and events only)")
        lines.append(f"Routine inspections: {state}")
        self._run_msg.set_text("\n".join(lines))

    # --- AI models (v0.5.1)
    def _ai_fields(self) -> dict:
        return {"brain": {"url": self._brain_url.get_value().strip(), "model": self._brain_model.get_value().strip(),
                          "api_key": self._brain_key.get_value_as_string().strip()},
                "vlm": {"url": self._vlm_url.get_value().strip(), "model": self._vlm_model.get_value().strip(),
                        "api_key": self._vlm_key.get_value_as_string().strip()}}

    def _fill_ai_fields(self, prof: dict, from_settings: bool = False) -> None:
        s = self._scenario.settings
        b, v = prof.get("brain") or {}, prof.get("vlm") or {}
        if from_settings:  # start-up: show what is actually in use
            b = {"url": s.nebius_base_url, "model": s.nemotron_model_id, "api_key": s.nebius_api_key}
            v = {"url": s.vlm_base_url, "model": s.vlm_model, "api_key": s.vlm_api_key}
        self._brain_url.set_value(b.get("url", "")); self._brain_model.set_value(b.get("model", ""))
        self._brain_key.set_value(b.get("api_key", ""))
        self._vlm_url.set_value(v.get("url", "")); self._vlm_model.set_value(v.get("model", ""))
        self._vlm_key.set_value(v.get("api_key", ""))

    def _on_provider_selected(self, label: str) -> None:
        self._provider = PROVIDER_LABELS.get(label, self._provider)
        if hasattr(self, "_profile_field"):
            self._fill_ai_fields(self._scenario.profile_fields(self._provider, self._profile_field.get_value()))
            self._brain_note = f"{label}: fields filled from the Profile File - press APPLY or LOAD to use them."
            self._refresh()

    def _on_apply_brain(self) -> None:
        self._brain_note = self._scenario.apply_ai(self._provider, self._ai_fields())
        self._refresh()

    def _on_save_ai(self) -> None:
        self._brain_note = self._scenario.save_ai(self._provider, self._ai_fields(), self._profile_field.get_value())
        if self._brain_note.startswith("Saved"):
            self._fill_ai_fields(self._scenario.profile_fields(self._provider, self._profile_field.get_value()))
        self._refresh()

    def _on_load_ai(self) -> None:
        self._brain_note, prof = self._scenario.load_ai(self._provider, self._profile_field.get_value())
        if prof:
            self._fill_ai_fields(prof)
        self._refresh()

    def _on_forget_ai(self) -> None:
        self._brain_note = self._scenario.forget_ai(self._provider, self._profile_field.get_value())
        self._refresh()

    def _on_test_brain(self) -> None:
        applied = self._scenario.apply_ai(self._provider, self._ai_fields())
        if not self._scenario.settings.nebius_api_key:
            self._brain_note = applied + "\nEnter the Nemotron API key first (or press LOAD)."
            self._refresh()
            return
        self._ping_shown = False
        self._brain_note = self._scenario.start_connection_test()
        self._refresh()
        try:
            import omni.kit.app
            stream = omni.kit.app.get_app().get_update_event_stream()
            self._ping_sub = stream.create_subscription_to_pop(self._on_update_tick, name="physical_ai_fleet_connection_test")
        except Exception:  # noqa: BLE001
            self._ping_sub = None

    def _on_update_tick(self, _event) -> None:
        if self._show_connection_result():
            self._ping_sub = None

    def _show_connection_result(self) -> bool:
        r = self._scenario.connection_result
        if r is None or self._ping_shown:
            return self._ping_shown
        self._ping_shown = True
        self._brain_note = ("OK - " if r["ok"] else "FAILED - ") + r["message"]
        self._render_brain()
        return True

    def _on_test_cosmos(self) -> None:
        applied = self._scenario.apply_ai(self._provider, self._ai_fields())  # test what is in the fields
        if applied.startswith("Not applied"):
            self._brain_note = applied
            self._refresh()
            return
        self._cosmos_shown = False
        self._brain_note = self._scenario.start_cosmos_test(self._cosmos_img_field.get_value())
        self._refresh()
        try:
            import omni.kit.app
            stream = omni.kit.app.get_app().get_update_event_stream()
            self._cosmos_sub = stream.create_subscription_to_pop(self._on_cosmos_tick, name="physical_ai_fleet_vlm_test")
        except Exception:  # noqa: BLE001
            self._cosmos_sub = None

    def _on_cosmos_tick(self, _event) -> None:
        if self._show_cosmos_result():
            self._cosmos_sub = None

    def _show_cosmos_result(self) -> bool:
        r = self._scenario.cosmos_test_result
        if r is None or self._cosmos_shown:
            return self._cosmos_shown
        self._cosmos_shown = True
        self._brain_note = "VLM test: " + r
        self._render_brain()
        return True

    def _on_ask_now(self) -> None:
        self._brain_note = self._scenario.ask_nemotron_now()
        self._refresh()

    def _render_brain(self) -> None:
        c = self._scenario.coordinator
        lines = ([self._brain_note] if self._brain_note else [])
        lines.append(f"Provider: {self._scenario.settings.ai_provider or PROVIDER_KEYS.get(self._provider, self._provider)}")
        lines.append("Nemotron: " + self._scenario.settings.nemotron_status())
        if c:
            lines.append(c.brain.health_line())
        lines.append(self._scenario.cosmos_status())
        self._brain_msg.set_text("\n".join(lines))

    def _render_demo(self) -> None:
        evidence = self._scenario.last_evidence()
        self._demo_msg.set_text("\n".join(([self._demo_note] if self._demo_note else [])
                                          + ([f"Latest camera evidence: {evidence}"] if evidence else [])))

    # --- fleet builder
    def _on_model_selected(self, label: str) -> None:
        self._sel_model = MODEL_LABELS[label]
        if hasattr(self, "_usd_field"):
            self._usd_field.set_value(MODELS[self._sel_model].usd)
            self._id_field.set_value(self._next_id())
            self._fill_policy_fields()

    def _on_mode_selected(self, label: str) -> None:
        self._sel_mode = MODE_LABELS[label]

    def _policy_override(self) -> tuple:
        pt, env = self._policy_field.get_value().strip(), self._env_yaml_field.get_value().strip()
        if pt.startswith("(") or env.startswith("("):
            return "", ""
        if MODELS[self._sel_model].custom_policy or (pt, env) != self._default_policy:
            return pt, env
        return "", ""

    def _on_add_robot(self) -> None:
        pt, env = self._policy_override()
        msg, loaded = self._scenario.add_robot(
            self._id_field.get_value(), self._sel_model, self._sel_mode, self._usd_field.get_value(),
            self._conn_field.get_value(), self._frame_field.get_value(),
            (self._x_field.get_value(), self._y_field.get_value(), self._yaw_field.get_value()), walking=self._walking,
            policy_path=pt, env_path=env)
        if loaded:
            self._policy_field.set_value(loaded[0])
            self._env_yaml_field.set_value(loaded[1])
            msg += f"\nPolicy: {loaded[0]}\nEnv:    {loaded[1]}"
            if (pt, env) == ("", ""):
                self._default_policy = tuple(loaded)
        self._builder_msg.set_text(msg)
        if msg.startswith("Added"):
            self._x_field.set_value(self._x_field.get_value() + (2.0 if self._sel_model in ("spot", "h1") else 1.5))
            self._id_field.set_value(self._next_id())
        self._robot_dd.repopulate()
        self._refresh()

    def _on_remove_robot(self) -> None:
        self._builder_msg.set_text(self._scenario.remove_robot(self._sel_robot or ""))
        self._robot_dd.repopulate()
        self._refresh()

    # --- assets & site
    def _on_type_selected(self, label: str) -> None:
        self._a_type = self._scenario.asset_types().get(label, self._a_type)
        if hasattr(self, "_a_id"):
            self._a_id.set_value(self._scenario.next_asset_id(self._a_type))

    def _on_add_asset(self) -> None:
        try:
            msg = self._scenario.add_asset(self._a_type, self._a_id.get_value(), self._a_x.get_value(), self._a_y.get_value(),
                                           self._a_yaw.get_value())
        except Exception as exc:  # noqa: BLE001 - a button handler must never fail silently
            traceback.print_exc()
            msg = f"Cannot add asset: unexpected error {type(exc).__name__}: {exc} (details in the Console)"
        self._asset_note = msg
        if msg.startswith("Added"):
            self._refresh_asset_widgets()
            self._layout_field.set_value(self._scenario.settings.layout_path)
            self._a_id.set_value(self._scenario.next_asset_id(self._a_type))
            self._a_y.set_value(self._a_y.get_value() + 2.5)
        self._refresh()

    def _on_remove_asset(self) -> None:
        asset = self._pick(self._a_sel, self._asset_ids())
        if not asset or asset == NO_ITEMS:
            self._asset_note = "No asset to remove."
        else:
            self._asset_note = self._scenario.remove_asset(asset)
            if self._asset_note.startswith("Removed"):
                self._refresh_asset_widgets()
                self._a_id.set_value(self._scenario.next_asset_id(self._a_type))
        self._refresh()

    def _on_load_layout(self) -> None:
        self._asset_note = self._scenario.load_layout(self._layout_field.get_value())
        if self._asset_note.startswith("Loaded"):
            self._layout_field.set_value(self._scenario.settings.layout_path)
            self._refresh_asset_widgets()
            self._robot_dd.repopulate()
        self._refresh()

    def _on_save_layout(self) -> None:
        self._asset_note = self._scenario.save_layout(self._layout_field.get_value())
        if self._asset_note.startswith("Saved"):
            self._layout_field.set_value(self._scenario.settings.layout_path)
        self._refresh()

    def _on_load_site(self) -> None:
        self._asset_note = self._scenario.load_site(self._site_field.get_value())
        self._refresh()

    def _on_show_site(self, value: bool) -> None:
        self._asset_note = self._scenario.set_site_visible(value)
        self._refresh()

    # --- run / manual
    def _on_auto(self, value: bool) -> None:
        if self._scenario.coordinator:
            self._scenario.coordinator.auto_dispatch = value

    def _on_nemotron(self, value: bool) -> None:
        if self._scenario.coordinator:
            self._scenario.coordinator.assigner.use_nemotron = value

    def _on_frames(self, value: bool) -> None:
        self._scenario.save_sim_frames = value

    def _selected_robot_ok(self) -> bool:
        if not self._scenario.coordinator or not self._sel_robot or self._sel_robot == NO_ITEMS:
            self._builder_msg.set_text("Load the scene and add/select a robot first.")
            return False
        return True

    def _on_dispatch(self) -> None:
        if not self._selected_robot_ok():
            return
        asset = self._pick(self._sel_asset, self._asset_ids())
        result = self._scenario.coordinator.operator_dispatch(self._sel_robot, self._sel_skill, asset if asset != NO_ITEMS else None)
        self._builder_msg.set_text(f"{self._sel_skill} -> {self._sel_robot}: {result}")
        self._refresh()

    def _on_stop_all(self) -> None:
        if self._scenario.coordinator:
            self._scenario.coordinator.stop_all()
        self._refresh()

    def _on_all_home(self) -> None:
        c = self._scenario.coordinator
        if c:
            for rid in list(c.fleet.robots):
                c.operator_dispatch(rid, "return_home", None)
        self._refresh()

    def _on_align(self) -> None:
        if self._selected_robot_ok():
            self._builder_msg.set_text(self._scenario.align_twin(self._sel_robot))
            self._refresh()

    def _on_jog(self, kind: str) -> None:
        if self._selected_robot_ok():
            self._builder_msg.set_text(self._scenario.jog_robot(self._sel_robot, kind))
            self._refresh()

    def _on_real_stop(self) -> None:
        if self._selected_robot_ok():
            self._builder_msg.set_text(self._scenario.stop_real_robot(self._sel_robot))
            self._refresh()

    def _on_reset_robot(self) -> None:
        if self._selected_robot_ok():
            self._builder_msg.set_text(self._scenario.reset_robot(self._sel_robot))
            self._refresh()

    # --- demo events
    def _demo_target(self):
        asset = self._pick(self._demo_asset, self._asset_ids())
        return None if asset in (None, NO_ITEMS) else asset

    def _on_inject(self) -> None:
        asset = self._demo_target()
        if asset is None:
            self._demo_note = "Add an asset first (Assets & Site panel)."
            self._refresh()
            return
        fault = self._pick(self._sel_fault, FAULT_KINDS)
        prio = self._pick(self._sel_prio, DEMO_PRIORITIES)
        msg = self._scenario.inject_fault(asset, fault, prio)
        if self._scenario.coordinator and not self._timeline.is_playing():
            msg += "\nPress RUN so the fleet can react."
        self._demo_note = msg
        self._refresh()

    def _on_repaired(self) -> None:
        asset = self._demo_target()
        self._demo_note = self._scenario.mark_repaired(asset) if asset else "Add an asset first."
        self._refresh()

    def _on_tools_loaded(self) -> None:
        asset = self._demo_target()
        self._demo_note = self._scenario.mark_tools_loaded(asset) if asset else "Add an asset first."
        self._refresh()

    def _on_photo(self) -> None:
        jpeg = load_operator_photo(self._photo_field.get_value())
        if not jpeg:
            self._demo_note = "Photo file not found."
        else:
            asset = self._demo_target()
            self._demo_note = self._scenario.submit_operator_photo(asset, jpeg) if asset else "Add an asset first."
        self._refresh()

    def _on_approval(self, approved: bool) -> None:
        c = self._scenario.coordinator
        if not c:
            return
        a, _ev = c.approvals.oldest()
        if a is None:
            self._approval_text.set_text("Nothing pending.")
            return
        self._builder_msg.set_text(f"{a.skill} -> {a.robot_id}: {c.approve(a.assignment_id, approved)}")
        self._refresh()

    # ------------------------------------------------------------------ status
    def _refresh(self) -> None:
        if not hasattr(self, "_status_text"):
            return
        s = self._scenario.settings
        c = self._scenario.coordinator
        self._show_connection_result()
        self._show_cosmos_result()
        self._render_brain()
        self._render_assets()
        self._render_demo()
        self._render_run()
        if not c:
            self._status_text.set_text("Press LOAD to create the scene and start the coordinator.\n"
                                       f"Nemotron: {s.nemotron_status()}")
            return
        header = [
            self._scenario.physics_status(),
            self._scenario.perf_status(),
            f"Environment: {self._scenario.env_info}",
            f"Site: {s.site.describe()}",
            f"Nemotron: {s.nemotron_status()}",
            self._scenario.cosmos_status(),
        ]
        self._status_text.set_text("\n".join(header + c.status_lines()))
        self._approval_text.set_text(
            "\n".join(f"{a.assignment_id}: {a.skill} -> {a.robot_id} ({a.source}) {a.rationale[:60]}"
                      for a in c.approvals.pending()) or "Nothing pending.")
        self._log_text.set_text("\n".join(AuditLog.one_line(e) for e in reversed(c.audit.recent(12))))

    def _reset_extension(self) -> None:
        self._scenario.teardown()
        if hasattr(self, "_run_btn"):
            self._run_btn.reset()
            self._run_btn.enabled = self._reset_btn.enabled = False
            self._add_btn.enabled = self._remove_btn.enabled = False
            self._brain_now_btn.enabled = False
            self._robot_dd.repopulate()
            self._refresh()

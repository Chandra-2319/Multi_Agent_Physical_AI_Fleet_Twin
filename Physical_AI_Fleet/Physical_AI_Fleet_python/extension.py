# SPDX-License-Identifier: Apache-2.0
# Based on the NVIDIA Isaac Sim extension template (Apache-2.0).
"""Extension entry point for Physical_AI_Fleet (boilerplate from the Loaded_scenario template)."""
import asyncio
import gc

import carb.eventdispatcher
import omni
import omni.kit.commands
import omni.physics.core
import omni.timeline
import omni.ui as ui
import omni.usd
from isaacsim.gui.components.element_wrappers import ScrollingWindow
from isaacsim.gui.components.menu import MenuItemDescription
from omni.kit.menu.utils import add_menu_items, remove_menu_items

from .global_variables import EXTENSION_TITLE
from .ui_builder import UIBuilder


class Extension(omni.ext.IExt):
    """Adds the Physical_AI_Fleet window to the Isaac Sim toolbar menu."""

    def on_startup(self, ext_id: str) -> None:
        self.ext_id = ext_id
        self._usd_context = omni.usd.get_context()
        self._window = ScrollingWindow(title=EXTENSION_TITLE, width=660, height=900, visible=False,
                                       dockPreference=ui.DockPreference.LEFT_BOTTOM)
        self._window.set_visibility_changed_fn(self._on_window)
        action_registry = omni.kit.actions.core.get_action_registry()
        action_registry.register_action(ext_id, f"CreateUIExtension:{EXTENSION_TITLE}", self._menu_callback,
                                        description=f"Add {EXTENSION_TITLE} Extension to UI toolbar")
        self._menu_items = [MenuItemDescription(name=EXTENSION_TITLE,
                                                onclick_action=(ext_id, f"CreateUIExtension:{EXTENSION_TITLE}"))]
        add_menu_items(self._menu_items, EXTENSION_TITLE)
        self.ui_builder = UIBuilder()
        self._physics_simulation_interface = omni.physics.core.get_physics_simulation_interface()
        self._physics_subscription = None
        self._timeline = omni.timeline.get_timeline_interface()

    def on_shutdown(self) -> None:
        remove_menu_items(self._menu_items, EXTENSION_TITLE)
        omni.kit.actions.core.get_action_registry().deregister_action(self.ext_id, f"CreateUIExtension:{EXTENSION_TITLE}")
        self._physics_subscription = None
        self._window = None
        self.ui_builder.shutdown()
        gc.collect()

    def _on_window(self, visible: bool) -> None:
        if self._window.visible:
            self._usd_context = omni.usd.get_context()
            dispatcher = carb.eventdispatcher.get_eventdispatcher()
            self._stage_event_sub_opened = dispatcher.observe_event(
                event_name=self._usd_context.stage_event_name(omni.usd.StageEventType.OPENED),
                on_event=self._on_stage_opened, observer_name="physical_ai_fleet._on_stage_opened")
            self._stage_event_sub_closed = dispatcher.observe_event(
                event_name=self._usd_context.stage_event_name(omni.usd.StageEventType.CLOSED),
                on_event=self._on_stage_closed, observer_name="physical_ai_fleet._on_stage_closed")
            self._timeline_event_sub_play = dispatcher.observe_event(
                event_name=omni.timeline.GLOBAL_EVENT_PLAY, on_event=self._on_timeline_play,
                observer_name="physical_ai_fleet._on_timeline_play")
            self._timeline_event_sub_stop = dispatcher.observe_event(
                event_name=omni.timeline.GLOBAL_EVENT_STOP, on_event=self._on_timeline_stop,
                observer_name="physical_ai_fleet._on_timeline_stop")
            self._build_ui()
        else:
            self._usd_context = None
            self._stage_event_sub_opened = self._stage_event_sub_closed = None
            self._timeline_event_sub_play = self._timeline_event_sub_stop = None
            self.ui_builder.cleanup()

    def _build_ui(self) -> None:
        with self._window.frame:
            with ui.VStack(spacing=5, height=0):
                self.ui_builder.build_ui()

        async def dock_window() -> None:
            await omni.kit.app.get_app().next_update_async()
            tgt = ui.Workspace.get_window("Viewport")
            window = omni.ui.Workspace.get_window(EXTENSION_TITLE)
            if window and tgt:
                window.dock_in(tgt, omni.ui.DockPosition.LEFT, 0.33)
            await omni.kit.app.get_app().next_update_async()

        self._task = asyncio.ensure_future(dock_window())

    def _menu_callback(self) -> None:
        self._window.visible = not self._window.visible
        self.ui_builder.on_menu_callback()

    def _on_timeline_play(self, event) -> None:
        if not self._physics_subscription:
            self._physics_subscription = self._physics_simulation_interface.subscribe_physics_on_step_events(
                pre_step=False, order=0, on_update=self._on_physics_step)
        self.ui_builder.on_timeline_event(event)

    def _on_timeline_stop(self, event) -> None:
        self._physics_subscription = None
        self.ui_builder.on_timeline_event(event)

    def _on_physics_step(self, step, context) -> None:
        self.ui_builder.on_physics_step(step)

    def _on_stage_opened(self, event) -> None:
        self._physics_subscription = None
        self.ui_builder.cleanup()
        self.ui_builder.on_stage_event(event)

    def _on_stage_closed(self, event) -> None:
        self._physics_subscription = None
        self.ui_builder.cleanup()
        self.ui_builder.on_stage_event(event)

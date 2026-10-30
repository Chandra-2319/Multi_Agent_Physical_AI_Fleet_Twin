#!/bin/bash
# Unitree Go2 multi-site map/graph switcher
# Usage:
#   ./site_manager.sh save <sitename>
#   ./site_manager.sh load <sitename>
#   ./site_manager.sh list
#   ./site_manager.sh current

set -e

WS_ROOT="/unitree/module/graph_pid_ws"
SITES_ROOT="$HOME/unitree_sites"

PCD_DIR="$WS_ROOT/config_files/lio_sam_config/maps/pcd/default"
GRAPH_MAP_DIR="$WS_ROOT/src/graph_process/graph_map"

if [ ! -d "$PCD_DIR" ] || [ ! -d "$GRAPH_MAP_DIR" ]; then
    echo "Expected directory missing:"
    echo "  PCD_DIR       = $PCD_DIR"
    echo "  GRAPH_MAP_DIR = $GRAPH_MAP_DIR"
    exit 1
fi

mkdir -p "$SITES_ROOT"
cmd="$1"
site="$2"

stack_running() {
    pgrep -f "main_process_multifloor" > /dev/null || pgrep -f "lio_sam_ros2_globalLocalize" > /dev/null
}

case "$cmd" in
  save)
    [ -z "$site" ] && { echo "Usage: $0 save <sitename>"; exit 1; }
    dest="$SITES_ROOT/$site"
    mkdir -p "$dest"
    echo "Saving current PCD map + graph into site '$site'..."
    rm -rf "$dest/pcd" "$dest/graph_map"
    cp -a "$PCD_DIR" "$dest/pcd"
    cp -a "$GRAPH_MAP_DIR" "$dest/graph_map"
    echo "$site" > "$SITES_ROOT/.current"
    echo "Saved:"
    echo "  $dest/pcd        (from $PCD_DIR)"
    echo "  $dest/graph_map  (from $GRAPH_MAP_DIR)"
    ;;

  load)
    [ -z "$site" ] && { echo "Usage: $0 load <sitename>"; exit 1; }
    src="$SITES_ROOT/$site"
    if [ ! -d "$src" ]; then
        echo "No saved site '$site' found in $SITES_ROOT"
        echo "Available sites:"
        ls -1 "$SITES_ROOT" 2>/dev/null | grep -v '^\.current$'
        exit 1
    fi
    if stack_running; then
        echo "SLAM stack looks like it's still running (main_process_multifloor / lio_sam_ros2_globalLocalize detected)."
        echo "Stop it first — Ctrl+C the 0_unitree_slam.sh launch — then rerun this."
        exit 1
    fi
    echo "Loading site '$site' into the live PCD + graph paths..."
    rm -rf "${PCD_DIR:?}"/*
    cp -a --no-preserve=timestamps "$src/pcd/." "$PCD_DIR/"
    rm -rf "${GRAPH_MAP_DIR:?}"/*
    cp -a --no-preserve=timestamps "$src/graph_map/." "$GRAPH_MAP_DIR/"
    echo "$site" > "$SITES_ROOT/.current"
    echo "Done. Now start 0_unitree_slam.sh — it will boot with site '$site''s map and graph."
    ;;

  list)
    echo "Saved sites in $SITES_ROOT:"
    ls -1 "$SITES_ROOT" 2>/dev/null | grep -v '^\.current$' || echo "  (none yet)"
    ;;

  current)
    if [ -f "$SITES_ROOT/.current" ]; then
        echo "Last loaded/saved site: $(cat "$SITES_ROOT/.current")"
    else
        echo "No site has been saved or loaded through this script yet."
    fi
    ;;

  *)
    echo "Usage: $0 {save|load|list|current} [sitename]"
    exit 1
    ;;
esac

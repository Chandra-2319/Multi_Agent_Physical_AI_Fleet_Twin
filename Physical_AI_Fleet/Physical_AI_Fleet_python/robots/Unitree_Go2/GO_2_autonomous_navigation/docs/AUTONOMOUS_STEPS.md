# Autonomous Navigation: Run Guide

Two workflows are covered:

- **[Scenario A](#scenario-a--mapping-a-brand-new-site)**: map a new place and define its waypoint graph
- **[Scenario B](#scenario-b--returning-to-a-previously-saved-site)**: reload a saved site and navigate (the hackathon demo)

## Prerequisites

- The Go2 is powered on and reachable over SSH (`ssh unitree@<ROBOT_IP>`).
- The LiDAR is connected and the onboard NIC is named `eth0`.
- `~/site_manager.sh` and `~/cyclonedds_wifi_test.xml` are installed (see the README install section).
- You have two SSH terminals open to the robot.

> **Critical rule:** never press `w` (new mapping) before saving whatever site is currently live.
> `w` and `e` overwrite `default/` in place, with no backup and no undo.

---

## Scenario A: Mapping a brand-new site

### Terminal 1: protect the current map, then start the stack

```bash
ssh unitree@<ROBOT_IP>
```

If a site is currently loaded and you have not saved it, save it now:

```bash
~/site_manager.sh save <current_site_name>
```

Skip this only if it is already saved, or if `default/` is empty or irrelevant.

Start the stack fresh:

```bash
export CYCLONEDDS_URI=~/cyclonedds_wifi_test.xml
source /unitree/module/graph_pid_ws/install/setup.bash
bash /unitree/module/graph_pid_ws/0_unitree_slam.sh
```

### Terminal 2: run the mapping client

```bash
ssh unitree@<ROBOT_IP>
export CYCLONEDDS_URI=~/cyclonedds_wifi_test.xml
source /unitree/module/graph_pid_ws/install/setup.bash
/unitree/lib/unitree_slam/rocky/build/demo_xt16_replan_manual_goal eth0
```

At the keyboard prompt:

| Step | Key | What it does |
|---|---|---|
| 1 | `w` | Start mapping. Walk the robot around the **whole** space before continuing. |
| 2 | `e` | End mapping. Writes `GlobalMap.pcd` etc. into `default/`. |
| 3 | `x` | Add node 1 (stand at node 1's spot first). |
| 4 | `x` | Add node 2 (walk to its spot, then press `x`). Repeat for every node. |
| 5 | `c` | Push the topology to the robot (writes into `graph_map/`). |
| 6 | `q` | Close the demo program. |

### Terminal 1: stop the stack and save the new site

Press `Ctrl+C` to stop the stack, then:

```bash
~/site_manager.sh save <new_site_name>
~/site_manager.sh list        # sanity check: the new site should appear
```

The site is now archived under `~/unitree_sites/<new_site_name>/`.

---

## Scenario B: Returning to a previously saved site

### Terminal 1: load the site and start the stack

```bash
ssh unitree@<ROBOT_IP>
~/site_manager.sh load hackathon        # or any saved site, e.g. Test_01
```

Expected output:

```
Loading site '<saved_site_name>' into the live PCD + graph paths...
Done. Now start 0_unitree_slam.sh — it will boot with site '<saved_site_name>''s map and graph.
```

If the script reports that the SLAM stack is still running, press `Ctrl+C` on the running
`0_unitree_slam.sh` in the other session and retry the load.

Start the stack. It boots already using that site's map and graph:

```bash
export CYCLONEDDS_URI=~/cyclonedds_wifi_test.xml
source /unitree/module/graph_pid_ws/install/setup.bash
bash /unitree/module/graph_pid_ws/0_unitree_slam.sh
```

### Terminal 2: run a navigation client

```bash
ssh unitree@<ROBOT_IP>
export CYCLONEDDS_URI=~/cyclonedds_wifi_test.xml
source /unitree/module/graph_pid_ws/install/setup.bash
```

Pick one:

```bash
# Single goal
/unitree/lib/unitree_slam/rocky/build/demo_xt16_replan_manual_goal eth0

# Multiple goals (priority queue)
/unitree/lib/unitree_slam/rocky/build/demo_xt16_priority_goal eth0
```

Then at the prompt:

```
a                   # relocalize on the restored map (no mapping or node re-entry needed;
                    # waits ~3 s to settle, then prompts you)
Enter goal node: 2
```

The robot then navigates to the goal with the obstacle watchdog active.

---

## Keyboard reference

| Key | Action |
|---|---|
| `w` | Start mapping (**overwrites live map**) |
| `e` | End mapping |
| `x` | Add a node at the robot's current position |
| `c` | Push the topology to the robot |
| `a` | Relocalize on the loaded map |
| `<n>` + Enter | Navigate to goal node `n` |
| `s` | Stop |
| `d` | Resume |
| `q` | Quit the demo |

## `site_manager.sh` reference

| Situation | Command |
|---|---|
| List saved sites | `~/site_manager.sh list` |
| Show the last loaded or saved site | `~/site_manager.sh current` |
| Save the current live map and graph as a site | `~/site_manager.sh save <name>` |
| Restore a saved site as the live one | `~/site_manager.sh load <name>` |

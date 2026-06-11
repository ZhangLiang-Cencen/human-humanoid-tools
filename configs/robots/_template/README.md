# Robot preset template

The repo ships two robot references:

| Purpose | Location |
|---------|----------|
| **Bundled example** (Unitree G1) | [`../unitree_g1/`](../unitree_g1/) |
| **Schema / comments for new robots** | [`robot.yaml`](robot.yaml) (this folder) |

Web uploads scaffold `robot.yaml` under `~/.config/hhtools/robots/<name>/` using the same
fields as this template (topology-based `ik_map` inference + default weights).

## New robot checklist

1. **Web**: drop URDF + meshes in the Robot panel, or copy `_template/` → `configs/robots/<name>/` for a bundled preset.
2. Edit `robot.yaml` — pay attention to **gimbal ik_map**, **smooth_joint_filter_masks**, **retarget** warm-up frames (see comments).
3. `hhtools robot validate <name>` (`--fix` to auto-repair ik_map where possible).
4. Web **Calibrate** once per motion reference (`soma_bvh`, `smpl`, `lafan_bvh`, …).
5. Retarget → export CSV / ZIP.

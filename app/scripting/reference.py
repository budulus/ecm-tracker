"""Plain-text reference of the scripting API, generated from the live docstrings."""
import inspect

from app.scripting.session import Session

SECTIONS = [
    ("Loading images", ["open_directory", "open_files", "load_sequence", "open_image_pair",
                        "set_pair_alignment", "open_pair_setup", "save_pair_setup", "adopt_sequence",
                        "validate_sequence"]),
    ("Frames", ["set_frame_range", "set_current_frame"]),
    ("Region of interest", ["set_roi", "set_roi_rectangle", "set_roi_circle", "clear_roi"]),
    ("Parameters", ["set_corner_params", "set_grid_params", "set_lk_params", "set_display_params"]),
    ("Seed points", ["detect_corners", "detect_grid", "set_seeds", "add_seed", "add_seeds",
                     "remove_seeds", "clear_seeds"]),
    ("Tracking", ["track", "clear_tracking"]),
    ("Cleanup", ["cleanup", "cleanup_mask", "remove_points", "apply_keep_mask", "undo_cleanup",
                 "metrics"]),
    ("Results", ["coords", "valid", "point_ids", "frame_indices", "exportable_mask"]),
    ("Export / save", ["export_npy", "export_csv", "save_trackers", "load_trackers"]),
    ("Recording", ["start_recording", "stop_recording", "suspend_recording", "record_command"]),
    ("Events", ["add_listener"]),
]


def _doc(obj) -> str:
    return inspect.cleandoc(obj.__doc__ or "").strip()


def api_reference() -> str:
    """Every documented Session command and property, grouped by workflow step."""
    out = [
        '"""ECM Tracker scripting API (read-only reference; generated from the docstrings).',
        "",
        "`session` is the open project; every call below is also what the macro recorder writes.",
        "Frame indices are global (position in the loaded sequence); tracked arrays are indexed",
        "by cut index on axis 0 (row 0 = reference frame).",
        '"""',
        "",
    ]
    for title, names in SECTIONS:
        out.append(f"# ---- {title} " + "-" * max(4, 80 - len(title)))
        for name in names:
            member = getattr(Session, name)
            signature = str(inspect.signature(member)).replace("(self, ", "(").replace("(self)", "()")
            out.append(f"session.{name}{signature}")
            out.extend(f"    {line}" if line else "" for line in _doc(member).splitlines())
            out.append("")
    properties = sorted(n for n, v in vars(Session).items() if isinstance(v, property))
    out.append("# ---- Read-only properties " + "-" * 56)
    for name in properties:
        first = _doc(getattr(Session, name)).split("\n")[0]
        out.append(f"session.{name}" + (f"  # {first}" if first else ""))
    out.append("")
    out.append("# ---- GUI only (`app`) " + "-" * 60)
    out.append("app.launch_plugin(plugin_id) / app.plugin(plugin_id) / app.plugins()")
    out.append("app.fit_view() / app.zoom_in() / app.zoom_out() / app.save_view(path)")
    out.append("app.message(text) / app.process_events() / app.window / app.canvas")
    return "\n".join(out) + "\n"

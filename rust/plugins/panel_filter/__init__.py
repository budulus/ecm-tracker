"""Panel Filter — example plugin exercising declared control panels (slice 3g-c).

Demonstrates the declarative control-panel API. Instead of building a Qt widget, the plugin
*declares* its controls in panel(self, ui) by calling the host-provided PanelBuilder
(ui.label / ui.slider / ui.checkbox / ui.button). The host renders them as an egui window, owns
the live values, and reports changes back via on_control(self, key, value):

  - slider  -> on_control(key, <float>)
  - checkbox-> on_control(key, <bool>)
  - button  -> on_control(key, None)

This plugin keeps every Nth active point (stride from the slider), or drops them instead when the
"invert" checkbox is set, applying the filter via ctx.apply_keep_mask when the button is clicked —
so the user can Undo it in the Cleanup dialog. numpy-free (keep is a plain list of bools).

It also persists its settings (slice 3g-d): launch() restores the stride/invert the user last chose
via ctx.get_settings(), and on_control() writes them back with ctx.save_settings(...), so the panel
comes back the way you left it on the next launch.
"""

from ecm_host import TrackerPlugin


class PanelFilter(TrackerPlugin):
    NAME = "Panel Filter"
    DESCRIPTION = "Decimate active points via a control panel (slider + checkbox + button)."

    def __init__(self, ctx):
        super().__init__(ctx)
        self.stride = 2
        self.invert = False

    def launch(self):
        # Restore the stride/invert saved on a previous launch (empty dict the first time).
        saved = self.ctx.get_settings()
        self.stride = max(1, int(saved.get("stride", self.stride)))
        self.invert = bool(saved.get("invert", self.invert))
        return "Panel Filter ready — set the stride, then click Apply."

    def panel(self, ui):
        ui.label("Keep every Nth point, then Apply.")
        ui.slider("stride", "Stride (keep every Nth)", float(self.stride), 1.0, 10.0)
        ui.checkbox("invert", "Invert (drop instead)", self.invert)
        ui.button("apply", "Apply filter")

    def on_control(self, key, value):
        if key == "stride":
            self.stride = max(1, int(round(value)))
            self._save()
        elif key == "invert":
            self.invert = bool(value)
            self._save()
        elif key == "apply":
            self._apply()

    def _save(self):
        self.ctx.save_settings({"stride": self.stride, "invert": self.invert})

    def _apply(self):
        ctx = self.ctx
        if not ctx.has_result or ctx.point_count == 0:
            return
        # Keep mask over all P points: keep index i iff (i % stride == 0), XOR-flipped by invert.
        keep = [((i % self.stride) == 0) != self.invert for i in range(ctx.point_count)]
        ctx.apply_keep_mask(keep)


PLUGIN = PanelFilter

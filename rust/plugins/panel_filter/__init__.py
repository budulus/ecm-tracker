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
        return "Panel Filter ready — set the stride, then click Apply."

    def panel(self, ui):
        ui.label("Keep every Nth point, then Apply.")
        ui.slider("stride", "Stride (keep every Nth)", float(self.stride), 1.0, 10.0)
        ui.checkbox("invert", "Invert (drop instead)", self.invert)
        ui.button("apply", "Apply filter")

    def on_control(self, key, value):
        if key == "stride":
            self.stride = max(1, int(round(value)))
        elif key == "invert":
            self.invert = bool(value)
        elif key == "apply":
            self._apply()

    def _apply(self):
        ctx = self.ctx
        if not ctx.has_result or ctx.point_count == 0:
            return
        # Keep mask over all P points: keep index i iff (i % stride == 0), XOR-flipped by invert.
        keep = [((i % self.stride) == 0) != self.invert for i in range(ctx.point_count)]
        ctx.apply_keep_mask(keep)


PLUGIN = PanelFilter

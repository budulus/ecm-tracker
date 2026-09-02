# Pressure–Strain

Loads a pressure logger CSV containing `Elapsed_s` and `Raw_mbar`, synchronizes it to the full
loaded image sequence using each image file's modification timestamp, and plots homogeneous
principal linear strain against pressure after tracking.

The first and final CSV readings are anchored to the first and final loaded images. Intermediate
frames preserve their relative modification-time spacing. Tracking arrays begin at the main
window's reference frame, so the plugin first aligns pressure globally and only then slices the
reference-to-last range; choosing a later reference therefore uses the corresponding later
pressure value.

The strain selector offers ε₁, ε₂, and their arithmetic mean. The optional zero-pressure setting
subtracts pressure at the current reference frame. Export writes one row per tracked frame and
always reflects the selected strain and zero-pressure setting shown by the current plot.

# Pressure–Strain

Loads a pressure logger CSV containing `Timestamp`, `Arduino_ms` and `Raw_mbar`, matches it to
the loaded image sequence by real capture time, and plots homogeneous principal linear strain
against pressure after tracking.

Each frame's capture time is read from its file name: `Image_<epoch ms>.<ext>` (13 digits, may be zero-padded to 16) or
`Image_YYYYMMDDHHMMSSmmm.<ext>` (local time). Other names are rejected; file modification time
is not used because it is the save or copy time, not the capture time. Pressure sample times come
from the logger's own `Arduino_ms` clock, shifted onto the PC clock by the median of
`Timestamp - Arduino_ms` (the PC `Timestamp` alone jitters because serial rows arrive in bursts).
Pressure is linearly interpolated at each frame's capture time. The logger may start before and
stop after the camera; frames outside the log's time range get no pressure, and a tracked
Reference-to-Last range that includes such frames cannot be plotted or exported.

Both clocks must belong to the same computer (or to synchronized computers), and the CSV's local
`Timestamp` and local-time image names are interpreted in this computer's time zone. Logger clock
drift is not modelled; on real logs it stayed below ~0.1 s over 8.5 minutes.

Tracking arrays begin at the main window's reference frame, so the plugin first aligns pressure
globally and only then slices the reference-to-last range; choosing a later reference therefore
uses the corresponding later pressure value.

The strain selector offers ε₁, ε₂, and their arithmetic mean. The optional zero-pressure setting
subtracts pressure at the current reference frame. Export writes one row per tracked frame and
always reflects the selected strain and zero-pressure setting shown by the current plot.

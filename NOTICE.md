# Notices

This repository distributes `google_gnm_model.tox`, a TouchDesigner component
that embeds a converted copy of model data from the **GNM Ecosystem**
(https://github.com/google/GNM), Copyright Google LLC, licensed under the
Apache License, Version 2.0 (see `LICENSE`).

Conversions applied to the original `gnm_head.npz` (v3.0): morph/skinning data
repacked into a tiled float16 texture; mesh exported as PLY; joint rig exported
as JSON. No changes were made to the underlying statistical model values beyond
float16 quantization (max positional error ≈ 0.25 mm).

The TouchDesigner integration code in the component is likewise licensed under
the Apache License, Version 2.0.

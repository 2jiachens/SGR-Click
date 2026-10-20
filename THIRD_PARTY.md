# Third-party code and models

- **SimpleClick**, <https://github.com/uncbiag/SimpleClick>, MIT:
  `src/isegm` and the Tk canvas/wrapper code originate from SimpleClick.
  SGR-Click adds semantic path construction, control, and evaluation.
  The upstream MIT license is retained in `licenses/SimpleClick.txt`.
- **UniGraCo**, <https://github.com/Zhao-Yian/UniGraCo>, Apache-2.0:
  `src/sgr_click/evaluation/official_clicker.py` and
  `official_oracle.py` retain the upstream click simulator and ten-branch
  evaluation function. The oracle function is extracted into a standalone
  module with local imports; its branch logic is unchanged.
  See `licenses/UniGraCo.txt`.
- **DINOv3**, <https://github.com/facebookresearch/dinov3>:
  obtained separately for TF, under the upstream DINOv3 license.
  DINOv3 weights are not redistributed in this repository. The Lite semantic
  head was trained using DINOv3 descriptors; consult the upstream model terms
  when using derived weights.
- **Dataset annotations and images** are obtained from their original providers
  and remain subject to their respective terms. Only selection metadata is
  included here.

Existing file-level copyright notices in upstream model components remain in
place. The root MIT license applies to original SGR-Click code, subject to the
third-party notices above.

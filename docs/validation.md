# Release validation

The public source layout was verified against the final local TF and Lite
implementations before publication.

- For each model, three real images were checked at all 101 granularities:
  **303 binary masks per model**, with exact pixel equality.
- Semantic anchor masks and their control ticks were identical.
- Negative-click correction predictions were identical, and undo restored the
  selected first-click mask exactly.
- Both English desktop interfaces initialized successfully.
- The main evaluation entry points ran real-image semantic and NoC smoke tests.
  NoC checks included repeated branches, fresh-path comparisons, and verification
  that correction receives the selected mask.
- The fixed Pascal-Part and PartImageNet metadata was checked against original
  annotations. Development/remaining SOR-eligible counts were **100/899** and
  **97/863**, respectively.
- Synthetic unit tests cover boundary-cache equivalence, clustering-cache
  invalidation, path endpoints, shared foreground, one-update-per-pixel behavior,
  metric definitions, oracle ties, failure caps, and PartImageNet label IDs.

These are release regression and integration checks. They are not a new full
benchmark run. Use the evaluation commands to regenerate dataset-level results.
The regression photographs are not redistributed.

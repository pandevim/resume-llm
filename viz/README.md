# viz/ — phase 2

Attention visualization renderer.

In phase 2, the C engine will (optionally) emit a JSON record per query
containing the per-token attention weights over the source PDF. This
directory will house the renderer that turns that into an HTML heatmap:
each PDF token shaded by its attention probability (viridis), top-k
tokens called out, and the highest-probability contiguous span boxed.

For phase 1 there is nothing to render — the engine just samples Guppy.

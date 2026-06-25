"""Trackers package.

Layout::

    trackers/
    ├── common/        # shared primitives (BaseTrack, KF, IoU, matching)
    ├── bytetrack/     # motion-only tracker (default)
    └── detr/          # motion + appearance tracker

Everything shared between concrete trackers lives in :mod:`trackers.common`.
New trackers should depend on :mod:`trackers.common` (and possibly
:mod:`trackers.detr.appearance_matching` for appearance-bearing
variants), not on each other.

The current pipeline-facing entry point is
:mod:`trackers.adapter.ByteTrackAdapter`, which wraps the BYTETracker.
When a DETR backend is wired into the pipeline the adapter swap is a
one-liner (see ``trackers/detr/README.md``).
"""

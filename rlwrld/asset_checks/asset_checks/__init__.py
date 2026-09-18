# SPDX-License-Identifier: Apache-2.0
"""Physics checks of one SimReady asset in three Isaac Sim environments: PhysX, Newton 1.2, Newton 1.5.

run.py (plain Python) launches one Kit process per asset x environment x experiment through
simready-bench's isaac-run; kit/entry.py runs inside that Kit and writes result.json. Scene
building, stepping and capture come from NVIDIA's engine-kit; test parameters and stability
checks come from NVIDIA's Foundation 7.1 tests; this package adds only what those assume away:
reading poses under Newton, selecting the asset's runtime physics variant, and the launch.
"""

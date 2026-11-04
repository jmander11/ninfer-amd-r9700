"""Physical R9700 host paths used by qualification tooling."""

from __future__ import annotations

import os
from pathlib import Path


def power_profile_path() -> Path:
    explicit = os.environ.get("NINFER_POWER_PROFILE")
    if explicit:
        return Path(explicit)
    pci = os.environ.get("NINFER_R9700_PCI_BUS_ID")
    if pci:
        return Path("/sys/bus/pci/devices") / pci / "power_dpm_force_performance_level"
    node = Path(os.environ.get("NINFER_DRM_RENDER_NODE", "/dev/dri/renderD128")).name
    return (
        Path("/sys/class/drm") / node / "device" / "power_dpm_force_performance_level"
    ).resolve()

"""Canonical RFI band list for the HIP 13375 cadence analysis (Capstone).

Single source of truth used by:
- The candidate-annotation pass (notebook 2 §4) — labels each frequency
  with the RFI band it falls in, for human inspection.
- The keep_mask filter (notebook 2 §5 / §7 / §8) — drops candidates inside
  any known RFI band or paper-excluded range from the ranked candidate list.

Band boundaries follow:
- ITU/FCC L-band allocations (Inmarsat, GNSS, Iridium/Globalstar, MSS, GSM-1800).
- GBT-known RFI: 1380 MHz aviation surveillance radar (NWS L-band radar at GBT).
- AWS-3 cellular per FCC Part 27.

Edit here, all call sites pick it up.
"""
from typing import List, Tuple

RFI_BANDS: List[Tuple[float, float, str]] = [
    (1370,   1390,   "Aviation surveillance radar"),
    (1525,   1559,   "Inmarsat downlink"),
    (1559,   1610,   "GNSS (GPS/Galileo)"),
    (1610,   1626,   "Iridium/Globalstar"),
    (1626,   1660.5, "MSS uplink (Inmarsat)"),
    (1670,   1700,   "Meteorological sat"),
    (1695,   1710,   "AWS-3 cellular"),
    (1710,   1785,   "GSM-1800 cellular"),
    (1785,   1805,   "GSM-1800 guard band"),
]

EXCLUDED: List[Tuple[float, float, str]] = [
    (0,    1100, "below 1.1 GHz"),
    (1200, 1340, "notch filter"),
    (1900, 9999, "above 1.9 GHz"),
]


def rfi_band(freq_mhz: float) -> str:
    """Return the RFI-band label for `freq_mhz`, or empty string if clean."""
    for lo, hi, name in RFI_BANDS:
        if lo <= freq_mhz <= hi:
            return name
    return ""


def excluded(freq_mhz: float) -> str:
    """Return the excluded-range label for `freq_mhz`, or empty string."""
    for lo, hi, name in EXCLUDED:
        if lo <= freq_mhz <= hi:
            return name
    return ""


def in_any(freq_mhz: float, bands: List[Tuple[float, float, str]]) -> bool:
    """True if `freq_mhz` falls in any (lo, hi, _) range."""
    return any(lo <= freq_mhz <= hi for lo, hi, _ in bands)

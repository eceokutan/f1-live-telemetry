"""
Fuel consumption lookup from AC car x track datasheet.

Provides estimated fuel_per_lap for any car/track combination before
real telemetry data is available (i.e., before lap 1 completes).

Data is loaded once from ac_fuel_data.json on first call.
Formula: fuel_per_lap = car.base * track.sf * (track.km / 5.0)
"""

import json
import logging
import os
import re
import unicodedata
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_DATA_PATH = os.path.join(os.path.dirname(__file__), "ac_fuel_data.json")
_FUEL_DATA: Optional[dict] = None

# Default fallback: Lotus Exos T125 on Monza GP
_DEFAULT_CAR = {"make": "Lotus", "name": "Exos T125 / Stage 1", "tank": 35, "base": 2.05}
_DEFAULT_TRACK = {"name": "Monza GP", "km": 5.79, "sf": 1.20, "type": "High-speed"}


def _load_data() -> dict:
    """Load and cache the fuel data JSON."""
    global _FUEL_DATA
    if _FUEL_DATA is None:
        with open(_DATA_PATH, "r", encoding="utf-8") as f:
            _FUEL_DATA = json.load(f)
        logger.info(
            "Loaded fuel data: %d cars, %d tracks",
            len(_FUEL_DATA.get("cars", [])),
            len(_FUEL_DATA.get("tracks", [])),
        )
    return _FUEL_DATA


def _strip_accents(text: str) -> str:
    """Remove diacritics/accents (e.g. u-umlaut -> u)."""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def _normalize(text: str) -> str:
    """Normalize for fuzzy matching: lowercase, strip prefixes, simplify."""
    text = _strip_accents(text.lower())
    # Strip common AC prefixes
    text = re.sub(r"^(ks_|ac_)", "", text)
    # Replace separators with spaces
    text = text.replace("_", " ").replace("-", " ")
    # Remove non-alphanumeric except spaces
    text = re.sub(r"[^a-z0-9 ]", "", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _token_overlap(a: str, b: str) -> float:
    """Jaccard-like token overlap score between two normalized strings."""
    tokens_a = set(a.split())
    tokens_b = set(b.split())
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union)


def _match_car(ac_car_id: str, cars: List[dict]) -> Optional[dict]:
    """Match an AC car ID to a datasheet car entry."""
    if not ac_car_id:
        return None

    norm_id = _normalize(ac_car_id)
    if not norm_id:
        return None

    # Build normalized display keys
    candidates = []
    for car in cars:
        display = f"{car['make']} {car['name']}"
        norm_display = _normalize(display)
        candidates.append((norm_display, car))

    # Pass 1: exact match
    for norm_display, car in candidates:
        if norm_id == norm_display:
            return car

    # Pass 2: substring containment (either direction)
    for norm_display, car in candidates:
        if norm_id in norm_display or norm_display in norm_id:
            return car

    # Pass 3: token overlap scoring
    best_score = 0.0
    best_car = None
    for norm_display, car in candidates:
        score = _token_overlap(norm_id, norm_display)
        if score > best_score:
            best_score = score
            best_car = car

    if best_score >= 0.4:
        return best_car

    return None


def _match_track(ac_track_id: str, tracks: List[dict]) -> Optional[dict]:
    """Match an AC track ID to a datasheet track entry."""
    if not ac_track_id:
        return None

    norm_id = _normalize(ac_track_id)
    if not norm_id:
        return None

    # Build normalized track keys (strip trailing "gp" for matching)
    candidates = []
    for track in tracks:
        norm_name = _normalize(track["name"])
        norm_name_no_gp = re.sub(r"\s*gp$", "", norm_name).strip()
        candidates.append((norm_name, norm_name_no_gp, track))

    # Pass 1: exact match (with or without "gp")
    for norm_name, norm_name_no_gp, track in candidates:
        if norm_id == norm_name or norm_id == norm_name_no_gp:
            return track

    # Pass 2: substring containment
    for norm_name, norm_name_no_gp, track in candidates:
        if norm_id in norm_name or norm_name_no_gp in norm_id:
            return track
        if norm_id in norm_name_no_gp or norm_name in norm_id:
            return track

    # Pass 3: token overlap
    best_score = 0.0
    best_track = None
    for norm_name, norm_name_no_gp, track in candidates:
        score = max(
            _token_overlap(norm_id, norm_name),
            _token_overlap(norm_id, norm_name_no_gp),
        )
        if score > best_score:
            best_score = score
            best_track = track

    if best_score >= 0.4:
        return best_track

    return None


def _compute_fuel_per_lap(car: dict, track: dict) -> float:
    """Compute estimated fuel per lap from raw car/track data."""
    return car["base"] * track["sf"] * (track["km"] / 5.0)


def lookup_fuel_consumption(ac_car_id: str, ac_track_id: str) -> Dict:
    """
    Look up estimated fuel consumption for a car/track combination.

    Args:
        ac_car_id: AC internal car model ID (e.g. "ks_ferrari_458")
        ac_track_id: AC internal track ID (e.g. "monza")

    Returns:
        Dict with keys: fuel_per_lap, tank_size, track_km, track_sf, track_type,
                        matched_car, matched_track, is_default
    """
    data = _load_data()
    cars = data.get("cars", [])
    tracks = data.get("tracks", [])

    car = _match_car(ac_car_id, cars)
    track = _match_track(ac_track_id, tracks)

    used_default = False
    if car is None:
        logger.warning("No car match for '%s'; using default (Lotus Exos T125)", ac_car_id)
        car = _DEFAULT_CAR
        used_default = True
    if track is None:
        logger.warning("No track match for '%s'; using default (Monza GP)", ac_track_id)
        track = _DEFAULT_TRACK
        used_default = True

    fuel_per_lap = _compute_fuel_per_lap(car, track)

    result = {
        "fuel_per_lap": fuel_per_lap,
        "tank_size": car["tank"],
        "track_km": track["km"],
        "track_sf": track["sf"],
        "track_type": track.get("type", "Unknown"),
        "matched_car": f"{car['make']} {car['name']}",
        "matched_track": track["name"],
        "is_default": used_default,
    }

    logger.info(
        "Fuel lookup: %s @ %s -> %.2f L/lap (tank: %dL)%s",
        result["matched_car"],
        result["matched_track"],
        fuel_per_lap,
        car["tank"],
        " [DEFAULT]" if used_default else "",
    )

    return result

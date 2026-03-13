"""Telemetry preprocessing for Jarvis Post.

This module converts raw telemetry samples into statistical summaries
for LLM consumption. It reduces context size while preserving analytical value.
"""

from collections import defaultdict
from statistics import mean, stdev


def preprocess_for_analysis(telemetry: list[dict], laps: list[dict]) -> dict:
    """
    Convert raw telemetry samples into statistical summaries.

    This reduces context size while preserving analytical value.

    Args:
        telemetry: List of telemetry sample dictionaries
        laps: List of lap data dictionaries

    Returns:
        Dictionary containing aggregated statistics and patterns
    """
    result = {
        "sample_count": len(telemetry),
        "lap_statistics": _compute_lap_stats(telemetry, laps),
        "throttle_patterns": _analyse_throttle(telemetry),
        "braking_patterns": _analyse_braking(telemetry),
        "tyre_evolution": _analyse_tyres(telemetry),
        "notable_events": _detect_events(telemetry),
    }

    # Add extended telemetry analysis if data is available
    g_force_data = _analyse_g_forces(telemetry)
    if g_force_data:
        result["g_force_analysis"] = g_force_data

    suspension_data = _analyse_suspension(telemetry)
    if suspension_data:
        result["suspension_analysis"] = suspension_data

    return result


def _compute_lap_stats(telemetry: list[dict], laps: list[dict]) -> dict:
    """
    Group telemetry by lap, compute per-lap aggregates.

    Args:
        telemetry: List of telemetry sample dictionaries
        laps: List of lap data dictionaries

    Returns:
        Dictionary with per-lap statistics
    """
    if not telemetry:
        return {"per_lap": []}

    # Group telemetry by lap number
    laps_telemetry = defaultdict(list)
    for sample in telemetry:
        lap_num = sample.get("lap_number")
        if lap_num is not None:
            laps_telemetry[lap_num].append(sample)

    per_lap_stats = []

    for lap_num in sorted(laps_telemetry.keys()):
        lap_samples = laps_telemetry[lap_num]
        speeds = [s.get("speed", 0) for s in lap_samples if s.get("speed") is not None]

        lap_stat = {
            "lap_number": lap_num,
            "sample_count": len(lap_samples),
            "avg_speed": mean(speeds) if speeds else 0.0,
            "max_speed": max(speeds) if speeds else 0.0,
            "min_speed": min(speeds) if speeds else 0.0,
        }

        # Add throttle and brake averages
        throttles = [s.get("throttle", 0) for s in lap_samples if s.get("throttle") is not None]
        brakes = [s.get("brake", 0) for s in lap_samples if s.get("brake") is not None]

        lap_stat["avg_throttle"] = mean(throttles) if throttles else 0.0
        lap_stat["avg_brake"] = mean(brakes) if brakes else 0.0

        per_lap_stats.append(lap_stat)

    return {"per_lap": per_lap_stats}


def _analyse_throttle(telemetry: list[dict]) -> dict:
    """
    Detect hesitations, average application points.

    Throttle hesitation is defined as a significant drop in throttle
    during acceleration (throttle > 0.5 dropping to < 0.5 and then
    rising again).

    Args:
        telemetry: List of telemetry sample dictionaries

    Returns:
        Dictionary with throttle analysis results
    """
    if not telemetry:
        return {
            "average_throttle": 0.0,
            "hesitation_count": 0,
            "full_throttle_pct": 0.0,
            "throttle_variance": 0.0,
        }

    throttles = [s.get("throttle", 0) for s in telemetry if s.get("throttle") is not None]

    if not throttles:
        return {
            "average_throttle": 0.0,
            "hesitation_count": 0,
            "full_throttle_pct": 0.0,
            "throttle_variance": 0.0,
        }

    average_throttle = mean(throttles)

    # Count full throttle samples (>= 0.95)
    full_throttle_count = sum(1 for t in throttles if t >= 0.95)
    full_throttle_pct = (full_throttle_count / len(throttles)) * 100.0

    # Detect hesitations: throttle drops significantly during application
    hesitation_count = 0
    in_throttle_zone = False
    was_high = False

    for i, throttle in enumerate(throttles):
        if throttle >= 0.6:
            in_throttle_zone = True
            was_high = True
        elif in_throttle_zone and was_high and throttle < 0.5:
            # Throttle dropped significantly while in application zone
            hesitation_count += 1
            was_high = False
        elif throttle < 0.3:
            # Reset when fully off throttle
            in_throttle_zone = False
            was_high = False

    # Calculate variance if enough samples
    throttle_variance = 0.0
    if len(throttles) > 1:
        try:
            throttle_variance = stdev(throttles)
        except Exception:
            throttle_variance = 0.0

    return {
        "average_throttle": average_throttle,
        "hesitation_count": hesitation_count,
        "full_throttle_pct": full_throttle_pct,
        "throttle_variance": throttle_variance,
    }


def _analyse_braking(telemetry: list[dict]) -> dict:
    """
    Compute braking point variance, pressure consistency.

    A braking zone is identified when brake pressure exceeds a threshold
    and then returns below it.

    Args:
        telemetry: List of telemetry sample dictionaries

    Returns:
        Dictionary with braking analysis results
    """
    if not telemetry:
        return {
            "average_brake": 0.0,
            "braking_zone_count": 0,
            "max_brake_pressure": 0.0,
            "brake_variance": 0.0,
        }

    brakes = [s.get("brake", 0) for s in telemetry if s.get("brake") is not None]

    if not brakes:
        return {
            "average_brake": 0.0,
            "braking_zone_count": 0,
            "max_brake_pressure": 0.0,
            "brake_variance": 0.0,
        }

    average_brake = mean(brakes)
    max_brake_pressure = max(brakes)

    # Count braking zones (transitions from low to high brake pressure)
    braking_zone_count = 0
    in_braking_zone = False
    brake_threshold = 0.1

    for brake in brakes:
        if brake > brake_threshold and not in_braking_zone:
            in_braking_zone = True
            braking_zone_count += 1
        elif brake <= brake_threshold:
            in_braking_zone = False

    # Calculate variance
    brake_variance = 0.0
    if len(brakes) > 1:
        try:
            brake_variance = stdev(brakes)
        except Exception:
            brake_variance = 0.0

    return {
        "average_brake": average_brake,
        "braking_zone_count": braking_zone_count,
        "max_brake_pressure": max_brake_pressure,
        "brake_variance": brake_variance,
    }


def _analyse_tyres(telemetry: list[dict]) -> dict:
    """
    Analyse tyre temperature and pressure evolution.

    Computes trends (change over session), averages, and imbalances
    between front/rear and left/right tyres.

    Args:
        telemetry: List of telemetry sample dictionaries

    Returns:
        Dictionary with tyre analysis results
    """
    corners = ["fl", "fr", "rl", "rr"]

    # Initialize default result structure
    default_result = {
        "pressure_trend": {c: 0.0 for c in corners},
        "temperature_trend": {c: 0.0 for c in corners},
        "avg_temp": {c: 0.0 for c in corners},
        "avg_pressure": {c: 0.0 for c in corners},
        "front_rear_temp_diff": 0.0,
        "left_right_temp_diff": 0.0,
    }

    if not telemetry:
        return default_result

    # Collect values for each corner
    pressures = {c: [] for c in corners}
    temps = {c: [] for c in corners}

    pressure_keys = {
        "fl": "tyre_pressure_fl",
        "fr": "tyre_pressure_fr",
        "rl": "tyre_pressure_rl",
        "rr": "tyre_pressure_rr",
    }
    temp_keys = {
        "fl": "tyre_temp_fl",
        "fr": "tyre_temp_fr",
        "rl": "tyre_temp_rl",
        "rr": "tyre_temp_rr",
    }

    for sample in telemetry:
        for corner in corners:
            pressure_val = sample.get(pressure_keys[corner])
            temp_val = sample.get(temp_keys[corner])

            if pressure_val is not None:
                pressures[corner].append(pressure_val)
            if temp_val is not None:
                temps[corner].append(temp_val)

    # Calculate trends (end value - start value)
    pressure_trend = {}
    temperature_trend = {}
    avg_temp = {}
    avg_pressure = {}

    for corner in corners:
        # Pressure trend
        if len(pressures[corner]) >= 2:
            pressure_trend[corner] = pressures[corner][-1] - pressures[corner][0]
        else:
            pressure_trend[corner] = 0.0

        # Temperature trend
        if len(temps[corner]) >= 2:
            temperature_trend[corner] = temps[corner][-1] - temps[corner][0]
        else:
            temperature_trend[corner] = 0.0

        # Averages
        avg_pressure[corner] = mean(pressures[corner]) if pressures[corner] else 0.0
        avg_temp[corner] = mean(temps[corner]) if temps[corner] else 0.0

    # Calculate front-rear temperature difference
    front_temps = avg_temp["fl"] + avg_temp["fr"]
    rear_temps = avg_temp["rl"] + avg_temp["rr"]
    front_rear_temp_diff = (front_temps / 2) - (rear_temps / 2) if front_temps and rear_temps else 0.0

    # Calculate left-right temperature difference
    left_temps = avg_temp["fl"] + avg_temp["rl"]
    right_temps = avg_temp["fr"] + avg_temp["rr"]
    left_right_temp_diff = (left_temps / 2) - (right_temps / 2) if left_temps and right_temps else 0.0

    return {
        "pressure_trend": pressure_trend,
        "temperature_trend": temperature_trend,
        "avg_temp": avg_temp,
        "avg_pressure": avg_pressure,
        "front_rear_temp_diff": front_rear_temp_diff,
        "left_right_temp_diff": left_right_temp_diff,
    }


def _detect_events(telemetry: list[dict]) -> list:
    """
    Detect notable events in telemetry data.

    Events include:
    - Maximum speed achieved
    - Heavy braking incidents
    - Unusual throttle patterns

    Args:
        telemetry: List of telemetry sample dictionaries

    Returns:
        List of event dictionaries with type, lap_number, and description
    """
    if not telemetry:
        return []

    events = []

    # Find maximum speed event
    max_speed = 0.0
    max_speed_sample = None

    for sample in telemetry:
        speed = sample.get("speed", 0)
        if speed and speed > max_speed:
            max_speed = speed
            max_speed_sample = sample

    if max_speed_sample and max_speed > 0:
        events.append({
            "type": "max_speed",
            "lap_number": max_speed_sample.get("lap_number", 0),
            "description": f"Maximum speed of {max_speed:.1f} achieved",
            "value": max_speed,
            "elapsed_time": max_speed_sample.get("elapsed_time", 0),
        })

    # Detect heavy braking events (brake > 0.9)
    for sample in telemetry:
        brake = sample.get("brake", 0)
        if brake and brake > 0.9:
            events.append({
                "type": "heavy_braking",
                "lap_number": sample.get("lap_number", 0),
                "description": f"Heavy braking ({brake:.0%} pressure)",
                "value": brake,
                "elapsed_time": sample.get("elapsed_time", 0),
            })

    # Detect lock-ups (speed dropping rapidly while braking)
    prev_speed = None
    for sample in telemetry:
        speed = sample.get("speed", 0)
        brake = sample.get("brake", 0)

        if prev_speed is not None and brake > 0.5:
            speed_drop = prev_speed - speed if speed else 0
            if speed_drop > 20:  # Significant speed drop
                events.append({
                    "type": "potential_lockup",
                    "lap_number": sample.get("lap_number", 0),
                    "description": f"Potential lock-up detected (speed drop: {speed_drop:.1f})",
                    "value": speed_drop,
                    "elapsed_time": sample.get("elapsed_time", 0),
                })

        prev_speed = speed

    # Detect high G-force events
    for sample in telemetry:
        g_lat = sample.get("g_force_lat", 0)
        g_lon = sample.get("g_force_lon", 0)
        if g_lat and abs(g_lat) > 2.5:
            events.append({
                "type": "high_lateral_g",
                "lap_number": sample.get("lap_number", 0),
                "description": f"High lateral G-force: {g_lat:.2f}g",
                "value": g_lat,
                "elapsed_time": sample.get("elapsed_time", 0),
            })
        if g_lon and abs(g_lon) > 2.0:
            events.append({
                "type": "high_longitudinal_g",
                "lap_number": sample.get("lap_number", 0),
                "description": f"High longitudinal G-force: {g_lon:.2f}g",
                "value": g_lon,
                "elapsed_time": sample.get("elapsed_time", 0),
            })

    # Detect car damage
    damage_keys = ["car_damage_front", "car_damage_rear", "car_damage_left",
                    "car_damage_right", "car_damage_centre"]
    for sample in telemetry:
        for key in damage_keys:
            val = sample.get(key, 0)
            if val and val > 0:
                zone = key.replace("car_damage_", "")
                events.append({
                    "type": "car_damage",
                    "lap_number": sample.get("lap_number", 0),
                    "description": f"Car damage detected on {zone}: {val:.1f}",
                    "value": val,
                    "elapsed_time": sample.get("elapsed_time", 0),
                })
                break  # One damage event per sample is enough

    return events


def _analyse_g_forces(telemetry: list[dict]) -> dict | None:
    """Analyse G-force data if available."""
    lat_values = [s.get("g_force_lat") for s in telemetry if s.get("g_force_lat") is not None]
    lon_values = [s.get("g_force_lon") for s in telemetry if s.get("g_force_lon") is not None]

    if not lat_values and not lon_values:
        return None

    result = {}
    if lat_values:
        result["avg_lateral_g"] = mean(lat_values)
        result["max_lateral_g"] = max(lat_values, key=abs)
    if lon_values:
        result["avg_longitudinal_g"] = mean(lon_values)
        result["max_braking_g"] = min(lon_values)
        result["max_accel_g"] = max(lon_values)

    return result


def _analyse_suspension(telemetry: list[dict]) -> dict | None:
    """Analyse suspension travel and ride height if available."""
    corners = ["fl", "fr", "rl", "rr"]
    susp_keys = {c: f"suspension_{c}" for c in corners}

    has_data = any(
        s.get(susp_keys[c]) is not None
        for s in telemetry
        for c in corners
    )
    if not has_data:
        return None

    result = {"avg_travel": {}, "max_travel": {}}
    for corner in corners:
        values = [s.get(susp_keys[corner]) for s in telemetry if s.get(susp_keys[corner]) is not None]
        if values:
            result["avg_travel"][corner] = mean(values)
            result["max_travel"][corner] = max(values)

    # Ride height
    rh_front = [s.get("ride_height_front") for s in telemetry if s.get("ride_height_front") is not None]
    rh_rear = [s.get("ride_height_rear") for s in telemetry if s.get("ride_height_rear") is not None]
    if rh_front:
        result["avg_ride_height_front"] = mean(rh_front)
        result["min_ride_height_front"] = min(rh_front)
    if rh_rear:
        result["avg_ride_height_rear"] = mean(rh_rear)
        result["min_ride_height_rear"] = min(rh_rear)

    return result

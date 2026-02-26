from __future__ import annotations

from typing import Any, Dict

GLOBAL_STREAM_ID = "__global_stream__"


def _to_token(value: Any) -> str:
    if value is None:
        return ""
    token = str(value).strip()
    return token if token else ""


def _get_feature(flow_features: Dict[str, Any], *keys: str) -> str:
    for key in keys:
        token = _to_token(flow_features.get(key))
        if token:
            return token

    lowered = {str(k).lower(): v for k, v in flow_features.items()}
    for key in keys:
        token = _to_token(lowered.get(str(key).lower()))
        if token:
            return token
    return ""


def derive_stream_id(
    *,
    flow_features: Dict[str, Any],
    flow_id: str | None = None,
    explicit_stream_id: Any | None = None,
) -> str:
    explicit = _to_token(explicit_stream_id)
    if explicit:
        return explicit

    # Prefer explicitly provided identifiers if present in features.
    stream_id = _get_feature(flow_features, "stream_id", "session_id", "connection_id", "conn_id")
    if stream_id:
        return stream_id

    src = _get_feature(flow_features, "srcip", "src_ip", "source_ip", "saddr", "src")
    dst = _get_feature(flow_features, "dstip", "dst_ip", "destination_ip", "daddr", "dst")
    sport = _get_feature(flow_features, "sport", "src_port", "source_port", "sp")
    dport = _get_feature(flow_features, "dport", "dst_port", "destination_port", "dp")
    proto = _get_feature(flow_features, "proto", "protocol")

    if src and dst and sport and dport:
        if proto:
            return f"{src}|{dst}|{sport}|{dport}|{proto}"
        return f"{src}|{dst}|{sport}|{dport}"

    if src and dst:
        if proto:
            return f"{src}|{dst}|{proto}"
        return f"{src}|{dst}"

    # UNSW fallback grouping if 5-tuple is unavailable.
    service = _get_feature(flow_features, "service")
    state = _get_feature(flow_features, "state")
    is_sm = _get_feature(flow_features, "is_sm_ips_ports")
    if proto or service or state:
        return f"unsw|{proto}|{service}|{state}|{is_sm}"

    fid = _to_token(flow_id)
    if fid:
        # Keep deterministic but avoid unique-per-flow grouping when numeric suffixes are used.
        parts = fid.split("_")
        if len(parts) > 1 and parts[-1].isdigit():
            return "_".join(parts[:-1]) or GLOBAL_STREAM_ID

    return GLOBAL_STREAM_ID

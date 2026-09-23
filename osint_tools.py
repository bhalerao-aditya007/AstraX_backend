# osint_tools.py — OSINT source adapters for AstraX Police AI.
# Mounted on the existing Gradio FastAPI app as /api/v1/osint/* endpoints.
#
# Final set: ExifTool + phonenumbers + dnstwist + Sherlock
# Three of four run with zero egress, keeping the air-gap claim honest.

import datetime
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile

# ── Selector validation ─────────────────────────────────────────────────
SELECTOR_PATTERNS = {
    "username": re.compile(r"^[A-Za-z0-9._\-]{3,32}$"),
    "domain":   re.compile(r"^(?=.{4,253}$)([A-Za-z0-9]([A-Za-z0-9\-]{0,61}[A-Za-z0-9])?\.)+[A-Za-z]{2,}$"),
    "phone":    re.compile(r"^\+?[0-9]{8,15}$"),
}

ALLOW_EGRESS = os.environ.get("OSINT_ALLOW_EGRESS", "true").lower() == "true"


def validate(selector_type: str, value: str) -> str:
    """Validate and sanitize an OSINT selector. Raises ValueError on bad input."""
    v = (value or "").strip()
    pat = SELECTOR_PATTERNS.get(selector_type)
    if not pat or not pat.match(v):
        raise ValueError(f"invalid {selector_type} selector: {value!r}")
    return v


def _run(cmd, timeout):
    """Run a subprocess with NO shell — selectors originate from OCR/ASR output."""
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return p.stdout, p.stderr, p.returncode


def _finding(kind, label, attrs, confidence, tool, args, source_url=None):
    """Build a standardized OSINT finding dict with hash-anchored provenance."""
    raw = json.dumps(attrs, sort_keys=True, default=str)
    return {
        "finding_type": kind,
        "label": label,
        "attributes": attrs,
        "confidence": round(confidence, 3),
        "source_url": source_url,
        "tool": tool,
        "tool_args": args,
        "raw_sha256": hashlib.sha256(raw.encode()).hexdigest(),
        "collected_at": datetime.datetime.utcnow().isoformat() + "Z",
        "egress_used": tool in ("sherlock", "maigret", "dnstwist-resolved"),
    }


# ── 1. ExifTool (offline) ───────────────────────────────────────────────
def run_exiftool(file_path: str):
    """Extract EXIF GPS, device serial, and timestamps from seized media.
    Fully offline — no network calls."""
    if not shutil.which("exiftool"):
        return []

    try:
        out, _, rc = _run(["exiftool", "-json", "-G", "-n", file_path], 30)
    except subprocess.TimeoutExpired:
        return []

    if rc != 0 or not out.strip():
        return []

    try:
        meta = json.loads(out)[0]
    except (json.JSONDecodeError, IndexError):
        return []

    def g(*keys):
        return next((meta[k] for k in keys if meta.get(k) not in (None, "", "0000:00:00 00:00:00")), None)

    findings = []

    # GPS geolocation
    lat = g("EXIF:GPSLatitude", "Composite:GPSLatitude")
    lon = g("EXIF:GPSLongitude", "Composite:GPSLongitude")
    if lat is not None and lon is not None:
        try:
            lat_f, lon_f = float(lat), float(lon)
            findings.append(_finding(
                "geolocation", f"{lat_f:.5f}, {lon_f:.5f}",
                {
                    "latitude": lat_f,
                    "longitude": lon_f,
                    "altitude": g("EXIF:GPSAltitude"),
                    "captured_at": g("EXIF:DateTimeOriginal", "EXIF:CreateDate"),
                },
                0.92, "exiftool", ["-json", "-G", "-n"],
            ))
        except (ValueError, TypeError):
            pass

    # Capture device identification
    make = g("EXIF:Make")
    model = g("EXIF:Model")
    serial = g("EXIF:SerialNumber", "EXIF:BodySerialNumber", "MakerNotes:SerialNumber")
    if make or model or serial:
        # Serial is an identifier (high conf); make/model is a class (lower conf)
        conf = 0.95 if serial else 0.70
        device_label = " ".join(str(x) for x in [make, model] if x) or "Unknown device"
        findings.append(_finding(
            "device", device_label,
            {
                "make": make,
                "model": model,
                "serial_number": serial,
                "software": g("EXIF:Software"),
                "lens": g("EXIF:LensModel"),
                "captured_at": g("EXIF:DateTimeOriginal"),
            },
            conf, "exiftool", ["-json", "-G", "-n"],
        ))

    # Capture timestamp
    dt_original = g("EXIF:DateTimeOriginal")
    if dt_original:
        findings.append(_finding(
            "timestamp", str(dt_original),
            {
                "original": dt_original,
                "modified": g("File:FileModifyDate"),
                "offset": g("EXIF:OffsetTimeOriginal"),
            },
            0.85, "exiftool", ["-json", "-G", "-n"],
        ))

    return findings


# ── 2. Phone (offline, replaces PhoneInfoga) ────────────────────────────
def run_phone(number: str, region="IN"):
    """Offline MSISDN validation using libphonenumber.
    Returns carrier, circle, line-type for Indian numbers.
    VoIP on a harassment FIR number is itself a red flag."""
    import phonenumbers
    from phonenumbers import carrier, geocoder, number_type, PhoneNumberType

    n = validate("phone", number)
    try:
        parsed = phonenumbers.parse(n if n.startswith("+") else n, region)
    except phonenumbers.NumberParseException:
        return []

    if not phonenumbers.is_valid_number(parsed):
        return [_finding(
            "phone_invalid", n,
            {"valid": False, "input": n, "region": region},
            0.99, "phonenumbers", [region],
        )]

    kind_map = {
        PhoneNumberType.MOBILE: "mobile",
        PhoneNumberType.FIXED_LINE: "landline",
        PhoneNumberType.VOIP: "voip",
        PhoneNumberType.FIXED_LINE_OR_MOBILE: "fixed_or_mobile",
        PhoneNumberType.TOLL_FREE: "toll_free",
        PhoneNumberType.PREMIUM_RATE: "premium_rate",
        PhoneNumberType.PERSONAL_NUMBER: "personal",
    }

    nt = number_type(parsed)
    attrs = {
        "e164": phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.E164),
        "country_code": parsed.country_code,
        "national_number": str(parsed.national_number),
        "carrier": carrier.name_for_number(parsed, "en") or None,
        "circle": geocoder.description_for_number(parsed, "en") or None,
        "line_type": kind_map.get(nt, "unknown"),
        "valid": True,
    }

    # VoIP on an Indian number in a harassment case is itself a signal
    conf = 0.90 if attrs["carrier"] else 0.75
    return [_finding(
        "phone_profile", attrs["e164"],
        attrs, conf, "phonenumbers", [region],
    )]


# ── 3. dnstwist (permutation offline, resolution needs DNS) ─────────────
def run_dnstwist(domain: str, resolve=True, limit=40):
    """Typosquat clustering — maps a phishing/sextortion domain to its
    registered family. Deterministic, fast, visually excellent in the graph."""
    d = validate("domain", domain)
    if not shutil.which("dnstwist"):
        return []

    cmd = ["dnstwist", "--format", "json"]
    resolved = resolve and ALLOW_EGRESS
    if resolved:
        cmd += ["--registered", "--mx"]
    cmd += [d]

    try:
        out, _, rc = _run(cmd, 180 if resolved else 30)
    except subprocess.TimeoutExpired:
        return []

    if rc != 0 or not out.strip():
        return []

    try:
        rows = json.loads(out)
    except json.JSONDecodeError:
        return []

    results = []
    for r in rows[:limit]:
        name = r.get("domain") or r.get("domain-name")
        if not name or name == d:
            continue

        a_records = r.get("dns_a") or r.get("dns-a") or []
        mx_records = r.get("dns_mx") or r.get("dns-mx") or []
        ns_records = r.get("dns_ns") or r.get("dns-ns") or []
        registered = bool(a_records or mx_records)

        if resolved and not registered:
            continue

        # A registered typosquat with a live MX is a phishing-capable host
        if a_records and mx_records:
            conf = 0.85
        elif a_records:
            conf = 0.70
        else:
            conf = 0.35

        results.append(_finding(
            "typosquat", name,
            {
                "fuzzer": r.get("fuzzer"),
                "a_records": a_records,
                "mx_records": mx_records,
                "ns_records": ns_records,
                "registered": registered,
                "parent_domain": d,
            },
            conf,
            "dnstwist-resolved" if resolved else "dnstwist",
            cmd[1:-1],
            source_url=f"http://{name}",
        ))

    return results


# ── 4. Sherlock (needs egress) ──────────────────────────────────────────
def run_sherlock(username: str, timeout=150):
    """Username → social accounts. ~60s runtime.
    Username reuse is the most common OPSEC failure in Indian cyber-fraud."""
    u = validate("username", username)
    if not ALLOW_EGRESS:
        return []
    if not shutil.which("sherlock"):
        return []

    with tempfile.TemporaryDirectory() as tmp:
        try:
            out, _, _ = _run(
                ["sherlock", "--print-found", "--no-color", "--timeout", "10",
                 "--folderoutput", tmp, u],
                timeout,
            )
        except subprocess.TimeoutExpired:
            return []

    results = []
    for line in out.splitlines():
        line = line.strip()
        if not line.startswith("[+]"):
            continue
        body = line[3:].strip()
        if ":" not in body:
            continue
        site, url = body.split(":", 1)
        url = url.strip()
        if not url.startswith("http"):
            continue
        results.append(_finding(
            "social_account", f"{site.strip()}/{u}",
            {"platform": site.strip(), "username": u, "profile_url": url},
            0.40,  # single-tool username hit is a lead, never an identification
            "sherlock", ["--print-found"],
            source_url=url,
        ))

    return results

"""IOC and identifier format validation and repair rules."""

from __future__ import annotations

import re

from src.schema.entities import IOCSubtype


# --- Regex patterns for IOC validation ---

CVE_PATTERN = re.compile(r"^CVE-\d{4}-\d{4,}$", re.IGNORECASE)
CVE_LOOSE_PATTERN = re.compile(r"CVE\s*[-–]?\s*(\d{4})\s*[-–]?\s*(\d{4,})", re.IGNORECASE)

ATTACK_TECHNIQUE_PATTERN = re.compile(r"^T\d{4}(\.\d{3})?$")
ATTACK_TACTIC_PATTERN = re.compile(r"^TA\d{4}$")
ATTACK_SOFTWARE_PATTERN = re.compile(r"^S\d{4}$")
ATTACK_GROUP_PATTERN = re.compile(r"^G\d{4}$")

IPV4_PATTERN = re.compile(
    r"^(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})$"
)
DOMAIN_PATTERN = re.compile(
    r"^(?:[a-zA-Z0-9](?:[a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?\.)+"
    r"[a-zA-Z]{2,}$"
)
URL_PATTERN = re.compile(r"^https?://", re.IGNORECASE)
EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[a-zA-Z]{2,}$")

# Common hash lengths
MD5_LEN = 32
SHA1_LEN = 40
SHA256_LEN = 64
HASH_PATTERN = re.compile(r"^[a-fA-F0-9]+$")

# Defanging patterns
DEFANG_PATTERNS = [
    (re.compile(r"\[\.\]"), "."),
    (re.compile(r"\[:\]"), ":"),
    (re.compile(r"hxxp", re.IGNORECASE), "http"),
    (re.compile(r"hXXp", re.IGNORECASE), "http"),
    (re.compile(r"\[at\]", re.IGNORECASE), "@"),
    (re.compile(r"\[dot\]", re.IGNORECASE), "."),
]


def detect_ioc_subtype(value: str) -> IOCSubtype:
    """Detect the subtype of an IOC entity from its value."""
    cleaned = _defang(value)

    if IPV4_PATTERN.match(cleaned):
        return IOCSubtype.IP
    if URL_PATTERN.match(cleaned):
        return IOCSubtype.URL
    if EMAIL_PATTERN.match(cleaned):
        return IOCSubtype.EMAIL
    if DOMAIN_PATTERN.match(cleaned) and "." in cleaned:
        return IOCSubtype.DOMAIN
    if HASH_PATTERN.match(cleaned) and len(cleaned) in (MD5_LEN, SHA1_LEN, SHA256_LEN):
        return IOCSubtype.HASH

    return IOCSubtype.UNKNOWN


def validate_cve(value: str) -> tuple[bool, str | None]:
    """Validate a CVE identifier. Returns (is_valid, repaired_value_or_None)."""
    if CVE_PATTERN.match(value):
        return True, value.upper()

    # Try loose match and repair
    match = CVE_LOOSE_PATTERN.search(value)
    if match:
        repaired = f"CVE-{match.group(1)}-{match.group(2)}"
        return True, repaired.upper()

    return False, None


def validate_attack_id(value: str) -> bool:
    """Validate a MITRE ATT&CK technique, tactic, software, or group ID."""
    value = value.strip()
    return bool(
        ATTACK_TECHNIQUE_PATTERN.match(value)
        or ATTACK_TACTIC_PATTERN.match(value)
        or ATTACK_SOFTWARE_PATTERN.match(value)
        or ATTACK_GROUP_PATTERN.match(value)
    )


def validate_ipv4(value: str) -> tuple[bool, str | None]:
    """Validate and normalize an IPv4 address."""
    cleaned = _defang(value)
    match = IPV4_PATTERN.match(cleaned)
    if not match:
        return False, None
    octets = [int(g) for g in match.groups()]
    if all(0 <= o <= 255 for o in octets):
        # Normalize: strip leading zeros
        normalized = ".".join(str(o) for o in octets)
        return True, normalized
    return False, None


def validate_domain(value: str) -> tuple[bool, str | None]:
    """Validate a domain name."""
    cleaned = _defang(value).lower()
    if DOMAIN_PATTERN.match(cleaned):
        return True, cleaned
    return False, None


def validate_hash(value: str) -> tuple[bool, str | None]:
    """Validate a file hash (MD5, SHA1, SHA256)."""
    cleaned = value.strip().lower()
    if HASH_PATTERN.match(cleaned) and len(cleaned) in (MD5_LEN, SHA1_LEN, SHA256_LEN):
        return True, cleaned
    return False, None


def _defang(value: str) -> str:
    """Remove common defanging patterns from IOC strings."""
    result = value
    for pattern, replacement in DEFANG_PATTERNS:
        result = pattern.sub(replacement, result)
    return result


def normalize_ioc(value: str) -> tuple[str, str]:
    """Normalize an IOC, returning (canonical_form, original_form).

    Canonical form is defanged and lowercased for storage.
    Original form is preserved for display.
    """
    canonical = _defang(value).lower().strip()
    return canonical, value.strip()

import logging
from typing import Any

from PIL import Image
from PIL.ExifTags import TAGS

logger = logging.getLogger(__name__)

# Known tags that heavily imply synthetic generation
# e.g., Midjourney, DALL-E, Stable Diffusion usually leave Software tags
SUSPICIOUS_SOFTWARE_KEYWORDS = ["midjourney", "dall-e", "stable diffusion", "comfyui", "ai generated"]


def extract_metadata(image: Image.Image) -> dict[str, Any]:
    """Extract EXIF and basic info from a PIL Image."""
    meta = {}

    # Basic info
    if "Software" in image.info:
        meta["Software"] = image.info["Software"]

    # EXIF data
    try:
        exif_data = image.getexif()
        if exif_data:
            for tag_id, value in exif_data.items():
                tag_name = TAGS.get(tag_id, tag_id)
                # Filter out raw byte blobs that might clutter or crash JSON
                if isinstance(value, bytes):
                    continue
                # TAGS.get returns the raw integer id for a tag it does not know.
                meta[str(tag_name)] = value
    except Exception as e:  # noqa: BLE001 - EXIF parsing; a malformed tag must never fail the tool
        logger.debug(f"Failed to parse EXIF: {e}")

    return meta


def analyze_metadata_anomalies(image: Image.Image) -> list[dict[str, str]]:
    """Checks for metadata signatures that indicate the image is AI generated.

    Returns a list of anomaly dicts (evidence).

    A hit here is strong evidence; an empty result is **no evidence either way**, and must
    not be reported as "not AI generated". This reads one tag (`Software`) against a fixed
    keyword list, so it only catches a generator that labelled its own output and whose
    label survived: re-saving, screenshotting, stripping EXIF, or any upload pipeline that
    normalises metadata all defeat it, as does any generator not on the list. Recall is
    therefore near zero and precision near total -- deliberately, since the alternative
    (inferring synthesis from *missing* EXIF) produces false positives on every edited or
    exported photograph.
    """
    anomalies: list[dict[str, str]] = []
    meta = extract_metadata(image)

    if not meta:
        return anomalies

    software = meta.get("Software", "")
    if isinstance(software, str):
        lower_software = software.lower()
        for kw in SUSPICIOUS_SOFTWARE_KEYWORDS:
            if kw in lower_software:
                anomalies.append(
                    {
                        "claim": "Image metadata indicates synthetic origin",
                        "evidence": f"Found '{software}' in EXIF Software tag.",
                        "source": "EXIF Forensics",
                    }
                )
                break

    # Further heuristics could check for impossibly missing EXIF (e.g. claims to be photo but no ISO/FocalLength)
    # but we will stick to explicit positive flags to minimize false positives.
    return anomalies

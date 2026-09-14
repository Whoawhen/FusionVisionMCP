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
                meta[tag_name] = value
    except Exception as e:
        logger.debug(f"Failed to parse EXIF: {e}")
        
    return meta

def analyze_metadata_anomalies(image: Image.Image) -> list[dict[str, str]]:
    """
    Checks for metadata signatures that indicate the image is AI generated.
    Returns a list of anomaly dicts (evidence).
    """
    anomalies = []
    meta = extract_metadata(image)
    
    if not meta:
        return anomalies
        
    software = meta.get("Software", "")
    if isinstance(software, str):
        lower_software = software.lower()
        for kw in SUSPICIOUS_SOFTWARE_KEYWORDS:
            if kw in lower_software:
                anomalies.append({
                    "claim": f"Image metadata indicates synthetic origin",
                    "evidence": f"Found '{software}' in EXIF Software tag.",
                    "source": "EXIF Forensics"
                })
                break
                
    # Further heuristics could check for impossibly missing EXIF (e.g. claims to be photo but no ISO/FocalLength) 
    # but we will stick to explicit positive flags to minimize false positives.
    return anomalies

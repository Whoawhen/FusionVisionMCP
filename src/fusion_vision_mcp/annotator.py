import os
import tempfile
import uuid

from PIL import Image, ImageDraw, ImageFont


def save_annotated_image(image: Image.Image, boxes: list[list[float]], labels: list[str] | None = None) -> str:
    """
    Draws bounding boxes and optional labels on a copy of the image,
    saves it to a local temp directory, and returns the absolute file path.
    """
    temp_dir = os.path.join(tempfile.gettempdir(), ".fusion_vision_temp")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Convert rather than copy: the output is saved as JPEG, which cannot encode an
    # alpha channel, so an RGBA source (any transparent PNG -- `get_images` does not
    # normalise mode) raised `OSError: cannot write mode RGBA as JPEG`. Converting also
    # makes the red box and label render correctly on a grayscale or palette source.
    img_draw = image.convert("RGB")
    draw = ImageDraw.Draw(img_draw)
    
    for i, box in enumerate(boxes):
        # box is [x1, y1, x2, y2]
        draw.rectangle(box, outline="red", width=3)
        if labels and i < len(labels):
            # Draw label background and text
            label = labels[i]
            x1, y1 = box[:2]
            # Try to get a default font, fallback if unavailable
            try:
                font = ImageFont.load_default()
            except OSError:
                font = None
            
            if font:
                # Basic text bounding box
                text_bbox = draw.textbbox((x1, y1), label, font=font)
                draw.rectangle([text_bbox[0], text_bbox[1], text_bbox[2], text_bbox[3]], fill="red")
                draw.text((x1, y1), label, fill="white", font=font)
                
    filename = f"annotated_{uuid.uuid4().hex[:8]}.jpg"
    filepath = os.path.abspath(os.path.join(temp_dir, filename))
    img_draw.save(filepath, "JPEG")
    return filepath


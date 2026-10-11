# Image sources to consider later

Candidates the user collected on 2026-10-10, sorted by fit for VisionBench. Licences are from
memory and **must be verified** before any download (accepted: public domain, CC0, CC BY, CC BY-SA,
US government works). New images follow the usual route: the worker proposes, the user approves,
and Claude adds a reproducible, licence-recording download step.

## Worth using

| Source | Gap it would fill | Why | Check first |
|---|---|---|---|
| **VizWiz-Captions** | `critique_composition`; blur and framing robustness (`ocr` on blurred images is −52 points); captions on hard photos | Everyday photos by blind and low-vision people: poor framing, blur, extreme close-ups, the faults a composition critique should catch. 5 human captions each. | The images' licence (user uploads), separately from the annotations' |
| **TextCaps** | Held-out check of `caption(read_text)`; `query_image` text questions | Human captions that require reading the text in the image: exactly what the OCR text line targets. Same images as TextVQA (Open Images), which VisionBench already uses. | Overlap with VisionBench's 200 TextVQA images |
| **nocaps** | Caption generalisation beyond COCO | ~400 object classes absent from COCO, on Open Images photos with human captions; a check that caption results are not COCO-shaped. | Annotation licence; Open Images per-image licences |
| **WIT (Wikipedia images)** | Caption "specialist knowledge" misses (art periods, named painters, landmarks) | Encyclopedic image-text pairs; Wikimedia images carry per-image licences, many CC BY-SA or public domain. | Per-image licence filtering; the text is CC BY-SA |
| **Screen2Words / Widget Captioning** | App and web screenshots (caption, `query_image`) | Human descriptions of mobile screens and UI elements (RICO screenshots). | RICO screenshot licence |
| **Visual Genome** | `dense_region_caption` | Human region boxes with phrases, attributes and relations: a much larger test than phase 7's 25 images. | Image licences (Flickr, mixed); overlap with COCO |

## Not for the benchmark

- **Flickr30k:** Flickr images under research-only terms, outside the licence list. COCO and nocaps
  cover the same ground.
- **DocVQA:** documents from the UCSF Industry Documents Library under restrictive terms; the
  benchmark's own documents set covers this.
- **CC3M / CC12M, SBU, LAION, COYO:** web-scale URL lists without image licences. Pretraining data,
  not benchmark images.
- **ShareGPT4V / LLaVA-150k:** captions written by GPT-4V, useful only as fine-tuning data, which
  waits until the suite is "good". The provider's terms on training with its outputs need checking
  first.

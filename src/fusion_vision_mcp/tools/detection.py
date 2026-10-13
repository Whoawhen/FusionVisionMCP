"""Detection and measurement tools: `detect_objects`, `dense_region_caption`,
`count_objects` and `spatial_relations`."""

from typing import Annotated, Any, Final, Literal

from mcp.server.mcpserver import Context, MCPServer
from pydantic import Field

from fusion_vision_mcp import geometry
from fusion_vision_mcp.ambiguity import check_semantic_ambiguity
from fusion_vision_mcp.analysis import (
    _add_silhouette,
    _attach_annotated_images,
    _first_int,
    _region_label_consensus,
    _separability,
)
from fusion_vision_mcp.constants import (
    COUNT_BOX_THRESHOLD,
    DETECT_BOX_THRESHOLD,
    MASK_DECODE_RESOLUTION,
    SAM3_MASK_DECODE_RESOLUTION,
)
from fusion_vision_mcp.detection_policy import execute_detection, locate_and_segment
from fusion_vision_mcp.images import get_images
from fusion_vision_mcp.params import ImagePath, ObjectName
from fusion_vision_mcp.protocols import AppContext

#: Ceiling on objects compared in one `spatial_relations` call. Relations grow as
#: n(n-1)/2, so an over-broad detection could otherwise return a huge payload.
_MAX_RELATED_OBJECTS: Final[int] = 12


def register(mcp: MCPServer) -> None:
    @mcp.tool()
    def detect_objects(
        ctx: Context[AppContext],
        src: ImagePath,
        object_name: ObjectName,
        return_annotated: Annotated[
            bool,
            Field(
                description=("If true, returns a local temp file path to an annotated image with drawn bounding boxes.")
            ),
        ] = False,
    ) -> list[dict[str, Any]] | dict[str, Any]:
        """Locate a named object in an image, as scored bounding boxes and center points.

        Returns `bboxes` ([x1, y1, x2, y2] each), `points` (the center of each box),
        `labels` and `scores`, all index-aligned -- so use this whether you want
        regions or coordinates; the centers come free with the boxes. `detector`
        says which model ran and `_routing` why.

        On photographs it uses SAM 3 and keeps boxes scoring 0.125 or more. SAM 3
        can say "not here", including for a plausible neighbour: asked for an
        object that a labeller named as plausible but absent (a fork in a kitchen,
        a smartphone on a screenshot), it boxed something on 9% of 166 held-out
        generalist images, where the previous detector (Grounding DINO) boxed
        something on 66%; for an absent COCO category, on none of 80 scenes. It
        found 89% of labelled objects in COCO scenes and 66% across generalist
        images, screenshots and documents included. Its weakness is strays: only
        a fifth to a third of its boxes land on a labelled instance, so read `scores`
        and treat low-scoring extra boxes as candidates, not finds. If SAM 3 cannot
        be loaded (it is gated on Hugging Face), this falls back to Grounding DINO
        at 0.35, and `detector` says which ran.

        Flat vector art and clip art go to Florence-2's grounding head instead.
        That path has no `scores`, and its label is the query echoed back rather
        than a classification: asked for "elephant" on a drawing of a dog, it
        boxes the dog and labels it "elephant". Grounding DINO did no better
        there -- it boxed an absent object on 95% of flat images and found fewer of
        the real ones. SAM 3 was not measured on flat art, so it is not used there.
        On either path, a box does not confirm the object is
        present: confirm presence with `caption` or `query_image`.

        If `return_annotated` is true, the return value becomes a dictionary containing
        the regular results plus an `annotated_image_path` key.

        Treat results as candidates to inspect, not a tally: one physical object
        can come back as a whole-object box plus part boxes, and two touching
        instances as one box. Use `count_objects` when you need "how many" -- it
        adds the silhouette, consensus and separability checks a count needs.
        Neither can separate heavily overlapping instances.

        Boxes cannot answer whether two objects actually touch or whether one is
        inside another -- they overlap the moment one object is merely in front
        of another. Use `spatial_relations` for that.

        A maximally vague `object_name` ("object", "thing", "item", ...) on a
        scene with nothing distinctive to point to gets a near-full-frame box
        filtered out rather than returned as a match -- detectors fall back to the
        whole image for such a name. This only applies to the exact generic
        vocabulary itself: a specific noun that genuinely fills the frame (a
        close-up of wood, asked for "wood") is returned unfiltered.
        """
        app = ctx.request_context.lifespan_context
        with get_images(src) as images:
            results = execute_detection(
                app,
                images,
                object_name,
                backend_preference="auto",
                threshold=DETECT_BOX_THRESHOLD,
                adaptive_threshold=False,
                locate=True,
            )

            if return_annotated:
                _attach_annotated_images(images, results)

            return results

    @mcp.tool()
    def dense_region_caption(ctx: Context[AppContext], src: ImagePath) -> list[dict[str, Any]]:
        """Caption every salient region of an image at once, with bounding boxes.

        Use this to inventory an image without knowing in advance what is in it --
        it returns `bboxes` and `labels` for each region it finds, discovering the
        objects itself. That is the difference from `detect_objects`, which needs
        you to name the object you are looking for, and from `caption`, which
        describes the whole scene in prose with no coordinates.

        Good at finding what is there, loose about where: against reference region
        annotations it found about three quarters of the regions, but its boxes
        overlapped them by only about 25% IoU on average. Use it to learn what an
        image contains, then `spatial_relations`, which segments each named
        object, when its exact extent matters.
        """
        with get_images(src) as images:
            return ctx.request_context.lifespan_context.florence2.dense_region_caption(images)

    @mcp.tool()
    def count_objects(
        ctx: Context[AppContext],
        src: ImagePath,
        object_name: ObjectName,
        verify_silhouette: Annotated[
            bool,
            Field(
                description=(
                    "When the detector finds only one region, segment it and measure how many "
                    "repeated lobes its outline has. Costs one SAM2 load on the first such call."
                )
            ),
        ] = True,
        consensus: Annotated[
            bool,
            Field(
                description=(
                    "When true (default), also tally how many `dense_region_caption` labels match "
                    "the object name as an independent second opinion, and report a `separable` "
                    "flag. The dense region captioner is a different Florence-2 head than Grounding "
                    "DINO, so agreement between the two is real evidence and disagreement is a "
                    "visible warning. Cheap -- uses the already-loaded Florence-2, no new model."
                )
            ),
        ] = True,
        vqa_estimate: Annotated[
            bool,
            Field(
                description=(
                    "When true and the detector has collapsed overlapping instances (separable is "
                    "'no'), also ask Moondream2 'how many <name>?' and attach its answer as "
                    "estimates.vqa -- a judgment estimate, clearly marked, not a tally. Off by "
                    "default so a session that never asks keeps Moondream unloaded."
                )
            ),
        ] = False,
        threshold: Annotated[
            float,
            Field(
                description=(
                    "Box confidence floor passed to Grounding DINO (default 0.425: chosen on real "
                    "photos, where it counted exactly on 61.8% of held-out cases against 43.8% at "
                    "the old 0.15, with every single-object control in the fixture suite still "
                    "counting as 1). "
                    "Raise it for a visually cluttered scene with distractor shapes near the target: "
                    "on one measured case (a target star among 18 muted-color distractors) the "
                    "default threshold counted every distractor, while 0.5 separated the true "
                    "target (score 0.90) from all of them (scores 0.19-0.32) for an exact count. "
                    "This is a manual lever, not automatic -- raising it blindly on a normal scene "
                    "can just as easily drop real instances, so use it once you suspect clutter, "
                    "not by default."
                )
            ),
        ] = COUNT_BOX_THRESHOLD,
        adaptive_threshold: Annotated[
            bool,
            Field(
                description=(
                    "When true, evaluates the raw detection distribution using kernel density "
                    "estimation to automatically raise the threshold just above noise clusters "
                    "if clear modes exist. Suppresses hundreds of false-positive sub-boxes in "
                    "texture-heavy scenes while preserving real detections. Overrides manual "
                    "threshold if the adaptive threshold is higher."
                )
            ),
        ] = True,
        clip_art: Annotated[
            bool,
            Field(
                description=(
                    "When true, count with Florence-2's grounding head instead of Grounding DINO. "
                    "Grounding DINO's training distribution is real photographs; on flat vector "
                    "art and iconography it can over-detect (measured on a mixed clip-art scene: "
                    "6 instead of 2 trees, and a spurious 4th house past the true 3). Florence-2 got "
                    "both exactly right on the same scene. Try it when the image is clip art / flat "
                    "vector style and the default count looks wrong; `scores` are unavailable on "
                    "this path (Florence-2's grounding head carries no per-box confidence) and "
                    "`group_boxes_dropped` is not applicable."
                )
            ),
        ] = False,
        return_annotated: Annotated[
            bool, Field(description="If true, saves an annotated image with counting boxes and returns the file path.")
        ] = False,
    ) -> list[dict[str, Any]]:
        """Count how many instances of a named object an image contains.

        Use this, not `detect_objects`, whenever the question is "how many".
        `detect_objects` returns candidate boxes for locating things; this tool uses
        a score floor tuned for exact counts on real photos (0.425) and adds the
        silhouette, consensus and separability checks that tell a real tally from a
        collapsed group.

        Returns `count`, plus `bboxes`/`points`/`labels`/`scores` for the instances
        found, index-aligned and in the same pixel-space convention `detect_objects`
        uses. `scores` are per-detection confidences, so a count resting on marginal
        detections is visible rather than implied. `group_boxes_dropped` counts
        detections that enclosed the whole arrangement rather than one instance.

        Accuracy depends heavily on the scene. On synthetic shapes whose counts are
        exact by construction it is right on 9 of 10 cases (mean error 0.1) and
        holds all 8 single-object controls. On real photographs it is reliable for
        a few instances and weak for many: it was exact on 61.8% of 427 held-out
        photo counts (383 of them 3 or fewer), and crowded scenes remain far less
        reliable. Treat a count above a handful on a real photo as a rough estimate.

        Three measured limits, worth checking before trusting a count:

        Crowded real scenes (see above) are the biggest one.

        Heavy overlap undercounts. Eight identical shapes in a ring count as 8 when
        separated or touching, but 6 once they overlap by roughly two-thirds of their
        width. A count below what you expect means "could not separate them", not a
        real tally -- and it will undercount rather than overcount.

        Some evidence is simply not in the picture. On a photo of a paper flower whose
        petals overlap, this returns 1 at every resolution from 128px to 768px, as does
        every other approach tried -- other detectors, outline geometry, tiled crops,
        and interior-colour analysis alike. That flower's outline is 98% convex and its
        interior contrast is near zero, so nothing measurable distinguishes the petals.
        Ask `query_image` for a count in that situation and treat it as an estimate.

        When only one region is found, a `silhouette` block is added measuring how
        many repeated lobes that region's outline contains. `count` and
        `silhouette.lobes` come from different methods and **neither overrides the
        other**: `count: 1` beside `silhouette.lobes: 8` means the detector could not
        separate the instances while the outline shows eight cores. `agreement: true`
        means a second, independent estimator matched it; `shattered` or `clipped`
        mean the number should not be used at all.

        Set `consensus=true` (the default) for two extra fields that surface
        structural ambiguity instead of hiding it:

        - `consensus`: `{detector_count, region_label_count, agree}` -- a second
          count from Florence-2's dense region captioner (a different head), which
          tally how many region labels contain the object name. Agreement is real
          evidence; disagreement is a warning.
        - `separable`: `"yes"` if the count is a real tally (the detector separated
          instances, or a single region the silhouette confirms is one lobe), `"no"`
          if the detector collapsed while the outline shows several lobes (the
          overlapping-petal case: nothing local can count it honestly), `"unknown"`
          when there's no silhouette check to confirm either way. Read `count` with
          this in mind: a `count: 1` with `separable: "no"` is a collapse, not a tally.

        When `separable` is `"no"` but the outline still carries the lobe pattern
        (`silhouette.by_radial > 1`), an `estimates` block is added reporting that
        outline count as an actionable number -- it's a *measurement* (angular
        notches in the silhouette), not a judgment, so it stays within "measure,
        don't judge". `count` is never overwritten; `estimates.outline` is the
        number the outline supports, with a `basis` string saying so. Set
        `vqa_estimate=true` to also ask Moondream2 "how many <name>?" and attach
        its answer as `estimates.vqa`, marked as a judgment estimate rather than a
        tally.

        Every result also reports `semantic_ambiguity` (bool) and
        `count_semantics` ("measured_tally", "minimum_visible_instances", or
        "unverified_tally"). When `semantic_ambiguity` is true, an `ambiguity`
        payload is attached detailing why the count represents a collapsed or
        ambiguous structure rather than a confirmed discrete tally.

        A maximally vague `object_name` ("object", "thing", "item", ...) is guarded
        the same way `detect_objects` is: Grounding DINO has the same fallback-to-
        the-whole-frame failure mode on a name with nothing distinctive to point to,
        so a near-full-frame box is dropped rather than counted as a real instance.
        `clip_art=true` gets the equivalent guard through Florence-2's own
        `exclude_full_frame`. Only the exact generic vocabulary triggers this; a
        specific noun that genuinely fills the frame is counted unfiltered.
        """
        app = ctx.request_context.lifespan_context
        with get_images(src) as images:
            backend_pref: Literal["florence2", "auto"] = "florence2" if clip_art else "auto"
            results = execute_detection(
                app,
                images,
                object_name,
                backend_preference=backend_pref,
                threshold=threshold,
                adaptive_threshold=adaptive_threshold,
            )

            for image, result in zip(images, results, strict=True):
                if verify_silhouette:
                    _add_silhouette(app, image, result)
                if consensus:
                    region_label_count = _region_label_consensus(app.florence2, image, object_name)
                    result["consensus"] = {
                        "detector_count": result.get("count"),
                        "region_label_count": region_label_count,
                        "agree": result.get("count") == region_label_count,
                    }
                    result["separable"] = _separability(result)

                # When the detector collapsed overlapping instances but the outline
                # still carries the lobe pattern (by_radial > 1), surface that pattern
                # as an actionable estimate instead of only flagging the collapse. The
                # outline count is a *measurement* (angular notches), not a judgment, so
                # it stays within "measure, don't judge". `count` is never overwritten.
                if result.get("separable") == "no":
                    silhouette = result.get("silhouette") or {}
                    by_radial = silhouette.get("by_radial", 0)
                    if isinstance(by_radial, (int, float)) and by_radial > 1:
                        estimates: dict[str, Any] = {
                            "outline": int(by_radial),
                            "basis": (
                                "outline rosette (angular notches); the detector collapsed "
                                "overlapping instances, but the outline still carries the lobe "
                                "pattern"
                            ),
                        }
                        if vqa_estimate:
                            vqa_answer = app.vqa.query([image], f"How many {object_name}? Answer with one number.")[0]
                            estimates["vqa"] = {
                                "value": _first_int(vqa_answer),
                                "raw": vqa_answer,
                                "note": "a Moondream2 judgment estimate, not a tally",
                            }
                        result["estimates"] = estimates

                # Semantic ambiguity reporting (Spec 15)
                ambiguity_res, count_semantics = check_semantic_ambiguity(result)
                result["semantic_ambiguity"] = ambiguity_res.ambiguous
                result["count_semantics"] = count_semantics
                if ambiguity_res.ambiguous:
                    result["ambiguity"] = ambiguity_res.as_dict()

            if return_annotated:
                _attach_annotated_images(images, results, fallback_label=object_name)

            return results

    @mcp.tool()
    def spatial_relations(
        ctx: Context[AppContext],
        src: ImagePath,
        objects: Annotated[
            list[str],
            Field(description="Names of the objects to locate and compare, e.g. ['hand', 'sword', 'shield']."),
        ],
    ) -> dict[str, Any]:
        """Measure how named objects in an image sit relative to one another.

        Locates each object, segments it, and reports measurements that are hard
        to judge by eye: whether two things actually touch, how many pixels apart
        they are, how much of one lies inside the other and how deeply, plus each
        object's own elongation, straightness and end-to-end width profile.

        This reports geometry, not verdicts — it does not decide what is wrong.
        Interpret the numbers against what the scene ought to look like: a hand
        and the grip it holds that come back `separate` with a large `gap` are not
        in contact; a hand `overlapping` a shield with `a_inside_b` near 1.0 and a
        large `embed_depth` is buried in the shield face rather than gripping its
        rim. `a_inside_b` counts only material overlap, so tea in a cup seen from
        above -- the cup segmented as its rim -- reads 0 there, while
        `a_enclosed_by_b` (the share of a within b's outline, holes filled) reads
        1.0: use the enclosed figures for "is X in Y"; an open shape encloses
        nothing. In a flat image, something held *in front of* an object also
        sits within its outline (a microphone before a face): on 85 labelled
        pairs, enclosed > 0.5 held for 84% of containment pairs, 19% of contact
        pairs and 9% of separate ones, so read it with the scene in mind. An
        elongated object whose `end_symmetry` is near 1.0 is equally wide
        at both ends, unlike a blade that tapers to a point at one end only.

        Useful for checking whether a generated or edited image holds together
        physically, for verifying that an object is where it should be relative to
        another, and for any question of contact, containment or clearance that a
        bounding box cannot answer — boxes overlap whenever one object is simply
        in front of another.

        Takes the single best-scoring match per name, so this assumes one instance of
        each named object. Asked for 'red circle'/'blue circle'/'green circle' on a
        scene with one of each, the detector returned the same three boxes for every
        query — color alone doesn't reliably discriminate same-shaped objects — but the
        correctly-matching box scored highest every time, which is what this relies on.
        For several instances of one kind of thing, give them distinguishing names, or
        use `count_objects` for a tally instead. The same rule can pick the wrong one of
        several: on 36 labelled photos, a racket, a violin and a surfboard each came
        back `separate` from the person holding them because the best-scoring match for
        one name was a different person or object of that name elsewhere in the frame.
        Check each returned `box` sits on the object you meant before trusting a "not
        touching" result.

        `locator` names the model that found and segmented the objects: SAM 3, or
        Grounding DINO + SAM2 if SAM 3 can't load.

        A maximally vague name in `objects` ("object", "thing", "item", ...) is guarded
        the same way `detect_objects`/`count_objects` are: a near-full-frame box from a
        name with nothing distinctive to point to is dropped rather than treated as a
        located match, so it doesn't get reported here as the "best-scoring" box for
        that name.
        """
        app = ctx.request_context.lifespan_context
        with get_images(src) as images:
            image = images[0]

            located, masks, locator = locate_and_segment(app, image, objects, _MAX_RELATED_OBJECTS)

            if not located:
                return {
                    "image_size": [image.width, image.height],
                    "objects": [],
                    "relations": [],
                    "locator": locator,
                    "note": "None of the requested objects were found.",
                }

            for obj, mask in zip(located, masks, strict=True):
                obj.update(geometry.describe(mask))

            relations = [
                {"a": located[i]["id"], "b": located[j]["id"], **geometry.relation(masks[i], masks[j])}
                for i in range(len(masks))
                for j in range(i + 1, len(masks))
            ]

            return {
                "image_size": [image.width, image.height],
                "mask_resolution": SAM3_MASK_DECODE_RESOLUTION if locator == "sam3" else MASK_DECODE_RESOLUTION,
                "locator": locator,
                "objects": located,
                "relations": relations,
            }

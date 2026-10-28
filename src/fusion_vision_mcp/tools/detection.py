"""Detection and measurement tools: `detect_objects`, `dense_region_caption`,
`count_objects` and `spatial_relations`."""

from typing import Annotated, Any, Final, Literal, cast

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
    crowd_range,
    drop_nested,
    lfm_count_check,
    possibly_more,
)
from fusion_vision_mcp.constants import (
    COUNT_BOX_THRESHOLD,
    COUNT_RANGE_FLOOR,
    CROWD_RANGE_RUN_UPPER,
    DETECT_BOX_THRESHOLD,
    MASK_DECODE_RESOLUTION,
)
from fusion_vision_mcp.detection_policy import execute_detection
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
        images, screenshots and documents included. Boxes come highest score first,
        and the first box is the one to trust: it landed on a labelled instance 71-73%
        of the time on sprint R's photos. The rest are candidates, not finds, whatever
        their score: in crowded scenes it returns dozens (a pile of apples came back as
        163 boxes for 8 labelled), many of them parts, duplicates or a second guess at
        one object, so pooled over all boxes only about a quarter land on a labelled
        instance. Use `count_objects` for "how many". If SAM 3 cannot
        be loaded (it is gated on Hugging Face), this falls back to Grounding DINO
        at 0.35, and `detector` says which ran.

        Flat images (clip art, logos, vector charts, plain UI screenshots: few
        distinct colours) use SAM 3 too, since 2026-10-14. On 88 fresh flat images
        it boxed an absent object on 11% where Florence-2's grounding head, used
        there before, boxed something on 100%, and it found 54% of labelled objects
        against 41%. It can return many boxes on a busy screenshot (over 20 for
        about 1 query in 13), so read `scores`. If SAM 3 cannot load, flat images
        fall back to Florence-2, which has no `scores` and echoes the query as its
        label (asked for "elephant" on a drawing of a dog, it boxes the dog).
        On every path, a box does not confirm the object is
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
                    "When true, also tally how many `dense_region_caption` labels match the object "
                    "name (Florence-2's dense region head) and report it under `consensus`. Off by "
                    "default since 2026-10-27: on 644 generalist photo counts the tally equalled the "
                    "truth 13% of the time, and the count was right *less* often when the two agreed "
                    "(43%) than when they disagreed (66%), so agreement is not evidence. Adds a "
                    "Florence-2 dense-region pass (a few seconds)."
                )
            ),
        ] = False,
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
                    "this path (Florence-2's grounding head carries no per-box confidence); a box "
                    "around the whole group is dropped as on the default path."
                )
            ),
        ] = False,
        lfm_check: Annotated[
            bool,
            Field(
                description=(
                    "On photos at the default threshold, let LFM2.5-VL-3B (the `query_image` model) confirm a weak "
                    "single box or rescue a zero: a count of 1 whose box scores under 0.6 becomes 0 if it answers "
                    "'no' to 'Is there a <name>?', and a count of 0 with a candidate box just under the threshold "
                    "becomes 1 if it counts 1 or more and answers 'yes'. On 227 fresh photo counts this took exact "
                    "counts of 1-5 from 63.9% to 68.1% and absent objects counted from 74.5% to 69.1%. Asked on most "
                    "photo counts, about 4 s per question (one or two), and loads the ~8 GB model if `caption` or "
                    "`query_image` has not. Set false to skip it; if the model cannot load, the detector's count "
                    "stands and `count_check.skipped` says why."
                )
            ),
        ] = True,
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
        detections that enclosed the whole arrangement rather than one instance, and
        `nested_boxes_dropped` those that were a part or a near-duplicate of another
        counted box (a box 90%+ inside a larger one, or IoU 0.55+ with a higher one),
        which on a single object otherwise counts it twice.

        On photos (Grounding DINO) the result also carries `possibly_more:
        {upper, extra_candidates, reliability}`: boxes scoring just under the
        threshold that are neither duplicates of nor inside a counted box, so the
        count may be too low by up to `extra_candidates`. It is a hint and only
        points upward. Measured on 953 photo counts: when `upper` > `count` the
        true count was higher than `count` about half the time (38-58%); when
        `upper` == `count` it was too low only 7-14% of the time, though it can
        still be too high. `count` is never changed by it.

        On crowds, photos also get `crowd_range: {low, high, sam3_count,
        reliability}`: Grounding DINO's count tends to run low on crowds and SAM 3's
        (the `detect_objects` model, at a stricter floor) high, so the range between
        them brackets the truth. SAM 3 runs only when `possibly_more.upper` is 3 or
        more, and the range appears when either count is 4 or more. On 92
        fresh photos of 4-30 objects the true count lay inside it 85.9% of the time
        when SAM 3 always ran (80.4% replayed with that gate, the rest falling back to
        `possibly_more`), against 65.2% for `possibly_more`'s range, and its typical width was 0.6 of
        the count. Prefer it to `count` for a crowd. Its appearing does not mean the
        scene is a crowd: on a generalist set where 90% of counts were one object, it
        showed on 16% of counts, nearly all of them 1-3 objects that one model
        overcounted (often screenshots and charts), and held the truth only about
        55-60% of the time. Use it when you expect many objects; when you expect one
        or a few, read its appearance as a sign `count` may be too high. Running SAM 3
        adds about 7 s to a photo count; about two-thirds of generalist counts skip it.
        Absent if SAM 3 is unavailable or skipped.

        On photos at the default threshold (`lfm_check`, on by default) a language
        model may change the count by one step, and only these two: a single box
        scoring under 0.6 is dropped when LFM2.5-VL says the object is not there
        (`count_basis: "presence_veto"`, the box kept under `vetoed_boxes`), and a
        zero becomes 1 when a candidate box sits just under the threshold and LFM
        both counts 1 or more and says it is there (`"presence_rescue"`; that box's
        score is under the threshold). `detector_count` is always the detector's own
        number, which `possibly_more` and `crowd_range` describe; `count_check`
        carries the model's replies. Objects that are absent are still counted as
        present far too often (69.1% of fresh hard negatives), so a count of 1 is not
        proof the object is there.

        Accuracy depends heavily on the scene. Flat images (synthetic shapes, clip art,
        vector charts, lossless UI captures) are counted by Florence-2: on shapes whose
        counts are exact by construction it is right on 5 of 10 cases and holds all 8
        single-object controls -- separated shapes count right, but touching or
        overlapping ones (a ring of 8 touching discs, a grid of 30) merge into one box.
        On 187 flat-image counts it was exact 68.4% of the time (50.8% before nested boxes
        were dropped), and nested shapes (concentric circles) can be undercounted. On real
        photographs it is reliable for a few instances and weak for many: it was exact on
        61.8% of 427 held-out photo counts (383 of them 3 or fewer), and on 65.4% of 644
        generalist photo counts (mostly single objects), and crowded scenes remain far less
        reliable. Treat a count above a handful on a real photo as a rough estimate,
        and read `crowd_range` with it.

        Three measured limits, worth checking before trusting a count:

        Crowded real scenes (see above) are the biggest one.

        Heavy overlap undercounts. With Grounding DINO (the photo path), eight identical
        shapes in a ring count as 8 when separated or touching, but 6 once they overlap
        by roughly two-thirds of their width; on a flat image even touching shapes can
        merge into one. A count below what you expect means "could not separate them", not a
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

        Two fields surface structural ambiguity instead of hiding it:

        - `consensus` (only with `consensus=true`): `{detector_count,
          region_label_count, agree}` -- a tally of Florence-2 dense-region labels
          containing the object name. On generalist photos it was not evidence either
          way (the count was right less often when they agreed), so it is off by default.
        - `separable` (always): `"yes"` if the count is a real tally (the detector separated
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
                candidates_floor=COUNT_RANGE_FLOOR,
                drop_group_box=True,
            )

            for image, result in zip(images, results, strict=True):
                drop_nested(result)
                candidates = result.pop("candidates_below", None)
                if candidates is not None:
                    result["possibly_more"] = possibly_more(result.get("bboxes") or [], candidates)
                    # Gate K measured the range against the default Grounding DINO count only; SAM 3 is skipped
                    # unless Grounding DINO itself sees room for 3 or more (`CROWD_RANGE_RUN_UPPER`).
                    if (
                        threshold == COUNT_BOX_THRESHOLD
                        and adaptive_threshold
                        and result["possibly_more"]["upper"] >= CROWD_RANGE_RUN_UPPER
                    ):
                        span = crowd_range(app, image, object_name, int(result.get("count") or 0))
                        if span is not None:
                            result["crowd_range"] = span
                    # Gates 1b/1c/2 measured these after the detector's count, at the default threshold only.
                    if lfm_check and threshold == COUNT_BOX_THRESHOLD and adaptive_threshold:
                        lfm_count_check(app, image, object_name, result, candidates)
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
        use `count_objects` for a tally instead.

        A maximally vague name in `objects` ("object", "thing", "item", ...) is guarded
        the same way `detect_objects`/`count_objects` are: a near-full-frame box from a
        name with nothing distinctive to point to is dropped rather than treated as a
        located match, so it doesn't get reported here as the "best-scoring" box for
        that name.
        """
        app = ctx.request_context.lifespan_context
        with get_images(src) as images:
            image = images[0]

            located: list[dict[str, Any]] = []
            for call_index, object_name in enumerate(objects):
                if len(located) >= _MAX_RELATED_OBJECTS:
                    break
                detected = execute_detection(app, [image], object_name, backend_preference="grounding_dino")[0]
                if not detected["bboxes"]:
                    continue
                best = max(range(len(detected["bboxes"])), key=lambda i: detected["scores"][i])
                box = detected["bboxes"][best]
                located.append(
                    {"id": f"{object_name}#{call_index}", "label": object_name, "box": [int(v) for v in box]}
                )

            if not located:
                return {
                    "image_size": [image.width, image.height],
                    "objects": [],
                    "relations": [],
                    "note": "None of the requested objects were found.",
                }

            masks = app.segmenter.segment(image, [cast(list[int], obj["box"]) for obj in located])
            for obj, mask in zip(located, masks, strict=True):
                obj.update(geometry.describe(mask))

            relations = [
                {"a": located[i]["id"], "b": located[j]["id"], **geometry.relation(masks[i], masks[j])}
                for i in range(len(masks))
                for j in range(i + 1, len(masks))
            ]

            return {
                "image_size": [image.width, image.height],
                "mask_resolution": MASK_DECODE_RESOLUTION,
                "objects": located,
                "relations": relations,
            }

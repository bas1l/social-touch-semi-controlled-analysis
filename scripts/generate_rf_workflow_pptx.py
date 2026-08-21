"""Build a PowerPoint walkthrough of the receptive-field mapping workflow.

Assembles the seven curated figures (see ``generate_rf_workflow_doc_figures.py``)
and the stage narrative into a 16:9 slide deck aimed at newcomers.  Worked
example: session ``2022-06-17_ST16-02`` (ST16-02).

Output: ``docs/receptive_field_workflow/rf_workflow_walkthrough.pptx``.

Run from an activated ``social-touch-analysis`` env:

    python scripts/generate_rf_workflow_pptx.py
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Emu, Inches, Pt

REPO_ROOT = Path(__file__).resolve().parents[1]
DOC_DIR = REPO_ROOT / "docs" / "receptive_field_workflow"
FIG = DOC_DIR / "figures"
OUT = DOC_DIR / "rf_workflow_walkthrough.pptx"

# 16:9
SLIDE_W = Inches(13.333)
SLIDE_H = Inches(7.5)

# Palette
NAVY = RGBColor(0x1F, 0x2A, 0x44)
ACCENT = RGBColor(0xE0, 0x6A, 0x2B)  # inferno-ish orange
INK = RGBColor(0x22, 0x22, 0x22)
GREY = RGBColor(0x66, 0x66, 0x66)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
LIGHT = RGBColor(0xF4, 0xF2, 0xEE)


def _require(p: Path) -> Path:
    if not p.exists():
        raise FileNotFoundError(f"generate_rf_workflow_pptx: missing asset: {p}")
    return p


def add_image_fit(slide, path: Path, x, y, w, h):
    """Place *path* fitted (contain) inside the box (x, y, w, h), centred."""
    _require(path)
    iw, ih = Image.open(path).size
    box_w, box_h = float(w), float(h)
    scale = min(box_w / iw, box_h / ih)
    dw, dh = iw * scale, ih * scale
    px = x + (box_w - dw) / 2
    py = y + (box_h - dh) / 2
    slide.shapes.add_picture(str(path), Emu(int(px)), Emu(int(py)),
                             Emu(int(dw)), Emu(int(dh)))


def add_rect(slide, x, y, w, h, color):
    from pptx.enum.shapes import MSO_SHAPE
    shp = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, y, w, h)
    shp.fill.solid()
    shp.fill.fore_color.rgb = color
    shp.line.fill.background()
    shp.shadow.inherit = False
    return shp


def add_text(slide, x, y, w, h, runs, *, align=PP_ALIGN.LEFT,
             anchor=MSO_ANCHOR.TOP, space_after=6):
    """runs: list of paragraphs; each paragraph is a list of (text, opts) runs."""
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    for i, para in enumerate(runs):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.space_after = Pt(space_after)
        p.space_before = Pt(0)
        for text, opts in para:
            r = p.add_run()
            r.text = text
            f = r.font
            f.size = Pt(opts.get("size", 18))
            f.bold = opts.get("bold", False)
            f.italic = opts.get("italic", False)
            f.name = opts.get("name", "Segoe UI")
            f.color.rgb = opts.get("color", INK)
        if "level" in para_opts(para):
            p.level = para_opts(para)["level"]
    return tb


def para_opts(_para):
    return {}


def bullets(slide, x, y, w, h, items, *, size=17, color=INK, gap=8):
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    for i, (txt, lvl) in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(gap)
        p.level = lvl
        bullet = "•  " if lvl == 0 else "–  "
        r = p.add_run()
        r.text = bullet + txt
        r.font.size = Pt(size if lvl == 0 else size - 2)
        r.font.name = "Segoe UI"
        r.font.color.rgb = color
    return tb


def title_band(slide, kicker, title):
    add_rect(slide, 0, 0, SLIDE_W, Inches(1.15), NAVY)
    add_rect(slide, 0, Inches(1.15), SLIDE_W, Pt(4), ACCENT)
    if kicker:
        add_text(slide, Inches(0.5), Inches(0.12), Inches(12.3), Inches(0.35),
                 [[(kicker, {"size": 13, "bold": True, "color": ACCENT,
                             "name": "Segoe UI Semibold"})]])
    add_text(slide, Inches(0.5), Inches(0.40), Inches(12.3), Inches(0.7),
             [[(title, {"size": 26, "bold": True, "color": WHITE})]])


def content_slide(prs, kicker, title, fig_name, blurb_bullets, caption):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_rect(slide, 0, 0, SLIDE_W, SLIDE_H, LIGHT)
    title_band(slide, kicker, title)

    # image left, bullets right
    img_x, img_y = Inches(0.35), Inches(1.45)
    img_w, img_h = Inches(8.15), Inches(5.15)
    add_image_fit(slide, FIG / fig_name, img_x, img_y, img_w, img_h)

    if caption:
        add_text(slide, img_x, Inches(6.7), img_w, Inches(0.7),
                 [[(caption, {"size": 12, "italic": True, "color": GREY})]])

    bullets(slide, Inches(8.75), Inches(1.6), Inches(4.25), Inches(5.4),
            blurb_bullets, size=17)
    return slide


def wide_slide(prs, kicker, title, fig_name, caption, takeaway):
    """Full-width image below the title band, with a takeaway line and caption."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_rect(slide, 0, 0, SLIDE_W, SLIDE_H, LIGHT)
    title_band(slide, kicker, title)
    if takeaway:
        add_text(slide, Inches(0.5), Inches(1.35), Inches(12.3), Inches(0.5),
                 [[(takeaway, {"size": 16, "bold": True, "color": NAVY})]])
    add_image_fit(slide, FIG / fig_name, Inches(0.35), Inches(1.95),
                  Inches(12.6), Inches(4.5))
    if caption:
        add_text(slide, Inches(0.5), Inches(6.6), Inches(12.3), Inches(0.8),
                 [[(caption, {"size": 12, "italic": True, "color": GREY})]])
    return slide


def build():
    prs = Presentation()
    prs.slide_width = SLIDE_W
    prs.slide_height = SLIDE_H
    blank = prs.slide_layouts[6]

    # ---- 1. Title ----
    s = prs.slides.add_slide(blank)
    add_rect(s, 0, 0, SLIDE_W, SLIDE_H, NAVY)
    add_rect(s, Inches(0.9), Inches(3.7), Inches(2.4), Pt(6), ACCENT)
    add_text(s, Inches(0.9), Inches(2.2), Inches(11.5), Inches(1.5),
             [[("How a Receptive Field is Built", {"size": 44, "bold": True, "color": WHITE})]])
    add_text(s, Inches(0.9), Inches(3.95), Inches(11.5), Inches(1.0),
             [[("A walkthrough of the receptive-field mapping pipeline",
                {"size": 22, "color": RGBColor(0xC9, 0xD2, 0xE0)})]])
    add_text(s, Inches(0.9), Inches(6.4), Inches(11.5), Inches(0.6),
             [[("Worked example: session 2022-06-17_ST16-02 (ST16-02)   ·   "
                "microneurography + psychophysics",
                {"size": 14, "color": RGBColor(0x8A, 0x96, 0xAB)})]])

    # ---- 2. What & overview ----
    s = prs.slides.add_slide(blank)
    add_rect(s, 0, 0, SLIDE_W, SLIDE_H, LIGHT)
    title_band(s, "ORIENTATION", "What we are building")
    add_text(s, Inches(0.5), Inches(1.5), Inches(12.3), Inches(1.2),
             [[("A neuron is one microneurography unit. Every millisecond we know its ",
                {"size": 18}),
               ("instantaneous firing frequency (IFF)", {"size": 18, "bold": True}),
               (" and where the forearm was touched. The ", {"size": 18}),
               ("receptive field", {"size": 18, "bold": True}),
               (" is the patch of skin where touch reliably drives the neuron.",
                {"size": 18})]])
    stages = [
        ("0 · Preprocessing", "Merged CSV → clean per-touch contacts on the forearm; unwrap the 3-D surface to a flat 2-D map."),
        ("1 · Reduce a touch", "Each single touch → a sparse per-vertex IFF map."),
        ("2 · Aggregate", "Average all touches → one smooth heatmap on the 2-D surface."),
        ("3 · Define the RF", "Trace a boundary around the responsive region; measure its area, centroid, shape."),
    ]
    y = Inches(3.0)
    for num, desc in stages:
        add_rect(s, Inches(0.5), y, Inches(3.15), Inches(0.9), NAVY)
        add_text(s, Inches(0.6), y, Inches(2.95), Inches(0.9),
                 [[(num, {"size": 16, "bold": True, "color": WHITE})]],
                 anchor=MSO_ANCHOR.MIDDLE)
        add_text(s, Inches(3.85), y, Inches(9.0), Inches(0.9),
                 [[(desc, {"size": 16, "color": INK})]], anchor=MSO_ANCHOR.MIDDLE)
        y += Inches(1.02)

    # ---- 3. Stage 0 preprocessing ----
    wide_slide(
        prs, "STAGE 0 · PREPROCESSING", "Inputs & preprocessing",
        "01_raw_input.png",
        "Merged CSV (1 kHz) + forearm point cloud → grouped, de-duplicated, "
        "snapped per-touch contacts.",
        "① the initial merged CSV (contacts, spikes, IFF, IDs per 1 kHz row); "
        "② the input forearm point cloud (17,578 RF-centred vertices); ③ the preprocessing "
        "steps; ④ session summary — 755 touches, 280 M contact points, 752 spike touches.")

    # ---- 4. Stage 0 SLIM ----
    content_slide(
        prs, "STAGE 0 · PREPROCESSING", "Unwrapping the forearm (SLIM UV)",
        "02_slim_unwrap.png",
        [("The forearm is curved — area & boundaries are easier on a flat map", 0),
         ("run_slim_pipeline_core builds + cleans a mesh, then flattens it", 0),
         ("Every vertex gets a 2-D (U, V) coordinate", 0),
         ("IFF-weighted centroid = origin, so the RF sits near (0, 0)", 0),
         ("Cached once per session as <session>_slim_uv.npz", 1)],
        "Left: 3-D forearm mesh. Right: the same mesh flattened to the UV plane — "
        "every later figure lives here.")

    # ---- 5. Stage 1 single touch ----
    content_slide(
        prs, "STAGE 1 · REDUCE A SINGLE TOUCH", "One touch → a sparse map",
        "03_single_touch.png",
        [("_compute_touch_rf accumulates depth-weighted IFF at touched vertices", 0),
         ("Weighted mean (np.add.at) and unweighted max (np.maximum.at) per vertex", 0),
         ("Drops vertices where the neuron signal was NaN", 0),
         ("Output: a short list of (vertex, mean-IFF) pairs per touch", 0),
         ("Saved to single_touch_rf_maps_mean.npz / _max.npz", 1)],
        "A representative proximal stroke (touch #166, 576 vertices) coloured by mean IFF, "
        "on the forearm (left) and the UV unwrap (right). Note the hot spot fading to the edges.")

    # ---- 5b. Stage 1 · frame by frame ----
    wide_slide(
        prs, "STAGE 1 · INSIDE ONE TOUCH", "Frame by frame — the RF in miniature",
        "03a_touch_frames.png",
        "Touch #166 = 200 frames at 1 kHz, but contact updates at ~30 Hz → 6 contact frames.",
        "The fingertip sweeps proximally; the neuron fires at ~5 Hz (frames 1–3), climbs to "
        "22 Hz (frame 4), spikes to 79 Hz (frame 5), then drops to 3 Hz (frame 6). It responds "
        "strongly only where the contact crosses one patch of skin — that patch is the receptive field.")

    # ---- 5c. Stage 1 · grouping & averaging ----
    wide_slide(
        prs, "STAGE 1 · INSIDE ONE TOUCH", "Grouping & averaging (mean)",
        "03b_grouping_averaging.png",
        "_compute_touch_rf: each vertex accumulates w·IFF over the frames that touch it, then Σ w·IFF ÷ Σ w.",
        "Each contact point's weight w is how deeply it was pressed relative to the deepest "
        "point of its own frame, raised to depth_weight_alpha. Left: Σ w·IFF per vertex. "
        "Middle: Σ w — evidence, not a frame count (it reduces to the frame count at alpha = 0). "
        "Right: mean = Σ w·IFF / Σ w — the 576-value sparse map. The bright core is where the "
        "high-IFF frame 5 landed.")

    # ---- 5d. Stage 1 · attributes ----
    s = prs.slides.add_slide(blank)
    add_rect(s, 0, 0, SLIDE_W, SLIDE_H, LIGHT)
    title_band(s, "STAGE 1 · INSIDE ONE TOUCH", "This touch's attributes")
    add_image_fit(s, FIG / "03c_attributes.png", Inches(0.4), Inches(1.45),
                  Inches(7.6), Inches(5.6))
    bullets(s, Inches(8.35), Inches(1.9), Inches(4.6), Inches(5.0),
            [("Every touch carries stimulus + kinematic context", 0),
             ("Instructed: light, one-finger-tip proximal stroke @ 18 cm/s", 0),
             ("Measured: ~210 mm/s, 6.7 mm depth, 187 mm² area", 0),
             ("Skin mechanics: ~448 kPa stress (MoS model)", 0),
             ("Neuron: 20 Hz mean, 282 Hz peak over the touch", 0),
             ("Used later to relate RF responses to how the skin was touched", 1)],
            size=16)

    # ---- 6. Stage 2 aggregate ----
    content_slide(
        prs, "STAGE 2 · AGGREGATE TOUCHES", "Averaging into one heatmap",
        "04_aggregate.png",
        [("compute_rf_heatmap averages each vertex across all touches", 0),
         ("25% overlap threshold drops fringe vertices (few touches)", 0),
         ("Island cleanup keeps the blob containing the peak", 0),
         ("PCA-align, then interpolate to a smooth 150×150 grid (grid_z)", 0),
         ("This grid IS the neuron's response field", 1)],
        "ST16-02 over all 755 touches (threshold = 189 touches). Left: per-vertex averages; "
        "right: interpolated grid_z with the fitted RF boundary (red) and centroid (+).")

    # ---- 7. Stage 3 boundary ----
    content_slide(
        prs, "STAGE 3 · DEFINE THE RF", "Radial “foot of the mountain” boundary",
        "05_boundary_steps.png",
        [("Treat the heatmap as a hill; the RF edge is where it flattens", 0),
         ("compute_radial_foot_boundary uses the Hessian λmax field", 0),
         ("Rays from the peak stop at the first concave-up point", 0),
         ("Curve is clipped to the actually-touched footprint", 0),
         ("Alternatives: gradient-ridge & inflection (Laplacian)", 1)],
        "Left: IFF field with the red radial-foot contour and peak (×). "
        "Right: the Hessian λmax field the rays are traced on.")

    # ---- 7b. Stage 3 deep dive · Hessian lmax ----
    wide_slide(
        prs, "STAGE 3 · IN DEPTH", "The Hessian λmax field",
        "05a_hessian_lmax.png",
        "λmax = larger principal curvature of the Gaussian-smoothed IFF surface "
        "(gauss σ=8, Hessian σ=5).",
        "λmax < 0 over the concave-down dome (blue), > 0 at the concave-up foot (red ring). "
        "The RF edge is that sign flip — not a fixed IFF threshold — so it adapts to each "
        "neuron's dome shape.")

    # ---- 7c. Stage 3 deep dive · ray profile ----
    wide_slide(
        prs, "STAGE 3 · IN DEPTH", "Contour selection — one ray",
        "05b_ray_profile.png",
        "360 rays from the peak; per ray, find_peaks locates the first positive λmax plateau.",
        "Along each ray λmax rises out of the dome (blue), crosses zero, and the first "
        "positive plateau centre is the foot radius. Each ray is snapped back to the last "
        "painted cell (dotted) so the foot never lands on non-contacted skin; the 360 radii "
        "are Savitzky-Golay smoothed.")

    # ---- 7d. Stage 3 deep dive · envelope ----
    wide_slide(
        prs, "STAGE 3 · IN DEPTH", "Contour selection — footprint envelope",
        "05c_envelope.png",
        "The star-convex radial curve is clipped to the painted footprint (grid_z > 0).",
        "One-radius-per-angle chords bulge into non-painted cells where the footprint is "
        "concave (left, orange). Pass 2 intersects with grid_z > 0, keeps the peak's "
        "connected component, and re-traces it (right, green) — fail-fast, no fallback ring.")

    # ---- 8. Stage 3 outline / metrics ----
    content_slide(
        prs, "STAGE 3 · DEFINE THE RF", "The receptive field & its metrics",
        "06_rf_outline.png",
        [("From the closed contour the pipeline derives:", 0),
         ("Area in real mm² (mapped back via uv_points_to_xyz)", 1),
         ("Perimeter, circularity, centroid, peak", 1),
         ("PCA ellipse: major/minor axis + orientation", 1),
         ("Saved to <session>_population_response_fields.npz", 0)],
        "The final receptive field for ST16-02: red outline on the interpolated heatmap "
        "(cropped, on the skin texture), iso-IFF contour lines inside.")

    # ---- 9. Stage 4 profile ----
    content_slide(
        prs, "STAGE 4 · PROFILES (DOWNSTREAM)", "Cross-section through the RF",
        "07_profile_1d.png",
        [("run_rf_profile_extraction slices through the centroid", 0),
         ("Characterises RF shape and edge sharpness", 0),
         ("Two gradient peaks = the RF's sharp edges", 0),
         ("Green diamonds = where the boundary crosses the axis", 1)],
        "Gradient magnitude |∇IFF| along the U axis through the centroid.")

    # ---- 10. Where this runs ----
    s = prs.slides.add_slide(blank)
    add_rect(s, 0, 0, SLIDE_W, SLIDE_H, LIGHT)
    title_band(s, "PIPELINE", "Where this runs")
    add_text(s, Inches(0.5), Inches(1.45), Inches(12.3), Inches(0.6),
             [[("DAG tasks in configs/analyse_workflow_processing_dag.yaml, chained by dependency:",
                {"size": 17, "color": INK})]])
    chain = [
        ("touch_prepare_sessions  +  touch_compute_series", "clean CSVs, kinematics"),
        ("spatial_map_single_touch", "Stage 1 — single_touch_rf_maps_*.npz"),
        ("spatial_precompute_slim_uv", "Stage 0 — <session>_slim_uv.npz"),
        ("spatial_extract_boundaries", "Stages 2 & 3 — population_response_fields.npz + PNGs"),
        ("spatial_extract_rf_profiles", "Stage 4 — 1-D profiles"),
    ]
    y = Inches(2.15)
    for i, (task, out) in enumerate(chain):
        add_rect(s, Inches(0.8), y, Inches(5.3), Inches(0.72), NAVY)
        add_text(s, Inches(0.95), y, Inches(5.1), Inches(0.72),
                 [[(task, {"size": 15, "bold": True, "color": WHITE, "name": "Consolas"})]],
                 anchor=MSO_ANCHOR.MIDDLE)
        add_text(s, Inches(6.35), y, Inches(6.4), Inches(0.72),
                 [[(out, {"size": 15, "color": INK})]], anchor=MSO_ANCHOR.MIDDLE)
        y += Inches(0.9)
        if i < len(chain) - 1:
            add_text(s, Inches(3.1), y - Inches(0.22), Inches(1.0), Inches(0.25),
                     [[("▼", {"size": 12, "color": ACCENT})]], align=PP_ALIGN.CENTER)
    add_text(s, Inches(0.8), Inches(6.75), Inches(12.0), Inches(0.5),
             [[("Key options on spatial_extract_boundaries:  iff_metric=mean · "
                "min_overlap_pct=25 · boundary_method=radial · flip_u=true",
                {"size": 13, "italic": True, "color": GREY})]])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(OUT))
    print(f"Saved {OUT}  ({len(prs.slides._sldIdLst)} slides)")


if __name__ == "__main__":
    build()

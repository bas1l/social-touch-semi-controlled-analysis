"""Build a PowerPoint deck for ``docs/spatial_extract_boundaries``.

Focused companion to the RF walkthrough deck: covers the
``spatial_extract_boundaries`` step for the **radial "foot of mountain"** method
only — the method the pipeline is actually configured to run
(``boundary_method: radial``) — as an ordered walk through its subprocesses with
the actual DAG parameter values.

Uses the three figures written by ``generate_boundary_workflow_figures.py``.
Worked example: session ``2022-06-17_ST16-02`` (ST16-02).

Output: ``docs/spatial_extract_boundaries/spatial_extract_boundaries.pptx``.

Run from an activated ``social-touch-analysis`` env:

    python scripts/generate_boundary_workflow_pptx.py
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

REPO_ROOT = Path(__file__).resolve().parents[1]
DOC_DIR = REPO_ROOT / "docs" / "spatial_extract_boundaries"
FIG = DOC_DIR / "figures"
OUT = DOC_DIR / "spatial_extract_boundaries.pptx"

SLIDE_W = Inches(13.333)
SLIDE_H = Inches(7.5)

NAVY = RGBColor(0x1F, 0x2A, 0x44)
ACCENT = RGBColor(0xE0, 0x6A, 0x2B)
INK = RGBColor(0x22, 0x22, 0x22)
GREY = RGBColor(0x66, 0x66, 0x66)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
LIGHT = RGBColor(0xF4, 0xF2, 0xEE)
RADIAL_C = RGBColor(0xD6, 0x27, 0x28)


def _require(p: Path) -> Path:
    if not p.exists():
        raise FileNotFoundError(f"generate_boundary_workflow_pptx: missing asset: {p}")
    return p


def add_image_fit(slide, path: Path, x, y, w, h):
    _require(path)
    iw, ih = Image.open(path).size
    scale = min(float(w) / iw, float(h) / ih)
    dw, dh = iw * scale, ih * scale
    px = x + (float(w) - dw) / 2
    py = y + (float(h) - dh) / 2
    slide.shapes.add_picture(str(path), Emu(int(px)), Emu(int(py)),
                             Emu(int(dw)), Emu(int(dh)))


def add_rect(slide, x, y, w, h, color):
    shp = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, y, w, h)
    shp.fill.solid()
    shp.fill.fore_color.rgb = color
    shp.line.fill.background()
    shp.shadow.inherit = False
    return shp


def add_text(slide, x, y, w, h, runs, *, align=PP_ALIGN.LEFT,
             anchor=MSO_ANCHOR.TOP, space_after=6):
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
    return tb


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
    img_x, img_y = Inches(0.35), Inches(1.45)
    img_w, img_h = Inches(8.15), Inches(5.15)
    add_image_fit(slide, FIG / fig_name, img_x, img_y, img_w, img_h)
    if caption:
        add_text(slide, img_x, Inches(6.7), img_w, Inches(0.7),
                 [[(caption, {"size": 12, "italic": True, "color": GREY})]])
    bullets(slide, Inches(8.75), Inches(1.6), Inches(4.25), Inches(5.4),
            blurb_bullets, size=16)
    return slide


def wide_slide(prs, kicker, title, fig_name, caption, takeaway):
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
    add_text(s, Inches(0.9), Inches(2.0), Inches(11.5), Inches(1.7),
             [[("Extracting the RF Boundary", {"size": 44, "bold": True, "color": WHITE})]])
    add_text(s, Inches(0.9), Inches(3.95), Inches(11.5), Inches(1.0),
             [[("The spatial_extract_boundaries step — the radial “foot of mountain” workflow",
                {"size": 21, "color": RGBColor(0xC9, 0xD2, 0xE0)})]])
    add_text(s, Inches(0.9), Inches(6.4), Inches(11.5), Inches(0.6),
             [[("Worked example: session 2022-06-17_ST16-02 (ST16-02)   ·   gesture = all",
                {"size": 14, "color": RGBColor(0x8A, 0x96, 0xAB)})]])

    # ---- 2. The input grid_z ----
    s = prs.slides.add_slide(blank)
    add_rect(s, 0, 0, SLIDE_W, SLIDE_H, LIGHT)
    title_band(s, "INPUT", "How grid_z is built")
    add_text(s, Inches(0.5), Inches(1.5), Inches(12.3), Inches(1.0),
             [[("The boundary method consumes one 2-D field, ", {"size": 18}),
               ("grid_z", {"size": 18, "bold": True, "name": "Consolas"}),
               (" — a 150×150 mean-IFF heatmap on the flattened forearm surface, "
                "NaN outside the touched footprint.", {"size": 18})]])
    bullets(s, Inches(0.6), Inches(2.7), Inches(12.2), Inches(4.3),
            [("Per-vertex mean IFF across the gesture's touches — iff_metric: mean "
              "(compute_rf_heatmap)", 0),
             ("Coverage threshold — drop vertices touched by < min_overlap_pct % (25%)", 0),
             ("SLIM UV flattening — unwrap the curved mesh to a flat (U,V) plane; "
              "PCA-aligned so the RF long axis is horizontal; flip_u: true", 0),
             ("Harmonic interpolation onto the 150×150 grid (igl.harmonic, ∇²z = 0); "
              "median_filter_size: null (disabled)", 0),
             ("Island cleanup — keep only the blob containing the peak → grid_z", 0)],
            size=17)

    # ---- 3. The subprocess workflow (centerpiece) ----
    wide_slide(
        prs, "WORKFLOW", "The radial-foot subprocesses, in order",
        "bd_01_workflow.png",
        "Smoothing comes first (Gaussian, σ=8), then the Hessian (σ=5), then the rays, "
        "then the footprint clip (σ=1.5).",
        "Build the λmax field (steps 2–4) → trace the foot per ray (5–6) → "
        "clip to the footprint & measure (7–8). The foot is the crest of the first "
        "positive-λmax plateau — a curvature landmark a step past the λmax=0 inflection — "
        "so the edge adapts to the dome, no fixed IFF threshold.")

    # ---- 4. The field progression ----
    wide_slide(
        prs, "WORKFLOW", "One grid_z through the subprocess chain",
        "bd_02_fields.png",
        "grid_z → Gaussian-smoothed (σ=8) → Hessian λmax (σ=5) with the contour → "
        "final foot contour on the heatmap.",
        "λmax is negative (blue) over the concave-down dome and positive (red) on the "
        "concave-up flank; the red contour tracks the crest of the first positive-λmax "
        "plateau, then is clipped to the painted footprint.")

    # ---- 4b. One of the 360 rays — the λmax profile ----
    wide_slide(
        prs, "WORKFLOW", "One of the 360 cast rays",
        "bd_04_ray_section.png",
        "Left: the Hessian λmax heatmap with one ray from the peak. Right: λmax along "
        "that ray — negative (dome cap) → zero (inflection) → positive crest; the traced "
        "contour sits at the crest of the first positive-λmax plateau, a step past the "
        "zero-crossing.",
        "The boundary is a curvature landmark, not an IFF threshold: on each ray the "
        "foot is the centre of the first λmax > 0 plateau — the crest of the concave-up "
        "ridge, a step past the λmax=0 inflection, not the sign-flip itself.")

    # ---- 5. Parameter values ----
    s = prs.slides.add_slide(blank)
    add_rect(s, 0, 0, SLIDE_W, SLIDE_H, LIGHT)
    title_band(s, "CONFIG", "The actual parameter values")
    add_text(s, Inches(0.5), Inches(1.4), Inches(12.3), Inches(0.5),
             [[("configs/analyse_workflow_processing_dag.yaml · spatial_extract_boundaries:",
                {"size": 15, "bold": True, "color": NAVY, "name": "Consolas"})]])
    cfg = [
        "neuron_mode: iff        iff_metric: mean        min_overlap_pct: 25",
        "median_filter_size:     # null (disabled)",
        "flip_u: true            cmap: inferno           contour_color: red",
        "boundary_method: radial",
        "radial_gauss_sigma: 8.0            # step 2 · Gaussian pre-smooth",
        "radial_hess_sigma: 5.0             # step 4 · Hessian derivative kernel",
        "radial_envelope_smooth_sigma: 1.5  # step 7 · footprint envelope smooth",
    ]
    add_rect(s, Inches(0.7), Inches(1.95), Inches(11.9), Inches(3.2),
             RGBColor(0x1B, 0x22, 0x33))
    add_text(s, Inches(0.95), Inches(2.1), Inches(11.5), Inches(2.9),
             [[(line, {"size": 14, "color": RGBColor(0xE6, 0xEA, 0xF2),
                       "name": "Consolas"})] for line in cfg], space_after=8)
    add_text(s, Inches(0.7), Inches(5.5), Inches(12.0), Inches(1.6),
             [[("inflection_sigma: 5.0", {"size": 14, "bold": True, "color": NAVY,
                                          "name": "Consolas"}),
               (" is present in the config but is ", {"size": 14, "color": INK}),
               ("not consumed by the radial path", {"size": 14, "bold": True, "color": INK}),
               (" — it only drives side-computed comparison markers.",
                {"size": 14, "color": INK})],
              [("Setting inflection_sigma: null disables boundary computation entirely.",
                {"size": 13, "italic": True, "color": GREY})]],
             space_after=10)

    # ---- 6. Result + shared metrics ----
    content_slide(
        prs, "OUTPUT", "The boundary → the RF metrics",
        "bd_03_result.png",
        [("The closed contour yields:", 0),
         ("area — shoelace polygon area in UV mm²", 1),
         ("perimeter — summed segment lengths (UV mm)", 1),
         ("centroid — polygon centroid", 1),
         ("circularity — 4·π·area / perimeter²", 1),
         ("PCA ellipse — major/minor axis + orientation", 1),
         ("Written to <session>_population_response_fields.npz "
          "under the boundary_ prefix (+ a boundary_method key)", 0),
         ("Consumed by spatial_extract_rf_profiles", 0)],
        "The final radial RF boundary on ST16-02 (peak ×, centroid +), with the derived "
        "metrics annotated.")

    # ---- 7. Where it runs ----
    s = prs.slides.add_slide(blank)
    add_rect(s, 0, 0, SLIDE_W, SLIDE_H, LIGHT)
    title_band(s, "PIPELINE", "Where it runs")
    chain = [
        "spatial_map_single_touch  +  spatial_precompute_slim_uv",
        "spatial_extract_boundaries   ← this step (grid_z → radial boundary + metrics)",
        "spatial_extract_rf_profiles   (consumes the active boundary contour)",
    ]
    y = Inches(1.9)
    for i, task in enumerate(chain):
        col = ACCENT if i == 1 else NAVY
        add_rect(s, Inches(0.8), y, Inches(11.7), Inches(0.8), col)
        add_text(s, Inches(1.0), y, Inches(11.3), Inches(0.8),
                 [[(task, {"size": 15, "bold": True, "color": WHITE, "name": "Consolas"})]],
                 anchor=MSO_ANCHOR.MIDDLE)
        y += Inches(1.05)
    add_text(s, Inches(0.8), Inches(5.4), Inches(12.0), Inches(1.6),
             [[("Run:  ", {"size": 16, "bold": True, "color": NAVY}),
               ("python scripts/analysis_workflow_processing.py",
                {"size": 16, "color": INK, "name": "Consolas"}),
               ("   — or the GUI runner.", {"size": 16, "color": INK})],
              [("Full write-up: docs/spatial_extract_boundaries/README.md",
                {"size": 14, "italic": True, "color": GREY})]],
             space_after=10)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(OUT))
    print(f"Saved {OUT}  ({len(prs.slides._sldIdLst)} slides)")


if __name__ == "__main__":
    build()

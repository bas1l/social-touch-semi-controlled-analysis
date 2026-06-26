"""Operator descriptions  (shown on double-click in the palette).

Rich-text (Qt HTML subset) explanations of what each operator computes and the
maths behind it.  Maths uses Unicode symbols + <sub>/<sup> because QTextBrowser
does not render LaTeX/MathML.  Keyed by op_id; a fail-fast check below ensures
every operator has an entry.
"""

from __future__ import annotations

from .operator_registry import OPERATORS

OPERATOR_DESCRIPTIONS: dict[str, str] = {
    "gaussian": """
<h3>Gaussian smoothing</h3>
<p>Convolves the heat-map with a 2-D isotropic Gaussian kernel, attenuating
high-spatial-frequency noise while preserving the broad shape of the response.</p>
<p><b>Definition.</b> The smoothed field is the convolution<br>
&nbsp;&nbsp; <i>z</i><sub>σ</sub> = <i>z</i> ∗ <i>G</i><sub>σ</sub>,&nbsp; where &nbsp;
<i>G</i><sub>σ</sub>(u,v) = (1 / 2πσ²) · exp( −(u² + v²) / 2σ² ).</p>
<p><b>NaN-aware normalisation.</b> Because the contacted region is masked, a plain
convolution would bleed zeros inward from the boundary. This operator delegates to
<code>compute_laplacian_arrays</code>, which divides the smoothed signal by the
smoothed mask (Gaussian <i>normalised</i> convolution) so values near the edge are
not biased toward zero, then re-masks the original NaN cells.</p>
<p><b>Parameter.</b></p>
<ul>
<li><b>σ</b> (kernel width, pixels) — the only knob; effective support ≈ 3σ.
<i>Increase</i> σ for stronger smoothing: fine bumps and noise wash out, the peak
broadens, and the valid Gaussian "halo" pushes ~3σ further beyond the data edge.
<i>Decrease</i> σ to keep fine detail but let more noise through; below ~1 px it
barely changes the field. Must be &gt; 0.</li>
</ul>
""",
    "median": """
<h3>Median filter</h3>
<p>Replaces each pixel with the median of its neighbourhood — a non-linear,
edge-preserving denoiser that removes salt-and-pepper spikes without the blurring a
Gaussian causes.</p>
<p><b>Definition.</b> For a window <i>W</i> of size <i>n</i>×<i>n</i> centred at (u,v):<br>
&nbsp;&nbsp; <i>z</i>′(u,v) = median{ <i>z</i>(u+i, v+j) : (i,j) ∈ <i>W</i> }.</p>
<p>Unlike the mean, the median is robust to outliers: a single very large value
cannot drag the result, so isolated hot pixels vanish while genuine plateaus and
step edges stay sharp.</p>
<p><b>Parameter.</b></p>
<ul>
<li><b>size</b> (window edge, pixels) — must be a positive <i>odd</i> integer (so the
window has a unique centre). <i>Increase</i> size to erase larger speckles and flatten
the field more, but large windows progressively round off sharp corners and can
swallow small genuine peaks. size = 1 is a no-op.</li>
</ul>
<p><b>NaN handling.</b> The masked region is nearest-neighbour extrapolated before
filtering, then re-masked.</p>
""",
    "gradient_magnitude": """
<h3>Gradient magnitude |∇z|</h3>
<p>The steepness of the response surface at every point — large on the flanks of the
"mountain" where the field rises or falls quickly, near zero on flat plateaus and at
the summit.</p>
<p><b>Definition.</b> After Gaussian smoothing at scale σ, the first partial
derivatives are taken and combined:<br>
&nbsp;&nbsp; |∇z| = √( (∂z/∂u)² + (∂z/∂v)² ).</p>
<p>This is rotation-invariant: it measures slope regardless of direction. Gradient
boundary methods locate the receptive-field edge along the ridge of maximum |∇z| —
the steepest descent encircling the peak.</p>
<p><b>Parameter.</b></p>
<ul>
<li><b>σ</b> (pre-smoothing scale) — smooths the field before differentiating, which
amplifies noise. <i>Increase</i> σ for a cleaner, broader gradient ridge that is more
robust to noise but localises the edge less precisely (the |∇z| peak shifts and
widens). <i>Decrease</i> σ to sharpen edge localisation at the cost of a noisier,
more fragmented gradient. Must be &gt; 0.</li>
</ul>
<p><b>Note.</b> The fill mask uses the <i>smoothed</i> field's NaN pattern, not the
original grid's, to avoid a spurious step at the data edge inside the Gaussian halo.</p>
""",
    "sobel": """
<h3>Sobel edge operator</h3>
<p>A classic discrete first-derivative estimator. It convolves the field with a 3×3
kernel that combines differentiation along one axis with light smoothing along the
other, then returns the gradient magnitude.</p>
<p><b>Kernels.</b><br>
&nbsp;&nbsp; <i>S</i><sub>u</sub> = [[−1,0,1],[−2,0,2],[−1,0,1]],&nbsp;
<i>S</i><sub>v</sub> = <i>S</i><sub>u</sub><sup>T</sup>.<br>
&nbsp;&nbsp; |∇z| ≈ √( (z∗S<sub>u</sub>)² + (z∗S<sub>v</sub>)² ).</p>
<p>The centre weight of 2 gives a mild smoothing that makes Sobel less
noise-sensitive than a bare finite difference, but its response is slightly
anisotropic (direction-dependent).</p>
<p><b>No parameters.</b> Fixed 3×3 support. The NaN region is extrapolated then
re-masked.</p>
""",
    "scharr": """
<h3>Scharr edge operator</h3>
<p>A refined 3×3 first-derivative operator with the same role as Sobel, but with
kernel weights optimised for <i>rotational symmetry</i> so the estimated gradient
direction stays accurate over a wider range of edge orientations.</p>
<p><b>Kernels.</b><br>
&nbsp;&nbsp; <i>K</i><sub>u</sub> = [[−3,0,3],[−10,0,10],[−3,0,3]],&nbsp;
<i>K</i><sub>v</sub> = <i>K</i><sub>u</sub><sup>T</sup>.</p>
<p>The 3:10:3 weighting minimises the angular error of the gradient relative to
Sobel's 1:2:1, making Scharr the better choice when edge <i>orientation</i> (not just
magnitude) matters. Returns √( (z∗K<sub>u</sub>)² + (z∗K<sub>v</sub>)² ).</p>
<p><b>No parameters.</b> The NaN region is extrapolated then re-masked.</p>
""",
    "prewitt": """
<h3>Prewitt edge operator</h3>
<p>The simplest 3×3 first-derivative operator: a finite difference combined with
uniform (box) averaging across the perpendicular axis.</p>
<p><b>Kernels.</b><br>
&nbsp;&nbsp; <i>P</i><sub>u</sub> = [[−1,0,1],[−1,0,1],[−1,0,1]],&nbsp;
<i>P</i><sub>v</sub> = <i>P</i><sub>u</sub><sup>T</sup>.</p>
<p>Because the perpendicular weights are equal (1:1:1) rather than emphasising the
centre, Prewitt smooths slightly less than Sobel and is marginally more
noise-sensitive. It is mainly useful as a baseline against Sobel/Scharr. Returns the
gradient magnitude.</p>
<p><b>No parameters.</b> The NaN region is extrapolated then re-masked.</p>
""",
    "directional": """
<h3>Directional derivative</h3>
<p>The rate of change of the field along a <i>chosen</i> direction θ. Unlike the
gradient magnitude it keeps the sign, so it distinguishes uphill (+) from downhill
(−) slopes along that axis.</p>
<p><b>Definition.</b> With the Gaussian-smoothed partials <i>z</i><sub>u</sub> = ∂z/∂u
and <i>z</i><sub>v</sub> = ∂z/∂v,<br>
&nbsp;&nbsp; <i>D</i><sub>θ</sub>z = ∇z · <b>n̂</b> =
z<sub>u</sub> cos θ + z<sub>v</sub> sin θ,&nbsp; <b>n̂</b> = (cos θ, sin θ).</p>
<p>θ = 0° measures slope along U, 90° along V. The partials are computed with
<code>gaussian_filter</code> at order (1,0) and (0,1) — convolution with the first
derivative of a Gaussian — so σ sets the differentiation scale.</p>
<p><b>Parameters.</b></p>
<ul>
<li><b>σ</b> (differentiation scale) — larger σ gives a smoother, less noisy derivative
with coarser spatial resolution; smaller σ resolves finer slope changes but is
noisier. Must be &gt; 0.</li>
<li><b>angle_deg</b> (θ, −180…180°) — rotates the measurement axis. 0° measures slope
along U, 90° along V; ±180° flips the sign (downhill ↔ uphill). Aligning θ with a
flank maximises the signed slope there, while θ perpendicular to it reads ≈ 0.</li>
</ul>
""",
    "laplacian": """
<h3>Laplacian ∇²z</h3>
<p>The sum of second partial derivatives — a measure of <i>curvature</i> / concavity.
It is positive in valleys (concave-up), negative on domes (concave-down), and crosses
zero exactly at inflection points where the surface changes its bending.</p>
<p><b>Definition.</b><br>
&nbsp;&nbsp; ∇²z = ∂²z/∂u² + ∂²z/∂v².</p>
<p>The <b>zero-crossings</b> of ∇²z trace inflection contours. For a single-peak
response these encircle the peak partway down its flank — the "shoulder" where the
dome turns into the surrounding skirt — which is one principled definition of the
receptive-field boundary.</p>
<p><b>Parameter.</b></p>
<ul>
<li><b>σ</b> (pre-smoothing scale) — the field is Gaussian-smoothed at scale σ before
the (very noise-sensitive) second derivative. <i>Increase</i> σ to stabilise the
Laplacian and produce smoother zero-crossing contours that sit further out on the
flank; too large blends neighbouring features and over-smooths the shoulder location.
<i>Decrease</i> σ to localise the inflection more tightly, at the cost of noisy,
fragmented zero-crossings. Must be &gt; 0.</li>
</ul>
""",
    "log": """
<h3>Laplacian of Gaussian (LoG)</h3>
<p>Smooths with a Gaussian and takes the Laplacian in a single convolution — the
canonical blob detector. It responds strongly to roughly circular bumps (or dips)
whose size matches the scale σ.</p>
<p><b>Definition.</b> LoG<sub>σ</sub> = (∇²G<sub>σ</sub>) ∗ z, with kernel<br>
&nbsp;&nbsp; ∇²G<sub>σ</sub>(u,v) = −(1/πσ⁴) · [ 1 − (u²+v²)/2σ² ] · exp( −(u²+v²)/2σ² ).</p>
<p>The kernel has a negative centre lobe and a positive surrounding ring (a "Mexican
hat"). Its response is extremal at the centre of a blob of radius ≈ σ√2, making LoG a
natural detector for the <i>foot</i> of the mountain — the ring where the dome meets
the baseline.</p>
<p><b>Parameter.</b></p>
<ul>
<li><b>σ</b> (scale) — sets the blob size the filter is matched to: response peaks for a
blob of radius ≈ σ√2. <i>Increase</i> σ to target larger feet/blobs and smooth more
(small structure is ignored); <i>decrease</i> σ to detect smaller structures, but the
output becomes noisier. Must be &gt; 0.</li>
</ul>
""",
    "hessian_eigval": """
<h3>Hessian largest eigenvalue λ<sub>max</sub></h3>
<p>The Hessian is the 2×2 matrix of second derivatives describing local surface
curvature. Its eigenvalues are the principal curvatures — the maximum and minimum
bending rates, along the directions in which they occur.</p>
<p><b>Definition.</b><br>
&nbsp;&nbsp; H = [[z<sub>uu</sub>, z<sub>uv</sub>], [z<sub>uv</sub>, z<sub>vv</sub>]],&nbsp;
eig(H) = {λ<sub>max</sub>, λ<sub>min</sub>},&nbsp; λ<sub>max</sub> ≥ λ<sub>min</sub>.</p>
<p>λ<sub>max</sub> (computed at scale σ with Gaussian derivatives) is large and
positive where the surface is strongly concave-up in <i>some</i> direction —
characteristic of the curved "foot" trough surrounding the peak. It is
rotation-invariant, unlike the axis-aligned Laplacian terms.</p>
<p><b>Parameter.</b></p>
<ul>
<li><b>σ</b> (scale of the Gaussian derivatives forming H) — <i>increase</i> σ to
measure curvature over a broader neighbourhood, capturing the broad foot trough and
suppressing noise but localising it less precisely; <i>decrease</i> σ for sharper but
noisier curvature. Must be &gt; 0.</li>
</ul>
""",
    "doh": """
<h3>Determinant of Hessian (DoH)</h3>
<p>The product of the two principal curvatures, det(H) = λ<sub>max</sub> · λ<sub>min</sub>.
A blob/peak detector that is large only where the surface curves strongly in
<i>both</i> directions at once (a true dome or bowl), and near zero on ridges or
saddles where one curvature vanishes or the signs oppose.</p>
<p><b>Definition.</b><br>
&nbsp;&nbsp; det(H) = z<sub>uu</sub> z<sub>vv</sub> − z<sub>uv</sub>².</p>
<p>Sign distinguishes shape: det &gt; 0 ⇒ dome or bowl (both curvatures same sign);
det &lt; 0 ⇒ saddle. This isolates the rounded summit from elongated ridge structure.</p>
<p><b>Parameter.</b></p>
<ul>
<li><b>σ</b> (scale at which the Hessian is estimated) — <i>increase</i> σ to favour
larger, smoother domes/bowls and damp noise; <i>decrease</i> σ to respond to finer
curvature features (and more noise). Must be &gt; 0.</li>
</ul>
""",
    "frangi": """
<h3>Frangi vesselness</h3>
<p>A multi-scale <i>ridge</i> (tube) detector originally designed for blood vessels.
It examines the two Hessian eigenvalues at several scales and responds where one
curvature is large and the other small — the signature of an elongated ridge or
trough rather than a blob.</p>
<p><b>Definition.</b> From λ<sub>1</sub>, λ<sub>2</sub> (|λ<sub>1</sub>| ≤ |λ<sub>2</sub>|)
it forms a blobness ratio R<sub>B</sub> = λ<sub>1</sub>/λ<sub>2</sub> and a
structureness S = √(λ<sub>1</sub>² + λ<sub>2</sub>²), then<br>
&nbsp;&nbsp; V = exp( −R<sub>B</sub>² / 2β² ) · ( 1 − exp( −S² / 2c² ) ),</p>
<p>maximised over a range of scales. High V marks tube-like ridges; blobs and flat
regions are suppressed.</p>
<p><b>Parameters.</b></p>
<ul>
<li><b>sigma_min</b> — smallest ridge half-width probed. <i>Lower</i> it to catch
thinner ridges, at the price of more noise sensitivity. Must be &gt; 0.</li>
<li><b>sigma_max</b> — largest ridge width probed. <i>Raise</i> it to respond to
broader ridges. The range [min, max] is sampled at 5 scales
(<code>np.linspace</code>) and the response is the maximum over them, so a wider span
covers more ridge widths but blurs scale specificity and costs more compute. Must be
≥ sigma_min.</li>
<li><b>black_ridges</b> — selects polarity. <i>True</i> detects dark/valley ridges
(the foot trough); <i>False</i> detects bright/crest ridges (the peak's spine). The
wrong polarity gives a near-zero response.</li>
</ul>
""",
    "sato": """
<h3>Sato tubeness</h3>
<p>A multi-scale tubular-structure filter, a predecessor of Frangi. It builds a
"tubeness" measure directly from the Hessian eigenvalues, emphasising elongated
ridge-like structures across the range of scales tested.</p>
<p><b>Idea.</b> Where a ridge's cross-section is locally cylindrical, one eigenvalue is
strongly negative (high curvature <i>across</i> the ridge) and the other near zero
(low curvature <i>along</i> it). Sato weights this eigenvalue pattern into a single
response, then takes the maximum across scales.</p>
<p><b>Parameters.</b></p>
<ul>
<li><b>sigma_min</b> — smallest ridge half-width sought; <i>lower</i> for thinner
ridges (more noise). Must be &gt; 0.</li>
<li><b>sigma_max</b> — largest ridge width sought; <i>raise</i> for broader ridges.
The [min, max] range is sampled at 5 scales and the response maxed over them — wider
span = more widths covered, less scale specificity. Must be ≥ sigma_min.</li>
<li><b>black_ridges</b> — <i>True</i> for dark/valley ridges (the foot trough),
<i>False</i> for bright/crest ridges; wrong polarity ⇒ near-zero response.</li>
</ul>
<p>Compared with Frangi, Sato has fewer free constants and is sometimes more stable on
faint ridges.</p>
""",
    "meijering": """
<h3>Meijering neuriteness</h3>
<p>A multi-scale ridge filter designed for detecting thin, faint neurite-like
structures. It uses a <i>modified</i> Hessian (a tuned linear combination of the
eigenvalues) to boost sensitivity to weak, elongated features that Frangi/Sato may
miss.</p>
<p><b>Idea.</b> The standard Hessian is re-mixed with a parameter α that recombines
λ<sub>1</sub> and λ<sub>2</sub>; the "neuriteness" is then a <i>normalised</i> function
of the modified eigenvalues, maximised across scales. The normalisation keeps the
response comparable between strong and weak ridges.</p>
<p><b>Parameters.</b></p>
<ul>
<li><b>sigma_min</b> — smallest ridge half-width probed; <i>lower</i> for thinner,
fainter ridges (more noise). Must be &gt; 0.</li>
<li><b>sigma_max</b> — largest ridge width probed; <i>raise</i> for broader ridges.
Sampled at 5 scales over [min, max], response maxed over them. Must be ≥ sigma_min.</li>
<li><b>black_ridges</b> — <i>True</i> for dark/valley ridges (the foot trough),
<i>False</i> for bright/crest ridges; wrong polarity ⇒ near-zero response.</li>
</ul>
<p>Useful here as a sensitive alternative when the foot trough is shallow.</p>
""",
    "gaussian_nth": """
<h3>Gaussian n-th derivative</h3>
<p>A general-purpose derivative operator: convolves with the (order_u, order_v)-th
partial derivative of a Gaussian. Differentiating the smooth kernel rather than the
noisy data is the numerically stable way to take high-order derivatives.</p>
<p><b>Definition.</b><br>
&nbsp;&nbsp; ∂<sup>(p+q)</sup>z / ∂u<sup>p</sup>∂v<sup>q</sup> ≈
z ∗ ( ∂<sup>(p+q)</sup>G<sub>σ</sub> / ∂u<sup>p</sup>∂v<sup>q</sup> ).</p>
<p>Special cases: (1,0)/(0,1) give first derivatives; (2,0)+(0,2) reconstruct the
Laplacian; (1,1) gives the mixed term z<sub>uv</sub>. This lets you probe arbitrary
directional/curvature components by hand.</p>
<p><b>Parameters.</b></p>
<ul>
<li><b>σ</b> (scale) — sets the smoothing built into the derivative. Higher orders
amplify noise sharply, so <i>increase</i> σ as you raise the orders to keep the result
usable; <i>decrease</i> σ for finer localisation when the orders are low. Must be &gt; 0.</li>
<li><b>order_u</b> / <b>order_v</b> (0…4 each) — the derivative order along U and V.
(0,0) = pure smoothing; (1,0)/(0,1) = slope; (2,0)/(0,2) = curvature along an axis;
(1,1) = the mixed saddle term z<sub>uv</sub>. Higher orders respond to finer-scale
structure and are progressively more noise-sensitive.</li>
</ul>
""",
    "dog": """
<h3>Difference of Gaussians (DoG)</h3>
<p>Subtracts a more-smoothed copy of the field from a less-smoothed one — a band-pass
filter that keeps structure at scales <i>between</i> the two σ's, and a fast,
well-known approximation to the Laplacian of Gaussian.</p>
<p><b>Definition.</b><br>
&nbsp;&nbsp; DoG = G<sub>σlow</sub> ∗ z − G<sub>σhigh</sub> ∗ z,&nbsp;
σ<sub>low</sub> &lt; σ<sub>high</sub>.</p>
<p>For a ratio σ<sub>high</sub>/σ<sub>low</sub> ≈ 1.6 the DoG closely matches a
scale-normalised LoG, responding to blobs/feet whose size lies in the pass-band.
Positive and negative lobes flank each transition, so zero-crossings again mark edges.</p>
<p><b>Parameters.</b></p>
<ul>
<li><b>sigma_low</b> — the sharper (less-smoothed) edge of the pass-band. <i>Lower</i>
it to keep finer detail in the result. Must be &gt; 0.</li>
<li><b>sigma_high</b> — the coarser (more-smoothed) edge. <i>Raise</i> it to strip out
broader background structure. Must be &gt; 0.</li>
</ul>
<p>The band between the two σ's is what survives: the ratio σ<sub>high</sub>/σ<sub>low</sub>
(≈ 1.6 to mimic a LoG) sets which feature size is kept — a narrow ratio is sharply
scale-selective, a wide one passes a broad range. Conventionally
σ<sub>low</sub> &lt; σ<sub>high</sub>; if reversed the response simply flips sign.</p>
""",
}

# Fail-fast: every operator must have a description popup (CLAUDE.md — no silent gaps).
_missing_descriptions = set(OPERATORS) - set(OPERATOR_DESCRIPTIONS)
if _missing_descriptions:
    raise RuntimeError(
        f"operators missing a description: {sorted(_missing_descriptions)}"
    )

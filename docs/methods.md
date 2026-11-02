# Methods: quantitative analysis of X-PEEM magnetic domains (XMCD and XLD)

This document describes every calculation performed by `xmcd_toolkit`,
in pipeline order, with the governing equations, the default parameters, and
the rationale. Symbols: an image is a grid of pixels indexed by position
`(x, y)`; `N` denotes a pixel count; angle brackets `⟨·⟩` denote a spatial
average over the flake (active area).

---

## 1. Image loading and contrast normalization

The raw image is converted to a single grayscale channel `g(x,y)`. For colour
screenshots, acquisition-software chrome (title bar, labels) is removed by
keeping only the largest connected region of near-grayscale pixels
(`|R−G|, |G−B| < 12`); 16-bit and float TIFFs are read directly.

Contrast is normalized **robustly**, between the 0.5 and 99.5 percentiles
rather than the raw extremes:

    g_lo = P_0.5(g),  g_hi = P_99.5(g)
    I(x,y) = clip( (g − g_lo) / (g_hi − g_lo), 0, 1 )

This prevents a few hot or dead pixels from compressing the true contrast into
a narrow band (on 16-bit TIFFs this recovered ~6× the usable dynamic range
versus min–max scaling). Percentile bounds are a parameter (`--clip`).

Optional local contrast enhancement (CLAHE, `--clahe`) applies adaptive
histogram equalization to pull weak domains out of a flat-gray background.

---

## 2. Definition of magnetization

The element-specific magnetic signal is the **XMCD asymmetry**, the normalized
difference of absorption taken with opposite X-ray helicities:

    A(x,y) = ( I⁺ − I⁻ ) / ( I⁺ + I⁻ )

`A` is proportional to the projection of the local magnetization onto the beam.
The two signs of the contrast are called **+ve** and **−ve**: at normal incidence
on a film with perpendicular anisotropy they are the two out-of-plane
magnetization directions; at grazing incidence, or for in-plane magnetization,
they are the two senses of the in-plane component along the beam. (Absolute moments would require
spectral sum-rule analysis; for domain statistics the normalized asymmetry is
the appropriate order parameter.)

**XLD (XMLD and XNLD).** The same pipeline applies to an **XLD asymmetry**, the
normalized difference of absorption with two orthogonal linear polarizations (or
at the two peaks of the L-edge multiplet):

    A(x,y) = ( I_LH − I_LV ) / ( I_LH + I_LV )

XLD is sensitive to the *axis* of the moments, not to their sign, so it images
antiferromagnetic as well as ferromagnetic domains. All quantities are computed
exactly as for XMCD.

**Notation.** The two states are called +ve (+1) and −ve (−1) for both kinds of
map, with 0 the neutral band; whether they are out-of-plane or in-plane
magnetization directions (XMCD) or two axis populations (XLD) is the user's
interpretation. The order parameter is written **⟨m⟩** (XMCD) in this document;
for XLD the program shows it as **⟨v_N⟩** (Néel vector) (*Dichroism*, step 1)
and every equation is the same.

After background removal (Section 3) the working signal `f(x,y)` is mapped to a
**normalized asymmetry** `m(x,y) ∈ [−1, 1]`, scaled so saturated domains sit
near ±1:

    s = P_99( |f| over flake ),   m(x,y) = clip( f / s, −1, 1 )

`m` is the per-pixel magnetization used throughout.

---

## 3. Flake (active-area) segmentation and background correction

The magnetic flake is isolated from the noisy background by **local-contrast
texture**: organized domains retain high local variance after denoising, while
shot-noise background collapses to low variance.

1. Total-variation denoise the normalized image (weight 0.12).
2. Local standard deviation in a 9×9 window → texture map.
3. Otsu threshold the texture map; morphological **closing** (disk radius 5),
   hole-filling, removal of components < 800 px, keep the largest → ROI mask
   `R(x,y)`.

The active area can instead be set manually (`--roi x0 y0 x1 y1`).

**Illumination flattening** removes slowly varying background so the +ve/−ve
threshold is unbiased:

    f = I − Gaussian(I, σ = 25 px)        (default)
    f = I − mean(I over background box)   (if --bg given: a defined zero level)

---

## 4. Three-state classification (the three probability cases)

Each pixel is assigned `m ∈ {+1, 0, −1}` (+ve / none / −ve). Background
(`R = 0`) is state 0. Inside the flake, the default value threshold is:

    m > +t   → +1 (+ve)
    m < −t   → −1 (−ve)
    |m| ≤ t  →  0 (gray: domain wall / ambiguous)

with `t` set by `--m-threshold` (e.g. 0.1 → gray band [−0.1, +0.1]). Two
alternatives exist: a z-score band (`|z| < z_neutral`, the legacy mode) and a
hard two-state split (`--states 2`, no gray). The threshold's influence is
quantified by the robustness sweep (Section 8).

---

## 5. Pixel calibration and area

The pixel size is **not** inferable from the image; it is supplied from the
scale bar:

    µm_per_px = (bar length in µm) / (bar length in px)

e.g. a 10 µm bar spanning 75 px → µm_per_px = 0.1333 (passed as `--um-per-px`).

**Area of one domain** = the pixel count of its connected component (4/8-
connectivity labelling). In physical units:

    Area_µm² = N_px × (µm_per_px)²

(length scales once, area twice). Equivalent diameter of a domain:

    d_eq = sqrt( 4 · Area / π )

---

## 6. Probability and number density

**Three-state PDF** over a region (flake or whole frame):

    p_s = N_s / N ,   s ∈ {+1, 0, −1},   p_{+1} + p_0 + p_{−1} = 1

**Local number density** — the fraction of each state inside a defined area
(tile) `C`:

    n_s(C) = N_s(C) / N_total(C)

computed on a grid of tiles (default 16×16 px); the denominator counts only
flake pixels per tile, and the three fractions sum to 1 in every tile. The
signed local field `n_{+1}(C) − n_{−1}(C)` maps local net magnetization.

---

## 7. Net magnetization order parameter and its change

The real-space order parameter — the imaging analog of the normalized
anomalous-Hall signal — is the spatial mean of the asymmetry over the flake,
equivalently the area asymmetry of +ve vs −ve:

    ⟨m⟩ = (1/N) Σ m(x,y)   ≈   (A₊ − A₋) / (A₊ + A₋) ,  ∈ [−1, 1]

The change between two conditions (e.g. before/after a pulse):

    Δ⟨m⟩ = ⟨m⟩_after − ⟨m⟩_before

---

## 8. Rigorous statistics on ⟨m⟩

A naive standard error `σ/√N_pixels` is **wrong**: pixels within a domain are
perfectly correlated, so the independent sample count is far below the pixel
count. The toolkit corrects for spatial correlation.

**Spatial autocorrelation.** The de-meaned asymmetry on the flake is
autocorrelated by FFT and divided, lag-by-lag, by the autocorrelation of the
flake mask (to correct for finite/irregular support):

    C(r) = [ FFT⁻¹{ |F{f·R}|² } ] / [ FFT⁻¹{ |F{R}|² } ] ,  normalized to C(0)=1

radially averaged to `C(r)`. The **correlation length** `ξ` is the lag where
`C(r) = 1/e`. The **integrated autocorrelation area** (effective area of one
independent domain) is

    τ_A = Σ_r 2π r · C(r)   (out to the first zero crossing)

**Effective sample size and corrected error:**

    N_eff = N_pixels / τ_A
    SE_corrected = σ / sqrt(N_eff)

For a domain pattern with ξ of a few pixels, `N_eff` is typically two orders of
magnitude below the pixel count and the corrected standard error an order of
magnitude above the naive one.

**Confidence intervals.** Analytic 95% CI uses the corrected error and the
t-distribution with `N_eff − 1` dof. Independently, a **block bootstrap**
resamples square tiles (side ≈ 2ξ) with replacement and recomputes ⟨m⟩
(default 1000–2000 iterations); the 95% CI is the 2.5/97.5 percentiles. The
bootstrap is the primary CI as it makes no parametric assumption and respects
correlation empirically.

**Significance vs zero** (is the flake net-magnetized?):

    z = ⟨m⟩ / SE_corrected ,   p = 2[1 − Φ(|z|)]

**Mixture decomposition.** A 3-Gaussian mixture (−ve / zero / +ve) is fit to the
per-pixel asymmetry histogram by nonlinear least squares; component means give
the saturated levels and weights give threshold-free area fractions.

**Threshold robustness.** ⟨m⟩ and the +ve/−ve fractions are recomputed across
`t ∈ [0.02, 0.30]`; a flat ⟨m⟩(t) demonstrates the result is not an artifact
of the threshold choice.

---

## 9. Domain-size distributions

Per-domain equivalent diameters (Section 5) are fit to the two distributions
standard in the magnetic-domain literature:

**Log-normal** (domain/grain sizes), 2-parameter fit (`floc = 0`):

    median = scale = exp(µ_ln),   shape = σ_ln,   goodness = KS statistic

**Power law** (Barkhausen/avalanche size, for difference images between
consecutive pulses), maximum-likelihood exponent (Clauset–Shalizi–Newman):

    τ = 1 + n / Σ ln( x_i / x_min ) ,   SE(τ) = (τ − 1) / sqrt(n)

A log-log survival plot accompanies the fit as a visual power-law check. Note
the power law is physically meaningful only on reversed-cluster sizes between
two images, not on a single static frame.

---

## 10. Domain shape

For each connected domain (skimage `regionprops`):

    eccentricity   0 (circle) → 1 (line)
    solidity       Area / ConvexArea   (1 = convex; < 1 = branched/concave)
    circularity    4π · Area / Perimeter²   (1 = circle; < 1 = irregular)

reported per state as mean values. Labyrinthine domains give eccentricity ≈ 0.8,
circularity ≈ 0.7.

---

## 11. Domain width and length (morphometry)

Because connected-component **area** is unreliable for an interconnected
labyrinth (the maze fuses into one component, so mean area ≫ median), the
robust size is the **width**, obtained by the distance-transform / medial-axis
("local thickness") construction:

1. Distance transform `DT(x)` = Euclidean distance from each domain pixel to
   the nearest wall.
2. Skeleton (medial axis) of the domain.
3. Local full width on the spine = `2 · DT` at skeleton pixels.

The **characteristic domain width** is the median of `2·DT` over the skeleton;
the **domain period** of the two-phase pattern is

    period ≈ width_up + width_down

**Skeleton length** (total domain "spine") counts adjacent skeleton-pixel steps
with diagonal steps weighted by √2:

    L_skel = N_ortho + √2 · N_diag   (× µm_per_px for physical length)

On the example: width ≈ 0.84 µm, period ≈ 1.69 µm. Width median (not mean) is
reported because skeleton spurs add a thin-width tail. The period should agree
to ~20% with the autocorrelation length (Section 8); disagreement flags a
segmentation problem.

---

## 12. Domain-wall density

The wall set is the boundary between +ve and −ve inside the flake (HIPR2
boundary operator `b = f − erosion(f)`, applied to the interface):

    W = dilation(+ve) ∩ −ve ∩ erosion(R)
    wall density = (Σ W) / (flake area)

a scalar that increases with fragmentation.

---

## 13. Pairwise before/after test (sda-compare)

For two images (before/after a pulse), each analyzed as above:

    Δ⟨m⟩ = ⟨m⟩_after − ⟨m⟩_before
    SE_Δ   = sqrt( SE_before² + SE_after² )            (independent acquisitions)
    z = Δ⟨m⟩ / SE_Δ ,   p = 2[1 − Φ(|z|)]

A **difference bootstrap** combines the two block-bootstrap distributions,
`Δ* = m*_after − m*_before`, and takes the 2.5/97.5 percentiles as the 95% CI.
The change is reported **significant** only if this CI excludes zero. Where the
analytic `p` and the bootstrap CI disagree near `p ≈ 0.05`, the result is
treated as marginal.

---

## 14. Limitations and reporting recommendations

- **Resolution floor.** Pixel size (e.g. 0.13 µm) is finer than instrument
  resolution (~0.5–0.7 µm for SPEEM); features below the resolution are not
  trusted as domains.
- **Labyrinth topology.** Report domain **width (median)** and **area
  fractions**, not mean domain area or domain count, for stripe/labyrinth
  states.
- **Per-image normalization.** `m` is scaled per image; for a fixed physical
  zero across a series, use `--bg`. The order parameter ⟨m⟩ measures the
  +ve/−ve *balance* and is invariant to overall brightness.
- **Statistical claims.** Net-magnetization changes are asserted only against
  the correlation-corrected confidence interval (`N_eff`, not pixel count);
  switching is claimed only when Δ⟨m⟩ clears that interval.

---

## 15. Comparison of two states (tabs 2 · Align and 3 · Compare)

**Alignment.** Each image is analysed on its own grid and then brought onto a common grid (the first image's;
other pixel counts are resampled) by a per-image transform (rotation, optional scale, flips, shift) followed by one
view rotation for the whole set; three-state maps are resampled nearest-neighbour, asymmetry maps bilinearly.
Transforms come from a rigid (or similarity) least-squares fit to clicked pairs of features that do not change
between states (flake edges, defects), optionally preceded by a phase-correlation estimate. As a check, the
Pearson coefficient r is computed for every integer shift of B up to ±20 px (and rotations ±2°): a maximum at zero
offset confirms the alignment; r ≈ 0 at all offsets means the patterns are unrelated rather than misaligned.

**Region.** All quantities use one region Ω: the active area of A ∩ the active area of B (after alignment),
optionally ∩ a drawn comparison region, restricted to pixels covered by both maps (N pixels).

**Flip map and switched fractions.** With the three-state maps s_A, s_B of the domain analysis (threshold m₀, same
despeckling), a pixel is *stayed +ve* / *stayed −ve* if s_A = s_B = ±1, *+ve→−ve* / *−ve→+ve* if it changed sign,
and *uncertain* if it is neutral in either map. Over the pixels classified in both,
f_{₊→₋} = N_{₊→₋}/N_{₊}^A, f_{₋→₊} = N_{₋→₊}/N_{₋}^A and switched = (N_{₊→₋} + N_{₋→₊})/N_classified. The two
fractions refer to different populations and do not add to one; M_flip = 1 − (f_{₊→₋} + f_{₋→₊}) is +1 for an
unchanged pattern, 0 for independent patterns and −1 for an inverted one.

**Memory r.** With a_i, b_i the asymmetry maps of A and B minus their means over Ω,
r = Σ a_i b_i / √(Σ a_i² Σ b_i²) (+1 same pattern, 0 unrelated, −1 inverted; insensitive to offset and contrast).
The 95 % interval comes from a block bootstrap: Ω is cut into ~1 µm blocks resampled with replacement (1000
resamples), which accounts for the spatial correlation of neighbouring pixels. The **local memory** r(x) is the
same coefficient in a Gaussian window (σ = 10 px) around each pixel, with the local means removed.

**Wall shift.** Wall pixels are +ve pixels with a −ve neighbour and vice versa; the wall shift is the median
distance from the walls of B to the nearest wall of A (and back), in nm when calibrated.

**Population changes.** From the counts of each state in Ω, f_k = N_k/N (k = +ve, neutral, −ve),
⟨m⟩ = f₊ − f₋, Δf_k = f_k^B − f_k^A (ΣΔf_k = 0) and Δ⟨m⟩ = Δf₊ − Δf₋. The smallest redistribution consistent
with the counts (the shrinking classes feed the growing ones) splits Δ⟨m⟩ into direct +ve↔−ve switching (each
pixel changes f₊ − f₋ by 2/N) and changes through the neutral class (1/N). Because counts fix only the marginals,
the fraction of Ω that changed class lies between ½Σ_k|Δf_k| and 1 − Σ_k max(0, f_k^A + f_k^B − 1); the actual
switching follows from the flip map. Without the neutral class, f̂_₊ = N_₊/(N_₊ + N_₋) and
m̂ = (N_₊ − N_₋)/(N_₊ + N_₋).

## Default parameters

| parameter | default | meaning |
|---|---|---|
| clip percentiles | (0.5, 99.5) | robust contrast normalization |
| tv_weight | 0.12 | denoising strength |
| contrast_window | 9 px | local-contrast ROI detector |
| flatten σ | 25 px | illumination background scale |
| min_roi_px | 800 | smallest ROI blob kept |
| min_domain_px | 12 | smallest domain counted |
| m-threshold `t` | 0.1 | gray band [−t, +t] on `m` |
| tile | 16 px | number-density "defined area" |
| n_boot | 1000 | block-bootstrap iterations |
| autocorr ξ | C(ξ)=1/e | correlation length |

## Key references (for the methods section)

- XMCD asymmetry / magnetic imaging: Stöhr et al., *Science* 259, 658 (1993).
- Quantitative moments via sum rules: Thole/Carra; Chen et al., *PRL* 75, 152 (1995).
- Morphological operations / distance transform / skeleton: Fisher, Perkins,
  Walker, Wolfart, *HIPR2* (Edinburgh); Gonzalez & Woods, *Digital Image
  Processing*.
- Image-analysis implementation: van der Walt et al., *scikit-image*, PeerJ 2 e453 (2014).
- Log-normal domain/grain sizes; power-law (Barkhausen) avalanche statistics:
  standard magnetic-domain literature.
- Companion transport (this material system): Kao et al., *Nat. Mater.* 21, 1029 (2022).

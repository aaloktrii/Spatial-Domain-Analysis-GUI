# Methods: quantitative analysis of XMCD-PEEM magnetic domains

This document describes every calculation performed by `xmcd_domain_toolkit`,
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

`A` is proportional to the projection of the local magnetization onto the beam;
in the normal-incidence PMA geometry used here, `A ∝ M_z`. Bright/dark contrast
therefore maps to up/down out-of-plane domains. (Absolute moments would require
spectral sum-rule analysis; for domain statistics the normalized asymmetry is
the appropriate order parameter.)

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

**Illumination flattening** removes slowly varying background so the up/down
threshold is unbiased:

    f = I − Gaussian(I, σ = 25 px)        (default)
    f = I − mean(I over background box)   (if --bg given: a defined zero level)

---

## 4. Three-state classification (the three probability cases)

Each pixel is assigned `m_z ∈ {+1, 0, −1}` (up / none / down). Background
(`R = 0`) is state 0. Inside the flake, the default value threshold is:

    m > +t   → +1 (up)
    m < −t   → −1 (down)
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
equivalently the area asymmetry of up vs down:

    ⟨m_z⟩ = (1/N) Σ m(x,y)   ≈   (A_up − A_down) / (A_up + A_down) ,  ∈ [−1, 1]

The change between two conditions (e.g. before/after a pulse):

    Δ⟨m_z⟩ = ⟨m_z⟩_after − ⟨m_z⟩_before

---

## 8. Rigorous statistics on ⟨m_z⟩

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

On the example image this gave `ξ = 6 px`, `N_eff ≈ 202` (vs 24,315 pixels),
and `SE_corrected ≈ 11 × SE_naive`.

**Confidence intervals.** Analytic 95% CI uses the corrected error and the
t-distribution with `N_eff − 1` dof. Independently, a **block bootstrap**
resamples square tiles (side ≈ 2ξ) with replacement and recomputes ⟨m_z⟩
(default 1000–2000 iterations); the 95% CI is the 2.5/97.5 percentiles. The
bootstrap is the primary CI as it makes no parametric assumption and respects
correlation empirically.

**Significance vs zero** (is the flake net-magnetized?):

    z = ⟨m_z⟩ / SE_corrected ,   p = 2[1 − Φ(|z|)]

**Mixture decomposition.** A 3-Gaussian mixture (down / zero / up) is fit to the
per-pixel asymmetry histogram by nonlinear least squares; component means give
the saturated levels and weights give threshold-free area fractions.

**Threshold robustness.** ⟨m_z⟩ and the up/down fractions are recomputed across
`t ∈ [0.02, 0.30]`; a flat ⟨m_z⟩(t) demonstrates the result is not an artifact
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

The wall set is the boundary between up and down inside the flake (HIPR2
boundary operator `b = f − erosion(f)`, applied to the interface):

    W = dilation(up) ∩ down ∩ erosion(R)
    wall density = (Σ W) / (flake area)

a scalar that increases with fragmentation.

---

## 13. Pairwise before/after test (compare.py)

For two images (before/after a pulse), each analyzed as above:

    Δ⟨m_z⟩ = ⟨m_z⟩_after − ⟨m_z⟩_before
    SE_Δ   = sqrt( SE_before² + SE_after² )            (independent acquisitions)
    z = Δ⟨m_z⟩ / SE_Δ ,   p = 2[1 − Φ(|z|)]

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
  zero across a series, use `--bg`. The order parameter ⟨m_z⟩ measures the
  up/down *balance* and is invariant to overall brightness.
- **Statistical claims.** Net-magnetization changes are asserted only against
  the correlation-corrected confidence interval (`N_eff`, not pixel count);
  switching is claimed only when Δ⟨m_z⟩ clears that interval.

---

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

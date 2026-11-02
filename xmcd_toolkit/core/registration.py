"""Alignment helpers: automatic rotation + shift, and the correlation-vs-shift check (no plotting, no GUI).

Domains change between states, so automatic alignment on the domain pattern is only a first guess;
point pairs on features that do not change (flake edges, defects) are the reference method."""
import numpy as np
from scipy import ndimage


def bandpass(m, lo=2.0, hi=30.0, border=0.12):
    """Band-passed, border-dropped (frame artefacts), Hann-windowed copy for registration."""
    m = np.nan_to_num(np.asarray(m, float))
    m = ndimage.gaussian_filter(m, lo) - ndimage.gaussian_filter(m, hi)
    by, bx = int(m.shape[0] * border), int(m.shape[1] * border)
    m = m[by:m.shape[0] - by, bx:m.shape[1] - bx]
    return m * np.outer(np.hanning(m.shape[0]), np.hanning(m.shape[1]))


def phase_corr(a, b):
    """Shift (dy, dx) moving b onto a, and the normalised peak height."""
    R = np.fft.fft2(a) * np.conj(np.fft.fft2(b))
    r = np.real(np.fft.ifft2(R / np.maximum(np.abs(R), 1e-12)))
    i = np.unravel_index(np.argmax(r), r.shape)
    sh = [i[k] if i[k] <= r.shape[k] // 2 else i[k] - r.shape[k] for k in (0, 1)]
    return np.array(sh, float), float(r[i])


def best_rotation(ref, moving_at, rot_range=3.0, coarse=0.5, fine=0.05, progress=None):
    """Extra rotation (deg) of the moving image that maximises the phase-correlation peak with `ref`.
    moving_at(angle) -> the moving image (final frame) with that extra rotation. Returns (angle, peak)."""
    R = bandpass(ref)
    grid = list(np.arange(-rot_range, rot_range + 1e-9, coarse)) if rot_range > 0 else [0.0]
    best = (-1.0, 0.0)
    n_total = 2 * len(grid)
    for i, ang in enumerate(grid):
        best = max(best, (phase_corr(R, bandpass(moving_at(ang)))[1], ang))
        if progress:
            progress((i + 1) / n_total)
    if rot_range > 0:
        fine_grid = list(np.arange(best[1] - coarse, best[1] + coarse + 1e-9, fine))
        for i, ang in enumerate(fine_grid):
            best = max(best, (phase_corr(R, bandpass(moving_at(ang)))[1], ang))
            if progress:
                progress(0.5 + (i + 1) / (2 * len(fine_grid)))
    return float(best[1]), float(best[0])


def best_shift(ref, moving):
    """Sub-pixel (0.1 px) shift (dy, dx) moving `moving` onto `ref` (band-passed phase correlation)."""
    from skimage.registration import phase_cross_correlation
    return phase_cross_correlation(bandpass(ref), bandpass(moving), upsample_factor=10)[0]


def highpass(d, sigma_px, near):
    """d minus its Gaussian low-pass over `near` (normalised convolution); NaN outside `near`."""
    fin = np.isfinite(d) & near
    if sigma_px <= 0:
        return np.where(fin, d, np.nan)
    lp = ndimage.gaussian_filter(np.where(fin, d, 0.0), sigma_px, truncate=3.0) / np.maximum(
        ndimage.gaussian_filter(fin.astype(float), sigma_px, truncate=3.0), 1e-6)
    return np.where(fin, d - lp, np.nan)


def shift_correlation(a, b_at, roi, max_shift=20, angles=(0.0,), sigma_px=0.0, progress=None):
    """Pearson r between A and B moved by every integer shift within +- max_shift px (and extra rotations of B),
    inside the ROI. b_at(angle) -> B (final frame) with that extra rotation. sigma_px > 0 removes a local
    background first (ROI + search margin only). r(0, 0) at angle 0 = the current alignment; a clear peak
    elsewhere = misaligned by that much; r ~ 0 everywhere = no common pattern at any offset.
    Returns dict(shifts, angles, r (n_angles, n, n), r0, best=(angle, dy, dx, r))."""
    m = int(max_shift)
    roi = roi & np.isfinite(a)
    near = ndimage.binary_dilation(roi, iterations=m + 1)
    ys, xs = np.nonzero(roi)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    za = highpass(np.asarray(a, float), sigma_px, near)[y0:y1, x0:x1]
    rc = roi[y0:y1, x0:x1] & np.isfinite(za)
    shifts = np.arange(-m, m + 1)
    out = np.full((len(angles), len(shifts), len(shifts)), np.nan)
    for ia, ang in enumerate(angles):
        bp = np.pad(highpass(np.asarray(b_at(ang), float), sigma_px, near), m, constant_values=np.nan)
        for iy, dy in enumerate(shifts):
            for ix, dx in enumerate(shifts):
                b = bp[y0 - dy + m:y1 - dy + m, x0 - dx + m:x1 - dx + m]     # B moved by (dy, dx)
                v = rc & np.isfinite(b)
                if v.sum() > 50:
                    aa, bb = za[v] - za[v].mean(), b[v] - b[v].mean()
                    out[ia, iy, ix] = float((aa * bb).sum() / np.sqrt((aa * aa).sum() * (bb * bb).sum()))
            if progress:
                progress((ia * len(shifts) + iy + 1) / (len(angles) * len(shifts)))
    i = np.unravel_index(np.nanargmax(out), out.shape)
    ia0 = int(np.argmin(np.abs(np.asarray(angles))))
    return {"shifts": shifts, "angles": list(angles), "r": out, "r0": float(out[ia0, m, m]),
            "best": (float(angles[i[0]]), int(shifts[i[1]]), int(shifts[i[2]]), float(out[i]))}

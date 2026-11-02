"""Sub-pixel domain-wall density. No plotting, no GUI."""
import numpy as np
from skimage import measure
from scipy.ndimage import gaussian_filter


def subpixel_wall_density(mz_field, roi, um_per_px=None):
    """
    Computes exact domain wall density (length / area) using sub-pixel 
    Marching Squares contour extraction at the mz=0.0 boundary.
    """
    # 1. Slightly smooth the field so contours can be interpolated smoothly 
    # between the +1 (up) and -1 (down) domains.
    m_smooth = gaussian_filter(mz_field.astype(float), sigma=1.0)
    
    # 2. Extract sub-pixel coordinates of all domain walls (level=0.0)
    contours = measure.find_contours(m_smooth, level=0.0)
    
    total_length_px = 0.0
    for contour in contours:
        # contour is an (N, 2) array of (row, col) coordinates
        # Calculate the Euclidean length of each sub-pixel segment
        diffs = np.diff(contour, axis=0)
        segment_lengths = np.linalg.norm(diffs, axis=1)
        
        # Only count wall segments that are actually inside our active flake ROI
        r_idx = np.clip(contour[:-1, 0].astype(int), 0, roi.shape[0]-1)
        c_idx = np.clip(contour[:-1, 1].astype(int), 0, roi.shape[1]-1)
        valid_mask = roi[r_idx, c_idx]
        
        total_length_px += np.sum(segment_lengths[valid_mask])
        
    area_px = np.sum(roi)
    if area_px == 0:
        return 0.0

    # 3. Scale to physical units if calibrated
    if um_per_px:
        total_length_um = total_length_px * um_per_px
        area_um2 = area_px * (um_per_px ** 2)
        return total_length_um / area_um2  # units: um / um^2
    else:
        return total_length_px / area_px   # units: px / px^2

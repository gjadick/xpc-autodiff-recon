import numpy as np
from scipy.ndimage import zoom, gaussian_filter


def phantom_crop_xy(vol, fov_X, fov_Y=None, dx=50e-6, dy=None, xc=None, yc=None):
    """
    Crop a 3D phantom volume in the x-y plane about a specified center.

    Parameters
    ----------
    vol : ndarray
        Input volume with shape [Nz, Ny, Nx].
    fov_X : float
        Desired x-direction field of view in meters.
    fov_Y : float, optional
        Desired y-direction field of view in meters. If None, uses `fov_X`.
    dx : float, optional
        Pixel size in the x direction in meters. Default is 50e-6.
    dy : float, optional
        Pixel size in the y direction in meters. If None, uses `dx`.
    xc : int, optional
        X-coordinate of the crop center in pixels. If None, uses the volume center.
    yc : int, optional
        Y-coordinate of the crop center in pixels. If None, uses the volume center.

    Returns
    -------
    ndarray
        Cropped volume with the same z dimension as the input and reduced x-y extent.
    """

    _, Ny, Nx = np.array(vol.shape) 
    
    if fov_Y is None:
        fov_Y = fov_X
    if dy is None:
        dy = dx
    if xc is None:
        xc = Nx // 2
    if yc is None:
        yc = Ny // 2
        
    Nx = int(fov_X / dx / 2) 
    Ny = int(fov_Y / dy / 2)
    
    return vol[:, yc-Ny:yc+Ny, xc-Nx:xc+Nx]


def phantom_upsample(vol, dx, matdict, Nz=1, dx0=50e-6, slice_thickness=None, DTYPE=np.float32, verbose=False):
    """
    Upsample a labeled phantom volume into non-overlapping soft per-material thickness maps.

    Each source z-slice is processed one at a time. For every material present in the
    slice, a local binary mask is cropped, upsampled by integer repetition or
    nearest-neighbor interpolation, and blurred with a material-specific Gaussian
    kernel. Instead of thresholding to a hard binary mask and assigning each pixel
    to a single winning material, the blurred response is converted into a soft
    fractional occupancy map. If multiple materials overlap at an upsampled pixel,
    their occupancies are normalized so that the summed occupancy does not exceed 1.
    The resulting fractional occupancies are accumulated into the output thickness maps,
    summed into Nz slices.

    Default Nz = 1 is used for projection-approximation XPCI simulations. It is the most
    memory-efficient, especially when target pixel size dx gets small with large arrays.
    However, Nz > 1 is also possible if multislice-based XPCI simulation is desired.

    Parameters
    ----------
    vol : ndarray
        Input labeled volume with shape [Nz0, Ny0, Nx0]. Pixel values correspond
        to material index in matdict, where 0 is assumed to denote air.
    dx : float
        Target in-plane pixel size in meters.
    matdict : dict
        Dictionary keyed by material index. Each value must provide `.k_blur`,
        `.thresh`, and `.name` attributes.
    Nz : int, optional
        Number of output z slices. Default is 1.
    dx0 : float, optional
        Original in-plane pixel size in meters. Default is 50e-6 (VICTRE examples).
    slice_thickness : float, optional
        Thickness assigned to each filled voxel in the output volume. If None,
        uses `dx0`.
    DTYPE : data-type, optional
        Output floating-point dtype. Default is `np.float32`.
    verbose : bool, optional
        If True, print basic per-slice progress information.

    Returns
    -------
    ndarray
        Output volume with shape [Nm, Nz, Ny, Nx], where Nm is the number of
        non-air materials present in `vol`. Each slice contains the thickness
        map for one material.
    """
    Nz0, Ny0, Nx0 = vol.shape
    Nx = int(round(Nx0 * dx0 / dx))
    Ny = int(round(Ny0 * dx0 / dx))

    if slice_thickness is None:
        slice_thickness = dx0

    mat_indices = np.unique(vol[vol > 0])
    Nm = len(mat_indices)
    mat_to_i = {m: i for i, m in enumerate(mat_indices)}

    target_vol = np.zeros([Nm, Nz, Ny, Nx], dtype=DTYPE)

    zx = Nx / Nx0
    zy = Ny / Ny0
    int_x = np.isclose(zx, round(zx))
    int_y = np.isclose(zy, round(zy))
    if int_x:
        zx_int = int(round(zx))
    if int_y:
        zy_int = int(round(zy))

    nonempty_ks = np.flatnonzero(np.any(vol > 0, axis=(1, 2)))

    for k in nonempty_ks:

        occ_sum = np.zeros([Ny, Nx], dtype=DTYPE)

        mats_in_k = np.unique(vol[k])
        mats_in_k = mats_in_k[mats_in_k > 0]

        if verbose:
            print(f'k = {k}, materials = {mats_in_k}')

        occ_maps = {}

        for m in mats_in_k:

            k_blur = matdict[m].k_blur
            thresh = matdict[m].thresh

            sigma = k_blur * dx0 / dx
            pad = int(np.ceil(k_blur))

            mask = vol[k] == m
            ys, xs = np.nonzero(mask)

            if len(ys) == 0:
                continue

            ymin, ymax = ys.min(), ys.max() + 1
            xmin, xmax = xs.min(), xs.max() + 1

            ymin = max(0, ymin - pad)
            ymax = min(mask.shape[0], ymax + pad)
            xmin = max(0, xmin - pad)
            xmax = min(mask.shape[1], xmax + pad)

            mask_crop = mask[ymin:ymax, xmin:xmax]

            if int_x and int_y:
                up_mask = np.repeat(np.repeat(mask_crop, zy_int, axis=0), zx_int, axis=1)
                y0 = ymin * zy_int
                x0 = xmin * zx_int
            else:
                up_mask = zoom(mask_crop, (zy, zx), order=0)
                y0 = int(round(ymin * zy))
                x0 = int(round(xmin * zx))

            up_mask = up_mask.astype(DTYPE)

            if sigma > 0:
                blur = gaussian_filter(up_mask, sigma=sigma)
            else:
                blur = up_mask

            occ = np.clip((blur - thresh) / max(1 - thresh, 1e-12), 0, 1).astype(DTYPE)

            if not np.any(occ > 0):
                continue

            y1 = y0 + occ.shape[0]
            x1 = x0 + occ.shape[1]

            occ_maps[m] = (y0, y1, x0, x1, occ)
            occ_sum[y0:y1, x0:x1] += occ

        k_target = min(int(k * Nz / Nz0), Nz - 1)

        for m, (y0, y1, x0, x1, occ) in occ_maps.items():
            i = mat_to_i[m]

            denom = occ_sum[y0:y1, x0:x1]
            frac = np.divide(
                occ,
                np.maximum(denom, 1),
                out=np.zeros_like(occ),
                where=occ > 0
            )

            target_vol[i, k_target, y0:y1, x0:x1] += frac * slice_thickness

    return target_vol



def centers_to_edges(coord, d=None):
    """
    Convert 1D sample centers to bin edges.

    Parameters
    ----------
    coord : ndarray, shape (N,)
        Sample centers.
    d : float, optional
        Bin width. Only needed if N == 1.

    Returns
    -------
    edges : ndarray, shape (N+1,)
        Bin edges.
    """
    coord = np.asarray(coord, dtype=np.float64)

    if coord.ndim != 1:
        raise ValueError('coord must be 1D.')

    if coord.size == 1:
        if d is None:
            raise ValueError('Need d when coord has length 1.')
        return np.array([coord[0] - d / 2, coord[0] + d / 2], dtype=np.float64)

    edges = np.empty(coord.size + 1, dtype=np.float64)
    edges[1:-1] = 0.5 * (coord[:-1] + coord[1:])
    edges[0] = coord[0] - 0.5 * (coord[1] - coord[0])
    edges[-1] = coord[-1] + 0.5 * (coord[-1] - coord[-2])
    return edges

def insert_sphere(xx, yy, z_coord, radius, xc=0.0, yc=0.0, zc=0.0, dz=None, DTYPE=np.float32):
    """
    Insert an exact multislice sphere into a 3D thickness volume.

    Each z-slice stores the exact path length of the sphere contained in that
    z-slab at each (y, x) location. This gives a direct thickness representation,
    for higher-res thickness projections relative to binary voxelization in z.

    Parameters
    ----------
    xx, yy : ndarray, shape (Ny, Nx)
        2D x- and y-coordinate grids at pixel centers.
    z_coord : ndarray, shape (Nz,)
        1D z coordinates of slice centers.
    radius : float
        Sphere radius in the same units as the coordinates.
    xc, yc, zc : float, optional
        Sphere center coordinates.
    dz : float, optional
        Slice thickness. Only needed when `z_coord` has length 1.
    DTYPE : dtype, optional
        Output volume dtype.

    Returns
    -------
    volume_out : ndarray, shape (Nz, Ny, Nx)
        Thickness volume. Summing over z gives the exact projected sphere
        thickness map, assuming the z coverage fully contains the sphere.
    """
    xx = np.asarray(xx, dtype=np.float64)
    yy = np.asarray(yy, dtype=np.float64)
    z_coord = np.asarray(z_coord, dtype=np.float64)

    if xx.shape != yy.shape:
        raise ValueError('xx and yy must have the same shape.')

    volume_out = np.zeros((z_coord.size, xx.shape[0], xx.shape[1]), dtype=DTYPE)

    z_edges = centers_to_edges(z_coord, d=dz)
    z_lo = z_edges[:-1][:, None, None]
    z_hi = z_edges[1:][:, None, None]

    r2 = (xx - xc)**2 + (yy - yc)**2
    half_chord = np.sqrt(np.clip(radius**2 - r2, 0.0, None))

    z_enter = zc - half_chord
    z_exit = zc + half_chord

    slab_thickness = np.clip(
        np.minimum(z_exit[None, :, :], z_hi) - np.maximum(z_enter[None, :, :], z_lo),
        0.0,
        None,
    )

    volume_out += slab_thickness.astype(volume_out.dtype, copy=False)
    return volume_out


def insert_ellipsoid(xx, yy, z_coord, rx, ry, rz, xc=0.0, yc=0.0, zc=0.0, dz=None,
                     DTYPE=np.float32):
    """
    Insert an exact multislice ellipsoid into a 3D thickness volume.

    Each z-slice stores the exact path length of the ellipsoid contained in that
    z-slab at each (y, x) location. This is a thickness representation, not a
    binary voxelized object.

    Ellipsoid equation:
        ((x - xc) / rx)^2 + ((y - yc) / ry)^2 + ((z - zc) / rz)^2 <= 1

    Parameters
    ----------
    xx, yy : ndarray, shape (Ny, Nx)
        2D x- and y-coordinate grids at pixel centers.
    z_coord : ndarray, shape (Nz,)
        1D z coordinates of slice centers.
    rx, ry, rz : float
        Ellipsoid semi-axes in x, y, z.
    xc, yc, zc : float, optional
        Ellipsoid center coordinates.
    dz : float, optional
        Slice thickness. Only needed when `z_coord` has length 1.
    DTYPE : dtype, optional
        Output volume dtype.

    Returns
    -------
    volume_out : ndarray, shape (Nz, Ny, Nx)
        Thickness volume. Summing over z gives the exact projected ellipsoid
        thickness map, assuming the z coverage fully contains the object.
    """
    xx = np.asarray(xx, dtype=np.float64)
    yy = np.asarray(yy, dtype=np.float64)
    z_coord = np.asarray(z_coord, dtype=np.float64)

    if xx.shape != yy.shape:
        raise ValueError('xx and yy must have the same shape.')

    if rx <= 0 or ry <= 0 or rz <= 0:
        raise ValueError('rx, ry, rz must all be positive.')

    volume_out = np.zeros((z_coord.size, xx.shape[0], xx.shape[1]), dtype=DTYPE)

    z_edges = centers_to_edges(z_coord, d=dz)
    z_lo = z_edges[:-1][:, None, None]
    z_hi = z_edges[1:][:, None, None]

    q = 1.0 - ((xx - xc) / rx)**2 - ((yy - yc) / ry)**2
    half_chord = rz * np.sqrt(np.clip(q, 0.0, None))

    z_enter = zc - half_chord
    z_exit = zc + half_chord

    slab_thickness = np.clip(
        np.minimum(z_exit[None, :, :], z_hi) - np.maximum(z_enter[None, :, :], z_lo),
        0.0,
        None,
    )

    volume_out += slab_thickness.astype(volume_out.dtype, copy=False)
    return volume_out


def insert_rod(xx, yy, z_coord, radius, axis='y', xc=0.0, yc=0.0, zc=0.0, length=None,
               dz=None, DTYPE=np.float32):
    """
    Insert an exact multislice cylindrical rod into a 3D thickness volume.

    The rod is aligned with either the x- or y-axis:

    - axis='y': circular cross section lies in the x-z plane
    - axis='x': circular cross section lies in the y-z plane

    Each z-slice stores the exact path length of the rod contained in that
    z-slab at each (y, x) location.

    Parameters
    ----------
    xx, yy : ndarray, shape (Ny, Nx)
        2D x- and y-coordinate grids at pixel centers.
    z_coord : ndarray, shape (Nz,)
        1D z coordinates of slice centers.
    radius : float
        Rod radius.
    axis : {'x', 'y'}, optional
        Rod axis direction.
        - 'y' means circular cross section in x-z
        - 'x' means circular cross section in y-z
    xc, yc, zc : float, optional
        Rod center coordinates.
    length : float or None, optional
        Finite rod length along its axis. If None, the rod spans the full field
        of view along that axis.
    dz : float, optional
        Slice thickness. Only needed when `z_coord` has length 1.
    DTYPE : dtype, optional
        Output volume dtype.

    Returns
    -------
    volume_out : ndarray, shape (Nz, Ny, Nx)
        Thickness volume. Summing over z gives the exact projected rod
        thickness map, assuming the z coverage fully contains the object.
    """
    xx = np.asarray(xx, dtype=np.float64)
    yy = np.asarray(yy, dtype=np.float64)
    z_coord = np.asarray(z_coord, dtype=np.float64)

    if xx.shape != yy.shape:
        raise ValueError('xx and yy must have the same shape.')

    if radius <= 0:
        raise ValueError('radius must be positive.')

    if axis not in ('x', 'y'):
        raise ValueError("axis must be either 'x' or 'y'.")

    volume_out = np.zeros((z_coord.size, xx.shape[0], xx.shape[1]), dtype=DTYPE)

    z_edges = centers_to_edges(z_coord, d=dz)
    z_lo = z_edges[:-1][:, None, None]
    z_hi = z_edges[1:][:, None, None]

    if axis == 'y':
        q = radius**2 - (xx - xc)**2
        half_chord = np.sqrt(np.clip(q, 0.0, None))

        if length is None:
            axial_mask = np.ones_like(yy, dtype=bool)
        else:
            axial_mask = np.abs(yy - yc) <= (length / 2)

    else:   # axis == 'x'
        q = radius**2 - (yy - yc)**2
        half_chord = np.sqrt(np.clip(q, 0.0, None))

        if length is None:
            axial_mask = np.ones_like(xx, dtype=bool)
        else:
            axial_mask = np.abs(xx - xc) <= (length / 2)

    z_enter = zc - half_chord
    z_exit = zc + half_chord

    slab_thickness = np.clip(
        np.minimum(z_exit[None, :, :], z_hi) - np.maximum(z_enter[None, :, :], z_lo),
        0.0,
        None,
    )

    slab_thickness *= axial_mask[None, :, :]

    volume_out += slab_thickness.astype(volume_out.dtype, copy=False)
    return volume_out



if __name__ == '__main__':
    pass

# def phantom_upsample_v2(vol, dx, matdict, Nz=1, dx0=50e-6, slice_thickness=None, DTYPE=np.float32, verbose=False):
#     """
    
#     Upsample a labeled phantom volume into non-overlapping per-material thickness maps.
    
#     Each source z-slice is processed one at a time. For every material present in the
#     slice, a local binary mask is cropped, upsampled by integer repetition or
#     nearest-neighbor interpolation, blurred with a material-specific Gaussian kernel,
#     and converted into a score map based on its threshold. Where multiple materials
#     compete for the same upsampled pixel, only the material with the largest
#     positive score is kept. The final winning labels are then accumulated into the
#     output per-material thickness maps, summed into Nz slices.

#     Default Nz = 1 is used for projection-approximation XPCI simulations. It is the most
#     memory-efficient, especially when target pixel size dx gets small with large arrays.
#     However, Nz > 1 is also possible if multislice-based XPCI simulation is desired.
    
#     Parameters
#     ----------
#     vol : ndarray
#         Input labeled volume with shape [Nz0, Ny0, Nx0]. Pixel values correspond
#         to material index in matdict, where 0 is assumed to denote air.
#     dx : float
#         Target in-plane pixel size in meters.
#     matdict : dict
#         Dictionary keyed by material index. Each value must provide `.k_blur`,
#         `.thresh`, and `.name` attributes.
#     Nz : int, optional
#         Number of output z slices. Default is 1.
#     dx0 : float, optional
#         Original in-plane pixel size in meters. Default is 50e-6 (VICTRE examples).
#     slice_thickness : float, optional
#         Thickness assigned to each filled voxel in the output volume. If None,
#         uses `dx0`.
#     DTYPE : data-type, optional
#         Output floating-point dtype. Default is `np.float32`.

#     Returns
#     -------
#     ndarray
#         Output volume with shape [Nm, Nz, Ny, Nx], where Nm is the number of
#         non-air materials present in `vol`. Each slice contains the thickness
#         map for one material.
#     """
#     Nz0, Ny0, Nx0 = vol.shape
#     Nx = int(round(Nx0 * dx0 / dx))
#     Ny = int(round(Ny0 * dx0 / dx))

#     if slice_thickness is None:
#         slice_thickness = dx0 
    
#     mat_indices = np.unique(vol[vol > 0])
#     Nm = len(mat_indices)
#     mat_to_i = {m: i for i, m in enumerate(mat_indices)}

#     target_vol = np.zeros([Nm, Nz, Ny, Nx], dtype=DTYPE)

#     zx = Nx / Nx0
#     zy = Ny / Ny0
#     int_x = np.isclose(zx, round(zx))
#     int_y = np.isclose(zy, round(zy))
#     if int_x:
#         zx_int = int(round(zx))
#     if int_y:
#         zy_int = int(round(zy))

#     nonempty_ks = np.flatnonzero(np.any(vol > 0, axis=(1, 2)))
    
#     for k in nonempty_ks:   # z index

#         label_img = np.zeros([Ny, Nx], dtype=np.uint16)
#         best_score = np.full([Ny, Nx], -np.inf, dtype=DTYPE)

#         mats_in_k = np.unique(vol[k])
#         mats_in_k = mats_in_k[mats_in_k > 0]

#         if verbose:
#             print(f'k = {k}, materials = {mats_in_k}')  # PRINT CHECK

#         for m in mats_in_k:

#             k_blur = matdict[m].k_blur
#             thresh = matdict[m].thresh

#             sigma = k_blur * dx0 / dx
#             pad = int(np.ceil(k_blur))

#             mask = vol[k] == m
#             ys, xs = np.nonzero(mask)

#             if len(ys) == 0:
#                 continue

#             ymin, ymax = ys.min(), ys.max() + 1
#             xmin, xmax = xs.min(), xs.max() + 1

#             ymin = max(0, ymin - pad)
#             ymax = min(mask.shape[0], ymax + pad)
#             xmin = max(0, xmin - pad)
#             xmax = min(mask.shape[1], xmax + pad)

#             mask_crop = mask[ymin:ymax, xmin:xmax]

#             if int_x and int_y:
#                 up_mask = np.repeat(np.repeat(mask_crop, zy_int, axis=0), zx_int, axis=1)
#                 y0 = ymin * zy_int
#                 x0 = xmin * zx_int
#             else:
#                 up_mask = zoom(mask_crop, (zy, zx), order=0)
#                 y0 = int(round(ymin * zy))
#                 x0 = int(round(xmin * zx))

#             if sigma > 0:
#                 blur = gaussian_filter(up_mask.astype(DTYPE), sigma=sigma)
#                 score = blur - thresh
#             else:
#                 blur = up_mask.astype(DTYPE)
#                 score = blur - thresh

#             y1 = y0 + score.shape[0]
#             x1 = x0 + score.shape[1]

#             score_view = best_score[y0:y1, x0:x1]
#             label_view = label_img[y0:y1, x0:x1]

#             update = score > score_view
#             update &= score > 0

#             score_view[update] = score[update]
#             label_view[update] = m

#         k_target = min(int(k * Nz / Nz0), Nz - 1)

#         for m in mats_in_k:
#             i = mat_to_i[m]
#             target_vol[i, k_target] += (label_img == m).astype(DTYPE) * slice_thickness
            
#     return target_vol


# def phantom_upsample_v1(vol, dx, matdict, Nz=1, dx0=50e-6, slice_thickness=None, DTYPE=np.float32, verbose=False):
#     """
#     Upsample a labeled phantom volume into per-material thickness maps.

#     Each non-air material is isolated, optionally upsampled by integer repetition
#     or nearest-neighbor interpolation, blurred with a Gaussian kernel according to
#     material-specific settings, thresholded, and accumulated into a target volume.

#     Parameters
#     ----------
#     vol : ndarray
#         Input labeled volume with shape [Nz0, Ny0, Nx0]. Pixel values correspond
#         to material index in matdict, where 0 is assumed to denote air.
#     dx : float
#         Target in-plane pixel size in meters.
#     matdict : dict
#         Dictionary keyed by material index. Each value must provide `.k_blur`,
#         `.thresh`, and `.name` attributes.
#     Nz : int, optional
#         Number of output z slices. Default is 1.
#     dx0 : float, optional
#         Original in-plane pixel size in meters. Default is 50e-6 (VICTRE examples).
#     slice_thickness : float, optional
#         Thickness assigned to each filled voxel in the output volume. If None,
#         uses `dx0`.
#     DTYPE : data-type, optional
#         Output floating-point dtype. Default is `np.float32`.

#     Returns
#     -------
#     ndarray
#         Output volume with shape [Nm, Nz, Ny, Nx], where Nm is the number of
#         non-air materials present in `vol`. Each slice contains the thickness
#         map for one material.
#     """
    
#     Nz0, Ny0, Nx0 = vol.shape
#     Nx = int(round(Nx0 * dx0 / dx))
#     Ny = int(round(Ny0 * dx0 / dx))

#     if slice_thickness is None:
#         slice_thickness = dx0 
    
#     mat_indices = np.unique(vol[vol > 0])  # ignore air (m=0)
#     Nm = len(mat_indices)

#     target_vol = np.zeros([Nm, Nz, Ny, Nx], dtype=DTYPE)

#     zx = Nx / Nx0
#     zy = Ny / Ny0
#     int_x = np.isclose(zx, round(zx))
#     int_y = np.isclose(zy, round(zy))
#     if int_x:
#         zx_int = int(round(zx))
#     if int_y:
#         zy_int = int(round(zy))
    
#     for i, m in enumerate(mat_indices):
#         k_blur = matdict[m].k_blur
#         thresh = matdict[m].thresh
        
#         sigma = k_blur * dx0 / dx
#         pad = int(np.ceil(k_blur))  #int(np.ceil(sigma))

#         mvol = (vol == m)
#         nonempty_ks = np.flatnonzero(np.any(mvol, axis=(1, 2)))

#         if verbose:
#             print(f'{m}: {matdict[m].name}, k={k_blur:.2f}, t={thresh:.2f}')  # PRINT CHECK
        
#         for k in nonempty_ks:   # z index

#             mask = mvol[k]
        
#             ys, xs = np.nonzero(mask)
#             ymin, ymax = ys.min(), ys.max() + 1
#             xmin, xmax = xs.min(), xs.max() + 1

#             ymin = max(0, ymin - pad)
#             ymax = min(mask.shape[0], ymax + pad)
#             xmin = max(0, xmin - pad)
#             xmax = min(mask.shape[1], xmax + pad)
        
#             mask_crop = mask[ymin:ymax, xmin:xmax]
        
#             if int_x and int_y:
#                 up_mask = np.repeat(np.repeat(mask_crop, zy_int, axis=0), zx_int, axis=1)
#                 y0 = ymin * zy_int
#                 x0 = xmin * zx_int
#             else:
#                 up_mask = zoom(mask_crop, (zy, zx), order=0)
#                 y0 = int(round(ymin * zy))
#                 x0 = int(round(xmin * zx))
        
#             if sigma > 0:
#                 blur = gaussian_filter(up_mask.astype(DTYPE), sigma=sigma)
#                 img = blur > thresh
#             else:
#                 img = up_mask
        
#             k_target = min(int(k * Nz / Nz0), Nz - 1)
        
#             y1 = y0 + img.shape[0]
#             x1 = x0 + img.shape[1]
        
#             target_vol[i, k_target, y0:y1, x0:x1] += img * slice_thickness
            
#     return target_vol
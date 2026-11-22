import atexit
import multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.patches import Patch
from surfplot import Plot
from neuromaps.datasets import fetch_fslr

# fsLR 32k surfaces and sulcal maps, loaded once at import time
_SURFS = fetch_fslr(density='32k')
_SULC = dict(zip(('left', 'right'), _SURFS['sulc']))


def _get_surf(name):
    """Return {'left': path, 'right': path} for the named fsLR surface."""
    lh, rh = _SURFS[name]
    return {'left': lh, 'right': rh}


def _label_colors(K, colors):
    """Return K RGBA colors from None, a colormap name, or an explicit list."""
    if colors is None:
        colors = 'tab10' if K <= 10 else 'tab20'
    if isinstance(colors, str):
        base = plt.get_cmap(colors)
        if base.N < 256:                                  # qualitative: cycle
            return [base(i % base.N) for i in range(K)]
        return list(base(np.linspace(0, 1, K)))           # continuous: sample
    if len(colors) < K:
        raise ValueError(f"Need at least {K} colors, got {len(colors)}.")
    return list(colors[:K])


# Render pool: VTK does not release its memory reliably, so panels are rendered
# in a worker process that is replaced after _POOL_TASKS panels.
_POOL = None
_POOL_TASKS = 20       # panels rendered per worker before it is replaced
_POOL_PROCESSES = 1    # >1 renders panels in parallel (memory scales with it)


def _init_worker():
    """Use a non-interactive backend in render workers."""
    plt.switch_backend('Agg')


def _get_pool():
    """Lazily create the render pool."""
    global _POOL
    if _POOL is None:
        _POOL = mp.get_context('fork').Pool(processes=_POOL_PROCESSES,
                                            maxtasksperchild=_POOL_TASKS,
                                            initializer=_init_worker)
    return _POOL


def close_render_pool():
    """Shut down the render workers (also called automatically at exit).
    Call it after editing this module so workers pick up the new code."""
    global _POOL
    if _POOL is not None:
        _POOL.terminate()
        _POOL.join()
        _POOL = None


atexit.register(close_render_pool)


def _render_panel(job):
    """Render one (hemisphere, view) panel with surfplot and return it as an image array."""
    h = job['hemi']
    p = Plot(job['surf_path'] if h == 'left' else None,
             job['surf_path'] if h == 'right' else None,
             views=job['view'], size=(job['panel_px'], job['panel_px']),
             zoom=job['zoom'])
    p.add_layer(job['sulc'], cmap='binary_r', cbar=False, alpha=0.5)
    p.add_layer(job['data'], cmap=job['cmap'],
                color_range=job['color_range'], cbar=False)
    if job['seed_outline']:
        p.add_layer((job['data'] != 0).astype(float), cmap='binary',
                    color_range=(0, 1), as_outline=True, cbar=False)

    overlay = job['overlay']
    if overlay is not None and overlay.any():
        p.add_layer(overlay, cmap=ListedColormap([job['overlay_color']]),
                    color_range=(0, 1), cbar=False,
                    as_outline=(job['overlay_style'] == 'outline'))

    tmp = p.build(scale=job['scale'])                 # temporary matplotlib figure
    img = np.asarray(tmp.axes[0].images[0].get_array())   # grab the rendered image
    plt.close(tmp)
    return img


def plot_surface_map(data_left=None, data_right=None,
                     views=('lateral', 'medial'),
                     axes=None, surf='veryinflated',
                     seed_outline=False,
                     categorical=False, colors=None, label_names=None,
                     overlay_left=None, overlay_right=None,
                     overlay_color='black', overlay_style='outline',
                     vmax=None, percent_vertices=None, fix_threshold=None,
                     cmap='RdYlBu_r', colorbar=True, cbar_label=None,
                     zoom=1.2, panel_px=400, scale=(2, 2), isolate=True):
    """
    Plot fsLR 32k surface maps, one panel per (hemisphere, view).

    Each panel is rendered offscreen with surfplot and drawn onto a matplotlib
    axes, so the result can be placed in any figure layout.

    Parameters
    ----------
    data_left, data_right : array-like of shape (32492,) or None
        Vertex data for each hemisphere. Use None to skip a hemisphere.
        At least one must be provided. NaNs are treated as 0 (transparent).
    views : str or tuple of str, default ('lateral', 'medial')
        Views to draw for each hemisphere. Any view supported by surfplot,
        e.g. 'lateral', 'medial', 'dorsal', 'ventral', 'anterior', 'posterior'.
    axes : Axes, sequence of Axes, or None
        Axes to draw on. If None, a new 1 x N figure is created. Otherwise
        exactly N axes are required, with N = n_hemispheres * n_views, ordered
        left (all views), then right (all views), skipping any hemisphere
        that is not plotted.
    surf : str, default 'veryinflated'
        fsLR surface to render: 'inflated', 'veryinflated', 'midthickness', ...
    seed_outline : bool, default False
        Draw a black outline around every non-zero region. Helps small seeds
        stand out against the background.
    categorical : bool, default False
        If True, the data are integer labels (0 = background, non-integers are
        rounded) drawn with one discrete color per label. Colors are shared
        across hemispheres. `vmax`, `percent_vertices`, `fix_threshold` and
        `cmap` are ignored in this mode.
    colors : None, str or list, optional
        Categorical mode only. None uses 'tab10' ('tab20' if more than 10
        labels). A colormap name or an explicit list of colors is also
        accepted. Colors are assigned to the sorted label IDs present in the
        data, so pass an explicit list with at least as many entries as there
        are labels if you need identical colors across separate figures.
    label_names : dict, optional
        Categorical mode only. Mapping {label_id: 'name'} used in the legend.
    overlay_left, overlay_right : array-like of shape (32492,) or None
        Optional seed/ROI masks drawn on top of the data. Any non-zero vertex
        is part of the overlay. Independent of the data threshold, so seeds
        stay visible even where the map is thresholded out. Only used for
        hemispheres that are plotted.
    overlay_color : color, default 'black'
        Color of the overlay.
    overlay_style : {'outline', 'fill'}, default 'outline'
        'outline' draws only the border, so the underlying correlation values
        remain visible; 'fill' paints the whole region.
    vmax : float, optional
        Continuous mode only. Color range is (-vmax, vmax). Defaults to the
        maximum absolute value across both hemispheres.
    percent_vertices : float in (0, 1), optional
        Continuous mode only. Keep this fraction of vertices (those with the
        largest |value|, pooled across hemispheres). Ignored if
        `fix_threshold` is given. Computed over all vertices, so it is not
        suitable for sparse maps such as seeds.
    fix_threshold : float, optional
        Continuous mode only. Values with |value| below this are hidden.
    cmap : str, default 'RdYlBu_r'
        Continuous mode only. Colormap for the data.
    colorbar : bool, default True
        Draw a colorbar (continuous mode) or a legend (categorical mode).
    cbar_label : str, optional
        Colorbar label, or legend title in categorical mode.
    zoom : float, default 1.2
        Zoom applied to each rendered panel.
    panel_px : int, default 400
        Side length, in pixels, of each rendered panel before `scale`.
    scale : tuple of float, default (2, 2)
        Render resolution multiplier. Use (1, 1) for fast previews and higher
        values for figure-quality output.
    isolate : bool, default True
        Render panels in a recycled worker process so VTK memory is released.
        Set False to render in the current process (faster for one-off plots,
        but memory accumulates over repeated calls). Automatically disabled
        when already running inside a multiprocessing worker.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : numpy.ndarray of Axes, one per panel

    Notes
    -----
    The categorical legend is placed outside the axes, so save with
    ``bbox_inches='tight'`` or set ``colorbar=False`` and build your own.

    Examples
    --------
    Continuous map, both hemispheres, lateral and medial views:

    >>> fig, axes = plot_surface_map(dl, dr, fix_threshold=0.2)

    Labelled seeds on the left hemisphere, four views:

    >>> fig, axes = plot_surface_map(
    ...     seeds_left, None,
    ...     views=('lateral', 'medial', 'dorsal', 'ventral'),
    ...     categorical=True, seed_outline=True,
    ...     label_names={1: 'PCC', 2: 'mPFC'})
    """
    if data_left is None and data_right is None:
        raise ValueError("Provide data for at least one hemisphere.")
    if isinstance(views, str):
        views = (views,)

    SURF = _get_surf(surf)

    # Clean the input data
    hemis = []
    for h, d in (('left', data_left), ('right', data_right)):
        if d is None:
            continue
        d = np.nan_to_num(np.asarray(d, dtype=float))
        if categorical:
            d = np.rint(d)
        hemis.append((h, d))
    all_data = np.concatenate([d for _, d in hemis])

    overlays = {}
    for h, o in (('left', overlay_left), ('right', overlay_right)):
        if o is not None:
            overlays[h] = (np.nan_to_num(np.asarray(o, dtype=float)) != 0).astype(float)
    if overlay_style not in ('outline', 'fill'):
        raise ValueError("overlay_style must be 'outline' or 'fill'.")
    
    if categorical:
        uniq = np.unique(all_data[all_data != 0])
        if len(uniq) == 0:
            raise ValueError("No non-zero labels found.")
        K = len(uniq)
        label_colors = _label_colors(K, colors)
        layer_cmap = ListedColormap(label_colors)
        color_range = (0.5, K + 0.5)    # label i (1..K) lands at the centre of bin i
        # remap IDs to 1..K, shared across hemispheres, so colors are consistent
        hemis = [(h, np.where(d != 0, np.searchsorted(uniq, d) + 1, 0).astype(float))
                 for h, d in hemis]
    else:
        if vmax is None:
            vmax = np.abs(all_data).max()
        if fix_threshold is not None:
            threshold = fix_threshold
        elif percent_vertices:
            threshold = np.quantile(np.abs(all_data), 1 - percent_vertices)
        else:
            threshold = None
        if threshold is not None:       # zeros are transparent in surfplot
            hemis = [(h, np.where(np.abs(d) < threshold, 0, d)) for h, d in hemis]
        layer_cmap = cmap
        color_range = (-vmax, vmax)

    panels = [(h, d, v) for h, d in hemis for v in views]
    n = len(panels)

    # Axes
    if axes is None:
        fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    else:
        fig = np.atleast_1d(axes).ravel()[0].figure
    axes = np.atleast_1d(axes).ravel()
    if len(axes) != n:
        raise ValueError(f"Got {len(axes)} axes but need {n} "
                         f"({len(hemis)} hemisphere(s) x {len(views)} view(s)).")

    # Render each panel, in a recycled worker process unless disabled
    jobs = [dict(hemi=h, surf_path=SURF[h], sulc=_SULC[h], data=d, view=v,
                 cmap=layer_cmap, color_range=color_range,
                 seed_outline=seed_outline, overlay=overlays.get(h),
                 overlay_color=overlay_color, overlay_style=overlay_style,
                 zoom=zoom, panel_px=panel_px, scale=scale)
            for h, d, v in panels]

    if isolate and not mp.current_process().daemon:
        images = _get_pool().map(_render_panel, jobs)
    else:
        images = [_render_panel(job) for job in jobs]

    # Show each rendered panel on its axes
    for ax, img in zip(axes, images):
        ax.imshow(img)
        ax.set_axis_off()

    # Colorbar or legend
    if colorbar:
        if categorical:
            names = label_names or {}
            handles = [Patch(facecolor=c, edgecolor='k',
                             label=str(names.get(int(l), int(l))))
                       for l, c in zip(uniq, label_colors)]
            fig.legend(handles=handles, loc='center left',
                       bbox_to_anchor=(1.0, 0.5), frameon=False, title=cbar_label)
        else:
            sm = cm.ScalarMappable(norm=Normalize(-vmax, vmax), cmap=cmap)
            fig.colorbar(sm, ax=list(axes), shrink=0.6, pad=0.01, label=cbar_label)

    return fig, axes
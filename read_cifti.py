"""Minimal CIFTI reader (CIFTI-1 and CIFTI-2) that doesn't rely on nibabel's CIFTI-2-only parser.

The data order is chosen from the CIFTI version: CIFTI-1 files (e.g. Midnight Scan Club)
store each frame contiguously ("C"); CIFTI-2 files follow the NIfTI column-major order ("F").
"""
import numpy as np
import xml.etree.ElementTree as ET

_NIFTI_DTYPES = {2: np.uint8, 4: np.int16, 8: np.int32, 16: np.float32,
                 64: np.float64, 256: np.int8, 512: np.uint16, 768: np.uint32}


def _read_cifti_raw(path, order=None):
    """Read a CIFTI (v1 or v2) file without nibabel.

    Parameters
    ----------
    path : str
        Path to a .dtseries.nii, .dscalar.nii or .dlabel.nii file.
    order : {None, 'C', 'F'}
        Memory order of the data matrix. None picks it from the CIFTI version
        ('C' for CIFTI-1, 'F' for CIFTI-2).

    Returns
    -------
    data : ndarray, 2D float32
    root : xml.etree.ElementTree.Element
        Root of the CIFTI XML header.
    """
    with open(path, "rb") as f:
        raw = f.read()
    endian = "<" if np.frombuffer(raw[:4], "<i4")[0] == 540 else ">"   # NIfTI-2 header size
    hdr = lambda off, dt, n=1: np.frombuffer(raw, endian + dt, n, off)

    dtype = np.dtype(_NIFTI_DTYPES[int(hdr(12, "i2")[0])]).newbyteorder(endian)
    dim = hdr(16, "i8", 8)
    shape = tuple(int(d) for d in dim[1:1 + dim[0]])
    vox_offset = int(hdr(168, "i8")[0])
    slope, inter = hdr(176, "f8", 2)

    # Header extensions start at byte 544; CIFTI XML has code 32
    xml, pos = None, 544
    while pos + 8 <= vox_offset:
        esize, ecode = hdr(pos, "i4", 2)
        if esize <= 0:
            break
        if ecode == 32:
            xml = raw[pos + 8:pos + esize].rstrip(b"\x00")
        pos += int(esize)
    if xml is None:
        raise ValueError("No CIFTI XML extension found.")
    root = ET.fromstring(xml)

    if order is None:
        order = "C" if root.get("Version", "2").startswith("1") else "F"

    data = np.frombuffer(raw, dtype, int(np.prod(shape)), vox_offset).reshape(shape, order=order)
    data = data.astype(np.float32)
    if slope not in (0, 1) or inter != 0:
        data = data * (slope if slope != 0 else 1) + inter
    data = np.atleast_2d(data.squeeze())
    return data, root


def cifti_to_hemis(path, order=None):
    """Return {'left': (n_maps, n_vertices), 'right': (n_maps, n_vertices)}.

    Cortical data are placed on the full surface mesh (32492 vertices per
    hemisphere for fsLR 32k), with zeros on the medial wall. Subcortical
    data are ignored.
    """
    data, root = _read_cifti_raw(path, order=order)

    n_grayord = max(int(bm.get("IndexOffset")) + int(bm.get("IndexCount"))
                    for bm in root.iter("BrainModel"))
    if data.shape[1] != n_grayord:                     # put grayordinates on axis 1
        data = data.T

    hemi_of = {"CIFTI_STRUCTURE_CORTEX_LEFT": "left", "CIFTI_STRUCTURE_CORTEX_RIGHT": "right"}
    out = {}
    for bm in root.iter("BrainModel"):
        hemi = hemi_of.get(bm.get("BrainStructure"))
        if bm.get("ModelType") != "CIFTI_MODEL_TYPE_SURFACE" or hemi is None:
            continue
        n_vert = int(bm.get("SurfaceNumberOfNodes") or bm.get("SurfaceNumberOfVertices"))
        idx_el = bm.find("NodeIndices")
        if idx_el is None:
            idx_el = bm.find("VertexIndices")
        vertices = np.array(idx_el.text.split(), dtype=int)
        start, count = int(bm.get("IndexOffset")), int(bm.get("IndexCount"))
        full = np.zeros((data.shape[0], n_vert), np.float32)
        full[:, vertices] = data[:, start:start + count]
        out[hemi] = full
    return out


def cifti_label_names(path):
    """Return {label_id: name} from a .dlabel.nii file (label 0 excluded)."""
    _, root = _read_cifti_raw(path)
    return {int(l.get("Key")): l.text.strip()
            for l in root.iter("Label") if int(l.get("Key")) > 0}

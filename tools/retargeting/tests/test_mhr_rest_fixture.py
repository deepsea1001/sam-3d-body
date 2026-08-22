import json
from pathlib import Path

import numpy as np
import pytest

FIX = Path(__file__).resolve().parent.parent / "bind_poses" / "mhr_skeleton_rest.json"
NPZ_ROWS = Path(__file__).resolve().parent / "fixtures" / "mhr_npz_rows.json"

BODY_PAIRS = [(1, 34), (34, 35), (35, 36), (36, 37), (37, 110), (110, 113),
              (75, 76), (39, 40), (2, 3), (18, 19), (76, 77), (40, 41)]


def _load():
    d = json.loads(FIX.read_text())
    q = np.asarray(d["rest_global_q_wxyz"], float)
    off = np.asarray(d["template_offsets_cm"], float)
    par = list(d["parents"])
    return d, q, off, par


def _qmat_wxyz(q):
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                     [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                     [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])


def _rest_fk(q, off, par):
    # rest_global_q are GLOBAL quats; positions accumulate with the PARENT's
    # global rotation applied to the child's template offset.
    R = np.stack([_qmat_wxyz(qi) for qi in q])
    P = np.zeros((127, 3))
    for j in range(127):
        P[j] = off[j] if par[j] < 0 else P[par[j]] + R[par[j]] @ off[j]
    return R, P


def test_fixture_exists_and_is_orthonormal():
    d, q, off, par = _load()
    assert len(d["names"]) == 127 and d["names"][35] == "c_spine1" and d["names"][37] == "c_spine3"
    assert d["names"][78] == "l_wrist" and d["names"][42] == "r_wrist"
    norms = np.linalg.norm(q, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-6)


def test_rest_fk_matches_pinned_geometry():
    _, q, off, par = _load()
    R, P = _rest_fk(q, off, par)
    assert np.allclose(P[78], [53.986, 111.331, 13.185], atol=0.01)   # l_wrist
    assert np.allclose(P[42], [-53.986, 111.331, 13.185], atol=0.01)  # r_wrist
    assert abs(P[110][1] - 144.191) < 0.01                            # c_neck Y
    z_curve = [P[i][2] for i in (34, 35, 36, 37)]
    assert np.allclose(z_curve, [-3.254, -2.150, -4.745, -3.713], atol=0.01)


def test_bone_direction_control_on_real_row_with_positive_control():
    _, q, off, par = _load()
    R, _ = _rest_fk(q, off, par)
    rows = json.loads(NPZ_ROWS.read_text())["rows"]
    row = rows["a6566802a6c9ddd63340ccb4520e0001"]
    Rp = np.asarray(row["joint_global_rots"], float)
    P = np.asarray(row["pred_joint_coords"], float)
    F = np.diag([1.0, -1.0, -1.0])

    def min_cos(Rp_):
        cs = []
        for pi, ci in BODY_PAIRS:
            d_obs = P[ci] - P[pi]; d_obs = d_obs / np.linalg.norm(d_obs)
            d = F @ (Rp_[pi] @ off[ci]); d = d / np.linalg.norm(d)
            cs.append(float(d_obs @ d))
        return min(cs)

    assert min_cos(Rp) >= 0.999
    # positive control: a sabotaged rotation MUST break the check
    Rbad = Rp.copy(); Rbad[35] = Rbad[35] @ _qmat_wxyz([0.7071, 0.7071, 0, 0])
    assert min_cos(Rbad) < 0.999

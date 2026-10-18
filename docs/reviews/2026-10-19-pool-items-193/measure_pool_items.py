"""Size the lab's tic-tac-toe pieces from Siva's annotation photos (#193).

Method.  The board pose is solved by PnP on the taped grid (photo_check/board_pnp.py, #188)
from the empty-board photo; the camera did not move between the empty-board and piece
photos (the empty-cell corners coincide).  Each piece photo is undistorted, its dark-blob
silhouette (tape pixels masked out) is compared with the silhouette of a model piece
projected through that pose, and the model's width and height are fitted to maximise the
overlap (IoU).  Cube: side w, height h, yaw, centre.  Cylinder: radius r, height h, centre.
Both camera models (Left/Right) are tried because the photos do not say which eye took them.

Run from the repo root:  python docs/reviews/2026-10-19-pool-items-193/measure_pool_items.py PHOTO_DIR
"""
import sys, os, glob, json
import numpy as np, cv2
from scipy.optimize import minimize

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '2026-10-14-taught-board-poses-185', 'photo_check'))
import board_pnp as bp

PHOTOS = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser('~/IITG-Reachy-Project/data/calibration/annotation')
DARK = 140          # grey level below which a pixel is "piece or tape" (a cube's lit top face is mid-grey)


def intr(eye):
    c = bp.CAM[eye]
    return np.array([[c['fx'], 0, c['cx']], [0, c['fy'], c['cy']], [0, 0, 1]]), np.array([*c['k'], 0, 0, 0])


def board_pose(eye, path):
    K, D = intr(eye)
    und = cv2.undistort(cv2.imread(path, 0), K, D)
    quads, partial, _ = bp.detect(und)
    grid = bp.pick_grid(quads, partial, und.shape)
    B = bp.cells_board(); obj, imgp = [], []
    for (r, c), qi in sorted(grid.items()):
        q = quads[qi][1]; ctr = B[(r, c)]
        q = q[np.argsort(q[:, 1])]; near = q[2:][np.argsort(q[2:, 0])]; far = q[:2][np.argsort(q[:2, 0])]
        for p, dx, dy in ((near[0], -1, 1), (near[1], -1, -1), (far[0], 1, 1), (far[1], 1, -1)):
            obj.append([ctr[0] + dx * bp.HALF, ctr[1] + dy * bp.HALF, 0.0]); imgp.append(p)
    obj, imgp = np.array(obj), np.array(imgp)
    imgp = cv2.cornerSubPix(und, imgp.astype(np.float32).reshape(-1, 1, 2), (5, 5), (-1, -1),
                            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.01)).reshape(-1, 2)
    ok, rvec, tvec = cv2.solvePnP(obj, imgp, K, None, flags=cv2.SOLVEPNP_ITERATIVE)
    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, None)
    rms = float(np.sqrt(((proj.reshape(-1, 2) - imgp) ** 2).sum(1).mean()))
    return K, rvec, tvec, rms, len(grid)


def project(K, rvec, tvec, P):
    return cv2.projectPoints(np.asarray(P, float).reshape(-1, 1, 3), rvec, tvec, K, None)[0].reshape(-1, 2)


def cube_pts(x, y, yaw, w, h):
    c, s = np.cos(yaw), np.sin(yaw)
    out = []
    for dx in (-.5, .5):
        for dy in (-.5, .5):
            for z in (0, h):
                out.append([x + c * dx * w - s * dy * w, y + s * dx * w + c * dy * w, z])
    return np.array(out)


def cyl_pts(x, y, r, h):
    a = np.linspace(0, 2 * np.pi, 40, endpoint=False)
    return np.array([[x + r * np.cos(t), y + r * np.sin(t), z] for z in (0, h) for t in a])


def silhouette(K, rvec, tvec, P, shape):
    m = np.zeros(shape, np.uint8)
    cv2.fillConvexPoly(m, cv2.convexHull(project(K, rvec, tvec, P).astype(np.float32)).astype(np.int32), 1)
    return m


def blobs(und_gray, tape):
    dark = (cv2.GaussianBlur(und_gray, (3, 3), 0) < DARK)
    return dark & ~tape, dark


def fit(kind, K, rvec, tvec, mask, ignore, x0, y0):
    shape = mask.shape; best = None
    def cost(p):
        if kind == 'cube':
            x, y, yaw, w, h = p; P = cube_pts(x, y, yaw, w, h)
            if not (0.02 < w < 0.12 and 0.02 < h < 0.12): return 1.0
        else:
            x, y, r, h = p; P = cyl_pts(x, y, r, h)
            if not (0.01 < r < 0.06 and 0.02 < h < 0.15): return 1.0
        s = silhouette(K, rvec, tvec, P, shape).astype(bool) & ~ignore
        m = mask & ~ignore
        return 1.0 - (s & m).sum() / max(1, (s | m).sum())
    starts = ([[x0, y0, yaw, w, h] for yaw in (0, .4, -.4) for w, h in ((.05, .05), (.07, .07))] if kind == 'cube'
              else [[x0, y0, r, h] for r, h in ((.02, .05), (.03, .08))])
    for s0 in starts:
        for _ in range(2):
            r = minimize(cost, s0, method='Nelder-Mead', options=dict(xatol=1e-4, fatol=1e-5, maxiter=1500))
            s0 = r.x
        if best is None or r.fun < best.fun: best = r
    return best


def pieces(mask, K, rvec, tvec):
    """Connected dark blobs off the tape -> board-plane position of each blob's lowest edge midpoint."""
    n, lab, st, cen = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    R, _ = cv2.Rodrigues(rvec); out = []
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] < 250 or st[i, cv2.CC_STAT_AREA] > 6000: continue
        x, y, w, h, _ = st[i]
        px = np.array([[x + w / 2, y + h - 3, 1.0]])           # bottom edge of the blob
        ray = np.linalg.inv(K) @ px.T; ray = R.T @ ray; origin = -R.T @ tvec.reshape(3, 1)
        t = -origin[2] / ray[2]; g = (origin + t * ray).ravel()
        d = min(np.hypot(g[0] - bp.cells_board()[k][0], g[1] - bp.cells_board()[k][1]) for k in bp.cells_board())
        if d < 0.08: out.append((g[0], g[1], int(st[i, cv2.CC_STAT_AREA])))
    return out


if __name__ == '__main__':
    results = {}
    for eye in ('Left', 'Right'):
        K, rvec, tvec, rms, ncell = board_pose(eye, f'{PHOTOS}/empty/image_003.jpg')
        print(f'{eye}: board pose from empty/image_003.jpg  cells={ncell}  rms={rms:.2f}px')
        und_e = cv2.undistort(cv2.imread(f'{PHOTOS}/empty/image_003.jpg', 0), *intr(eye))
        tape = cv2.dilate((cv2.GaussianBlur(und_e, (3, 3), 0) < DARK).astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
        for f in sorted(glob.glob(f'{PHOTOS}/cube/*.jpg')) + sorted(glob.glob(f'{PHOTOS}/cylinder/*.jpg')):
            kind = 'cube' if '/cube/' in f else 'cylinder'
            und = cv2.undistort(cv2.imread(f, 0), *intr(eye))
            mask, dark = blobs(und, tape)
            ignore = tape                                         # tape pixels: do not score
            for (gx, gy, area) in pieces(mask, K, rvec, tvec):
                r = fit(kind, K, rvec, tvec, mask, ignore, gx + 0.0, gy)
                rec = dict(eye=eye, photo=f'{kind}/{os.path.basename(f)}', at=[round(gx, 3), round(gy, 3)],
                           iou=round(1 - r.fun, 3), params=[round(float(v), 4) for v in r.x])
                results.setdefault(kind, []).append(rec)
                print(' ', rec)
    json.dump(results, open(os.path.join(os.path.dirname(__file__), 'measure_pool_items.json'), 'w'), indent=1)

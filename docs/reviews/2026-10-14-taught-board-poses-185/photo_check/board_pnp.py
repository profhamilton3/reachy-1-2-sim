"""Board pose from each MCC photo (no head angles): PnP on the taped grid, then torso frame.

Intrinsics: scenes/calibration_measured_2026_08_27.yaml, in the 480x640 stored frame.
Board: 3x3 cells on a 6 in pitch, cell interior 5.08 in (measured 5.06-5.12).
Board frame: origin at the centre cell, X forward (away from robot), Y left, Z up.
"""
import sys, os, glob, json
import numpy as np, cv2

IN = 0.0254
PITCH, HALF = 6 * IN, 5.08 * IN / 2
CAM = {  # portrait 480x640: fx (short axis), fy (long axis), cx, cy, k1, k2; y offset in head
    'Left':  dict(fx=407.0, fy=408.4, cx=253.6, cy=296.4, k=(-0.3163, 0.1027), y=+0.03625),
    'Right': dict(fx=398.8, fy=400.9, cx=225.6, cy=331.1, k=(-0.3895, 0.1838), y=-0.03625),
}
NECK = np.array([0.015, 0.0, 0.095])
OPT2HEAD = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], float)   # optical (x right, y down, z fwd) -> head

def cells_board():
    out = {}
    for r in range(3):          # r = 0 near .. 2 far   (sim r1..r3)
        for c in range(3):      # c = 0 left (+y) .. 2 right
            out[(r, c)] = np.array([(r - 1) * PITCH, (1 - c) * PITCH])
    return out

def detect(gray):
    """White cells = bright components enclosed by the dark tape (global Otsu on the lower image)."""
    h, w = gray.shape
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    t, _ = cv2.threshold(blur[h // 3:], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th = (blur > t).astype(np.uint8) * 255
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    cnts, _ = cv2.findContours(th, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    quads, partial = [], []
    for c in cnts:
        a = cv2.contourArea(c)
        if a < 800 or a > 60000: continue
        x, y, bw, bh = cv2.boundingRect(c)
        touches = x <= 2 or y <= 2 or x + bw >= w - 3 or y + bh >= h - 3
        ap = cv2.approxPolyDP(c, 0.03 * cv2.arcLength(c, True), True)
        if touches:
            partial.append((x, y, bw, bh)); continue
        if len(ap) != 4 or not cv2.isContourConvex(ap) or not (0.4 < bw / bh < 3.5): continue
        if a / cv2.contourArea(cv2.convexHull(c)) < 0.92: continue
        quads.append((a, ap.reshape(4, 2).astype(float)))
    return quads, partial, th

def pick_grid(quads, partial, shape):
    """Label visible cells (r, c). Needs whole rows of 3; resolves which rows are visible."""
    h, w = shape
    if len(quads) < 6: return None
    cents = np.array([q[1].mean(0) for q in quads]); areas = np.array([q[0] for q in quads])
    med = np.median(areas)
    keep = [i for i in range(len(quads)) if 0.3 < areas[i] / med < 3]
    # cluster into rows by image y
    keep.sort(key=lambda i: -cents[i, 1])
    rows, cur = [], [keep[0]]
    for i in keep[1:]:
        if abs(cents[i, 1] - np.mean([cents[j, 1] for j in cur])) < 0.5 * np.sqrt(med): cur.append(i)
        else: rows.append(cur); cur = [i]
    rows.append(cur)
    rows = [sorted(r, key=lambda i: cents[i, 0]) for r in rows if len(r) == 3]
    if len(rows) not in (2, 3): return None
    rows = rows[:3]
    if len(rows) == 3: labels = [0, 1, 2]
    else:
        xs = [cents[i, 0] for i in rows[0]]
        below = [p for p in partial if p[1] + p[3] >= h - 3 and min(xs) - 40 < p[0] + p[2] / 2 < max(xs) + 40]
        above = [p for p in partial if p[1] <= 2 and min(xs) - 40 < p[0] + p[2] / 2 < max(xs) + 40]
        if len(below) >= 2 and not above: labels = [1, 2]
        elif len(above) >= 2 and not below: labels = [0, 1]
        else: return None
    return {(labels[k], c): rows[k][c] for k in range(len(rows)) for c in range(3)}

def solve(img_path, eye):
    """Board centre in the torso frame from one photo; no head angles needed.

    The board is level and square to the torso (taught poses: under 1 deg), so the
    board's orientation in the image gives the camera's orientation in the torso,
    and with it the camera position (URDF neck origin + camera offset in the head).
    """
    cam = CAM[eye]
    K = np.array([[cam['fx'], 0, cam['cx']], [0, cam['fy'], cam['cy']], [0, 0, 1]])
    D = np.array([cam['k'][0], cam['k'][1], 0, 0, 0])
    gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    und = cv2.undistort(gray, K, D)                      # detect on undistorted pixels (same K)
    quads, partial, th = detect(und); grid = pick_grid(quads, partial, und.shape)
    if grid is None: return None, und, (quads, None, None)
    B = cells_board(); obj, imgp = [], []
    for (r, c), qi in sorted(grid.items()):
            q = quads[qi][1]; ctr = B[(r, c)]
            # corner -> board corner via the cell's own orientation: sort by image y then x
            q = q[np.argsort(q[:, 1])]; near = q[2:][np.argsort(q[2:, 0])]; far = q[:2][np.argsort(q[:2, 0])]
            for (img_pt, dx, dy) in ((near[0], -1, +1), (near[1], -1, -1), (far[0], +1, +1), (far[1], +1, -1)):
                    obj.append([ctr[0] + dx * HALF, ctr[1] + dy * HALF, 0.0]); imgp.append(img_pt)
    obj, imgp = np.array(obj), np.array(imgp)
    for _ in range(2):  # refine corners on the undistorted image
        imgp = cv2.cornerSubPix(und, imgp.astype(np.float32).reshape(-1, 1, 2), (5, 5), (-1, -1),
                                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.01)).reshape(-1, 2).astype(float)
    ok, rvec, tvec = cv2.solvePnP(obj, imgp, K, None, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok: return None, und, quads
    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, None)
    rms = float(np.sqrt(((proj.reshape(-1, 2) - imgp) ** 2).sum(1).mean()))
    R_cb, _ = cv2.Rodrigues(rvec)                        # board -> camera
    # Torso frame, assuming the board is level and square to the torso (taught poses: <1 deg).
    R_tb = np.eye(3)
    R_tc = R_tb @ R_cb.T                                 # camera (optical) -> torso
    R_head = R_tc @ OPT2HEAD.T                           # head -> torso
    cam_t = NECK + R_head @ np.array([0.0033, cam['y'], 0.061])
    # board origin in camera = tvec; in torso = cam_t + R_tc @ tvec
    board_t = cam_t + R_tc @ tvec.ravel()
    # head angles implied (Ry(0.174) @ Rx(roll) @ Ry(pitch) @ Rz(yaw) convention, for sanity only)
    Rn = cv2.Rodrigues(np.array([0, -0.174, 0.0]))[0] @ R_head
    return dict(eye=eye, file=os.path.basename(img_path), rms_px=round(rms, 2),
                board_in_camera_m=np.round(tvec.ravel(), 4).tolist(),
                cells=len(grid), board_centre_in_torso_m=np.round(board_t, 4).tolist(),
                camera_in_torso_m=np.round(cam_t, 4).tolist(),
                head_rotation_deg=np.round(np.degrees(cv2.Rodrigues(Rn)[0].ravel()), 1).tolist()), und, (quads, grid, imgp)

if __name__ == '__main__':
    pics, out = sys.argv[1], sys.argv[2]
    results = []
    for f in sorted(glob.glob(os.path.join(pics, '*.png'))):
        eye = 'Left' if os.path.basename(f).startswith('Left') else 'Right'
        r, und, extra = solve(f, eye)
        if r is None:
            print('NO GRID', os.path.basename(f)); continue
        results.append(r)
        quads, grid, imgp = extra
        vis = cv2.cvtColor(und, cv2.COLOR_GRAY2BGR)
        for p in imgp: cv2.circle(vis, tuple(int(v) for v in p), 3, (0, 0, 255), -1)
        cv2.imwrite(os.path.join(out, os.path.basename(f).replace(' ', '_')), vis)
        print(r['eye'], r['file'][-12:], 'cells', r['cells'], 'rms', r['rms_px'], 'board in torso', r['board_centre_in_torso_m'], 'head rot', r['head_rotation_deg'])
    json.dump(results, open(os.path.join(out, 'results.json'), 'w'), indent=1)
    B = np.array([r['board_centre_in_torso_m'] for r in results if r['rms_px'] < 2.0])
    print(f'\nusable views (rms < 2 px): {len(B)} of {len(results)}')
    if len(B):
        print('board centre in torso: median', np.round(np.median(B, 0), 4), ' std', np.round(B.std(0), 4))

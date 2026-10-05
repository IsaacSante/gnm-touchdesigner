"""Precompute the data that drives the GNM head from MediaPipe face landmarks.

Writes tools/face_fit/face_fit.npz, which install_face_tracking.py embeds in
the component. Run once (or after changing the solver settings below):

    python3 tools/face_fit/build_face_fit.py /path/to/gnm_head.npz

Inputs:
  * gnm_head.npz -- Google GNM Head v3.0 model data (Apache-2.0).
  * correspondence.npz -- MediaPipe-landmark -> GNM-vertex mapping from
    edualvarado/gnm-webcam-puppet (Apache-2.0), downloaded if missing. It holds
    473 of MediaPipe's 478 landmarks, each with a GNM vertex, a "rigid" flag for
    skull-fixed points, and where the landmarker places that point on a
    *neutral* GNM face.

The method follows gnm-webcam-puppet's solver: per frame, a similarity fit of
the rigid points gives head pose; the remaining displacement from the neutral
reference is projected onto a per-region subset of the expression basis with
ridge regularization; the solve is a single matrix product per frame.
"""
import os
import sys
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CORR_URL = ('https://raw.githubusercontent.com/edualvarado/gnm-webcam-puppet/'
            'master/webcam_puppet/assets/correspondence.npz')

# Solver settings, from gnm-webcam-puppet (tuned there on synthetic recovery).
# Tongue and pupils aren't observable from landmarks, so they're not solved.
REGION_COMPONENTS = {'left_eye_region': 40, 'right_eye_region': 40,
                     'lower_face_region': 80}
EXPRESSION_REGULARIZATION = 1e-5
IDENTITY_COMPONENTS = 40
IDENTITY_REGULARIZATION = 2e-3
# The tongue isn't visible to the tracker, so its coefficients would stay at
# zero -- and then an opening jaw leaves the tongue behind, bulging into the
# mouth (it reads as a swollen lower lip). Instead, tongue components are set
# so the tongue moves with the lower teeth.
TONGUE_FOLLOW_REGULARIZATION = 1e-4

# MediaPipe iris centres. Iris landmarks drive the eye joints (gaze) rather
# than the expression solve, where they'd be misread as eyelid shapes.
IRIS_FIRST = 468
IRIS_CENTRES = (468, 473)


# MediaPipe's inner-lip contours (from mediapipe face_mesh_connections), minus
# the shared mouth corners 78 and 308.
UPPER_INNER_LIP = [191, 80, 81, 82, 13, 312, 311, 310, 415]
LOWER_INNER_LIP = [95, 88, 178, 87, 14, 317, 402, 318, 324]
# Missing from the correspondence -> the landmark whose reference position
# stands in for it (left/right twin, mirrored in x; 14 sits under 13 on a
# closed mouth).
MISSING_LIP_TWINS = {415: (191, True), 317: (87, True), 14: (13, False)}


def fix_inner_lips(gnm, landmarks, vertices, rigid, reference):
    """Re-attach MediaPipe's inner-lip landmarks to the correct GNM lip.

    On a closed mouth the inner edges of both lips coincide, so the
    render-based correspondence often matched an upper-inner-lip landmark to
    a lower-lip vertex and vice versa (and dropped three). Driven live, an
    opening mouth then pulls GNM's lips the wrong way: the jaw barely opens
    while the lower lip swells. Each inner-lip landmark is re-matched to the
    nearest vertex on the *inner edge* of its own lip.
    """
    groups = [str(n) for n in gnm['vertex_group_names']]
    verts = gnm['template_vertex_positions']
    upper = np.where(gnm['vertex_groups'][groups.index('upper_lip')] > 0.5)[0]
    lower = np.where(gnm['vertex_groups'][groups.index('lower_lip')] > 0.5)[0]

    def inner_edge(own, other):
        # vertices of one lip lying within 1.5 mm of the other lip
        d = np.linalg.norm(verts[own][:, None] - verts[other][None], axis=-1).min(1)
        return own[d < 0.0015]

    edges = {'upper': inner_edge(upper, lower), 'lower': inner_edge(lower, upper)}
    landmarks, vertices = list(landmarks), list(vertices)
    rigid, reference = list(rigid), list(reference)

    for lm, (twin, mirror) in MISSING_LIP_TWINS.items():
        if lm in landmarks or twin not in landmarks:
            continue
        ref = reference[landmarks.index(twin)].copy()
        if mirror:
            ref[0] = -ref[0]
        landmarks.append(lm); vertices.append(-1); rigid.append(False); reference.append(ref)

    changed = []
    for side, lms in (('upper', UPPER_INNER_LIP), ('lower', LOWER_INNER_LIP)):
        edge = edges[side]
        for lm in lms:
            if lm not in landmarks:
                continue
            row = landmarks.index(lm)
            # nearest inner-edge vertex to where the landmarker puts this point
            best = int(edge[np.argmin(np.linalg.norm(verts[edge] - reference[row], axis=1))])
            if best != vertices[row]:
                changed.append(lm)
                vertices[row] = best
    print('Inner-lip landmarks re-attached: %s' % changed)
    order = np.argsort(landmarks)
    return (np.array(landmarks, np.int32)[order], np.array(vertices, np.int32)[order],
            np.array(rigid, bool)[order], np.array(reference, np.float32)[order])


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    gnm = np.load(sys.argv[1])

    corr_path = os.path.join(HERE, 'correspondence.npz')
    if not os.path.exists(corr_path):
        urllib.request.urlretrieve(CORR_URL, corr_path)
    corr = np.load(corr_path)

    landmarks = corr['landmark_indices'].astype(np.int32)
    vertices = corr['vertex_indices'].astype(np.int32)
    rigid = corr['rigid'].astype(bool)
    reference = corr['reference_landmarks'].astype(np.float32)
    landmarks, vertices, rigid, reference = fix_inner_lips(
        gnm, landmarks, vertices, rigid, reference)
    expr_rows = landmarks < IRIS_FIRST

    # Expression components, budgeted per region (the basis is concatenated
    # per region, variance-ordered within each).
    names = [str(n) for n in gnm['expression_names']]
    components = []
    for region, count in REGION_COMPONENTS.items():
        idx = [i for i, n in enumerate(names) if n.rsplit('_', 1)[0] == region]
        components += idx[:count]
    components = np.array(sorted(components), np.int32)

    expr_basis = gnm['expression_basis'][components][:, vertices[expr_rows]]   # (K, Me, 3)

    id_basis = gnm['vertex_identity_basis'][:IDENTITY_COMPONENTS][:, vertices[expr_rows]]

    # Tongue-follows-jaw: for each solved component k, the tongue coefficients
    # t_k that make the tongue move like the lower teeth do under k. Linear in
    # the expression, so per frame it's tongue = tongue_follow @ expr.
    groups_all = [str(n) for n in gnm['vertex_group_names']]
    tongue_v = gnm['vertex_groups'][groups_all.index('tongue')] > 0.5
    teeth_v = gnm['vertex_groups'][groups_all.index('lower_teeth_and_gums')] > 0.5
    tongue_comps = [i for i, n in enumerate(names) if n.startswith('tongue')]
    tb = gnm['expression_basis'][tongue_comps][:, tongue_v].reshape(len(tongue_comps), -1).T.astype(np.float64)
    full_b = gnm['expression_basis'][components]
    target = (full_b[:, teeth_v].mean(1)[:, None, :] - full_b[:, tongue_v]).reshape(len(components), -1).T
    tongue_follow = np.linalg.solve(tb.T @ tb + TONGUE_FOLLOW_REGULARIZATION * np.eye(tb.shape[1]),
                                    tb.T @ target).astype(np.float32)          # (T, K)

    # Which iris belongs to which GNM eye joint: match by side (x sign).
    joints = [str(n) for n in gnm['joint_names']]
    jpos = gnm['template_joint_positions']
    iris_rows = {}
    for lm in IRIS_CENTRES:
        row = int(np.where(landmarks == lm)[0][0])
        side = 'left_eye' if np.sign(reference[row, 0]) == np.sign(jpos[joints.index('left_eye'), 0]) else 'right_eye'
        iris_rows[side] = row
    # Eyeball radius: eye joint (the eyeball centre) to the sclera surface.
    # Not to the iris landmark's vertex -- the correspondence puts iris
    # landmarks on interior eye geometry, seen through the cornea.
    groups = [str(n) for n in gnm['vertex_group_names']]
    sclera = gnm['vertex_groups'][groups.index('scleras')] > 0.5
    radius = float(np.mean([
        np.median(np.linalg.norm(
            gnm['template_vertex_positions'][sclera & (gnm['vertex_groups'][groups.index(s)] > 0.5)]
            - jpos[joints.index(s)], axis=1))
        for s in ('left_eye', 'right_eye')]))

    out = os.path.join(HERE, 'face_fit.npz')
    np.savez_compressed(
        out,
        landmark_indices=landmarks,
        vertex_indices=vertices,
        rigid=rigid,
        reference=reference,
        expr_rows=expr_rows,
        # Basis rows at the corresponded points, (K, 3 * points). The ridge
        # projections are built from these at runtime, so the depth weight
        # can be a parameter (face_fit_callbacks.projections).
        expr_basis=expr_basis.reshape(len(components), -1).astype(np.float32),
        expr_regularization=np.float32(EXPRESSION_REGULARIZATION),
        expr_channels=np.array(['e_' + names[i] for i in components]),
        id_basis=id_basis.reshape(IDENTITY_COMPONENTS, -1).astype(np.float32),
        id_regularization=np.float32(IDENTITY_REGULARIZATION),
        id_channels=np.array(['i_' + str(n) for n in gnm['identity_names'][:IDENTITY_COMPONENTS]]),
        tongue_follow=tongue_follow,
        tongue_channels=np.array(['e_' + names[i] for i in tongue_comps]),
        left_iris_row=np.int32(iris_rows['left_eye']),
        right_iris_row=np.int32(iris_rows['right_eye']),
        eye_radius=np.float32(radius),
    )
    print('Wrote %s (%.1f MB): %d landmarks (%d rigid), %d expression + %d identity components, '
          'eye radius %.1f mm, left eye = landmark %d, right eye = landmark %d' % (
              os.path.relpath(out), os.path.getsize(out) / 1e6, len(landmarks), rigid.sum(),
              len(components), IDENTITY_COMPONENTS, radius * 1000,
              landmarks[iris_rows['left_eye']], landmarks[iris_rows['right_eye']]))


if __name__ == '__main__':
    main()

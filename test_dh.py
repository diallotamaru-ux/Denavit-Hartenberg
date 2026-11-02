"""Tests de validation du module dh.py.

Lancer : python test_dh.py   (ou : pytest -v test_dh.py)
"""

import numpy as np

from dh import (
    dh_modified,
    dh_standard,
    forward_kinematics,
    pose_to_xyz_rpy,
    standard_to_modified,
    tool_from_standard_last_row,
)

RNG = np.random.default_rng(0)

# --- Robot 1 : bras plan 2R -------------------------------------------------
L1, L2 = 0.5, 0.3
TABLE_2R = [
    (0.0, 0.0, L1, 0.0),
    (0.0, 0.0, L2, 0.0),
]

# --- Robot 2 : UR5, paramètres DH standard publiés par Universal Robots -----
# (theta_offset, d, a, alpha) ; 6 articulations rotoïdes.
# Valeurs retranscrites de mémoire : à recouper avec la page "DH parameters"
# d'Universal Robots avant toute utilisation sur un robot réel.
TABLE_UR5 = [
    (0.0, 0.089159, 0.0, np.pi / 2),
    (0.0, 0.0, -0.425, 0.0),
    (0.0, 0.0, -0.39225, 0.0),
    (0.0, 0.10915, 0.0, np.pi / 2),
    (0.0, 0.09465, 0.0, -np.pi / 2),
    (0.0, 0.0823, 0.0, 0.0),
]


def is_rigid_transform(T: np.ndarray, tol: float = 1e-12) -> bool:
    R = T[:3, :3]
    return (
        np.allclose(R @ R.T, np.eye(3), atol=tol)
        and np.isclose(np.linalg.det(R), 1.0, atol=tol)
        and np.allclose(T[3], [0, 0, 0, 1])
    )


def test_matrices_are_rigid_transforms():
    """Toute matrice DH doit être une transformation rigide (R orthogonale, det = 1)."""
    for _ in range(100):
        theta, alpha = RNG.uniform(-np.pi, np.pi, 2)
        d, a = RNG.uniform(-1, 1, 2)
        assert is_rigid_transform(dh_standard(theta, d, a, alpha))
        assert is_rigid_transform(dh_modified(theta, d, a, alpha))


def test_2r_matches_analytic_formula():
    """x = L1 cos q1 + L2 cos(q1+q2) ; y = L1 sin q1 + L2 sin(q1+q2)."""
    for _ in range(100):
        q1, q2 = RNG.uniform(-np.pi, np.pi, 2)
        T = forward_kinematics(TABLE_2R, [q1, q2])
        x = L1 * np.cos(q1) + L2 * np.cos(q1 + q2)
        y = L1 * np.sin(q1) + L2 * np.sin(q1 + q2)
        assert np.isclose(T[0, 3], x)
        assert np.isclose(T[1, 3], y)
        assert np.isclose(T[2, 3], 0.0)
        # orientation de l'effecteur : rotation d'angle q1 + q2 autour de z
        phi = np.arctan2(T[1, 0], T[0, 0])
        diff = np.angle(np.exp(1j * (phi - (q1 + q2))))
        assert np.isclose(diff, 0.0, atol=1e-12)


def test_2r_zero_and_90_degrees():
    """Deux configurations évidentes à vérifier à la main."""
    T0 = forward_kinematics(TABLE_2R, [0.0, 0.0])
    assert np.allclose(T0[:3, 3], [L1 + L2, 0.0, 0.0])
    T90 = forward_kinematics(TABLE_2R, [np.pi / 2, 0.0])
    assert np.allclose(T90[:3, 3], [0.0, L1 + L2, 0.0], atol=1e-12)


def test_standard_equals_modified_after_reindexing():
    """Un même robot décrit dans les deux conventions donne la même pose,
    à la transformation fixe finale Trans_x(a_n) Rot_x(alpha_n) près."""
    table_mod = standard_to_modified(TABLE_UR5)
    T_tool = tool_from_standard_last_row(TABLE_UR5)
    for _ in range(50):
        q = RNG.uniform(-np.pi, np.pi, 6)
        T_std = forward_kinematics(TABLE_UR5, q, convention="standard")
        T_mod = forward_kinematics(table_mod, q, convention="modified") @ T_tool
        assert np.allclose(T_std, T_mod, atol=1e-12)


def test_naive_copy_between_conventions_is_wrong():
    """Piège de la section 7 : réutiliser le tableau standard tel quel en
    convention modifiée donne une pose fausse, sans aucune erreur levée."""
    q = np.array([0.3, -0.5, 0.8, 0.1, -0.2, 0.4])
    T_std = forward_kinematics(TABLE_UR5, q, convention="standard")
    T_wrong = forward_kinematics(TABLE_UR5, q, convention="modified")
    assert not np.allclose(T_std, T_wrong, atol=1e-3)


def test_ur5_zero_configuration():
    """UR5 en configuration zéro (tous q = 0), valeurs calculées à la main :
    avec ces paramètres DH, l'effecteur est à
        x = a2 + a3 = -0.81725
        y = -(d4 + d6) = -0.19145   (d4 = 0.10915, d6 = 0.0823)
        z = d1 - d5 = -0.00544 ... voir calcul ci-dessous
    """
    T = forward_kinematics(TABLE_UR5, np.zeros(6))
    d1, d4, d5, d6 = 0.089159, 0.10915, 0.09465, 0.0823
    a2, a3 = -0.425, -0.39225
    expected = np.array([a2 + a3, -(d4 + d6), d1 - d5])
    assert np.allclose(T[:3, 3], expected, atol=1e-9), (T[:3, 3], expected)


def test_prismatic_joint():
    """Articulation prismatique : q s'ajoute à d. Robot 1 articulation P le long de z."""
    table = [(0.0, 0.2, 0.0, 0.0)]
    T = forward_kinematics(table, [0.5], joint_types=["P"])
    assert np.allclose(T[:3, 3], [0.0, 0.0, 0.7])


def test_input_validation():
    for bad in (
        lambda: forward_kinematics(TABLE_2R, [0.0]),
        lambda: forward_kinematics(TABLE_2R, [0.0, 0.0], convention="xyz"),
        lambda: forward_kinematics(TABLE_2R, [0.0, 0.0], joint_types=["R", "X"]),
    ):
        try:
            bad()
        except ValueError:
            continue
        raise AssertionError("ValueError attendue")


def test_pose_to_xyz_rpy_roundtrip():
    """R = Rz(yaw) Ry(pitch) Rx(roll) doit être retrouvée par pose_to_xyz_rpy."""
    for _ in range(100):
        roll, yaw = RNG.uniform(-np.pi, np.pi, 2)
        pitch = RNG.uniform(-1.4, 1.4)  # loin de la singularité
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)
        Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
        Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
        Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
        T = np.eye(4)
        T[:3, :3] = Rz @ Ry @ Rx
        T[:3, 3] = [0.1, 0.2, 0.3]
        p, rpy = pose_to_xyz_rpy(T)
        assert np.allclose(p, [0.1, 0.2, 0.3])
        assert np.allclose(rpy, [roll, pitch, yaw], atol=1e-9)


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"OK    {name}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"ECHEC {name} : {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} tests réussis")
    raise SystemExit(1 if failed else 0)

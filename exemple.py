"""Exemple court pour illustrer le post : bras plan 2R puis robot UR5."""
import numpy as np

from dh import forward_kinematics

np.set_printoptions(precision=3, suppress=True)

# Bras plan 2R : L1 = 0,5 m, L2 = 0,3 m, q1 = 30°, q2 = 45°
table_2r = [(0.0, 0.0, 0.5, 0.0), (0.0, 0.0, 0.3, 0.0)]
T = forward_kinematics(table_2r, np.radians([30, 45]))
print("Bras plan 2R, position de l'effecteur (m) :", T[:3, 3])

# Robot 6 axes UR5 : paramètres DH standard (theta, d, a, alpha)
ur5 = [
    (0.0, 0.089159, 0.0, np.pi / 2),
    (0.0, 0.0, -0.425, 0.0),
    (0.0, 0.0, -0.39225, 0.0),
    (0.0, 0.10915, 0.0, np.pi / 2),
    (0.0, 0.09465, 0.0, -np.pi / 2),
    (0.0, 0.0823, 0.0, 0.0),
]
q = np.radians([0, -110, 100, -80, -90, 0])
print("\nUR5, matrice T = T1·T2·T3·T4·T5·T6 :")
print(forward_kinematics(ur5, q))

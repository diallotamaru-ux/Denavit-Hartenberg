"""Cinématique directe par la convention de Denavit-Hartenberg.

Deux conventions sont implémentées. Leurs tableaux ne sont PAS interchangeables.

Standard (Siciliano, Spong, cours français) :
    T(i-1, i) = Rot_z(theta_i) Trans_z(d_i) Trans_x(a_i) Rot_x(alpha_i)

Modifiée (Craig) :
    T(i-1, i) = Rot_x(alpha_{i-1}) Trans_x(a_{i-1}) Rot_z(theta_i) Trans_z(d_i)

Dans les deux cas, un tableau est une liste de lignes (theta, d, a, alpha)
avec la convention d'indexation propre à chaque variante.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

# Une ligne de tableau DH : (theta_offset, d, a, alpha)
# Le type d'articulation est donné séparément par `joint_types`.
DHRow = tuple[float, float, float, float]


def dh_standard(theta: float, d: float, a: float, alpha: float) -> np.ndarray:
    """Matrice homogène 4x4 T(i-1, i), convention standard.

    theta : rotation autour de z_{i-1} (rad)
    d     : translation le long de z_{i-1} (m)
    a     : translation le long de x_i (m)
    alpha : rotation autour de x_i (rad)
    """
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array(
        [
            [ct, -st * ca, st * sa, a * ct],
            [st, ct * ca, -ct * sa, a * st],
            [0.0, sa, ca, d],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def dh_modified(theta: float, d: float, a_prev: float, alpha_prev: float) -> np.ndarray:
    """Matrice homogène 4x4 T(i-1, i), convention modifiée (Craig).

    theta      : rotation autour de z_i (rad)
    d          : translation le long de z_i (m)
    a_prev     : a_{i-1}, translation le long de x_{i-1} (m)
    alpha_prev : alpha_{i-1}, rotation autour de x_{i-1} (rad)
    """
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha_prev), np.sin(alpha_prev)
    return np.array(
        [
            [ct, -st, 0.0, a_prev],
            [st * ca, ct * ca, -sa, -d * sa],
            [st * sa, ct * sa, ca, d * ca],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def forward_kinematics(
    table: Sequence[DHRow],
    q: Iterable[float],
    joint_types: Sequence[str] | None = None,
    convention: str = "standard",
) -> np.ndarray:
    """Modèle géométrique direct : renvoie T(0, n) pour le vecteur q.

    table       : lignes (theta_offset, d, a, alpha), une par articulation.
    q           : variables articulaires. Pour une articulation 'R', q_i
                  s'ajoute à theta_offset_i. Pour 'P', q_i s'ajoute à d_i.
    joint_types : 'R' (rotoïde) ou 'P' (prismatique) par articulation.
                  Par défaut, toutes rotoïdes.
    convention  : 'standard' ou 'modified'.
    """
    q = list(q)
    n = len(table)
    if len(q) != n:
        raise ValueError(f"q a {len(q)} éléments, le tableau en a {n}")
    if joint_types is None:
        joint_types = ["R"] * n
    if len(joint_types) != n:
        raise ValueError("joint_types doit avoir la même longueur que le tableau")

    if convention == "standard":
        builder = dh_standard
    elif convention == "modified":
        builder = dh_modified
    else:
        raise ValueError("convention doit être 'standard' ou 'modified'")

    T = np.eye(4)
    for (theta0, d0, a, alpha), qi, jt in zip(table, q, joint_types):
        if jt == "R":
            theta, d = theta0 + qi, d0
        elif jt == "P":
            theta, d = theta0, d0 + qi
        else:
            raise ValueError(f"type d'articulation inconnu : {jt!r}")
        T = T @ builder(theta, d, a, alpha)
    return T


def standard_to_modified(table_std: Sequence[DHRow]) -> list[DHRow]:
    """Convertit un tableau standard en tableau modifié (toutes articulations
    rotoïdes ou prismatiques, chaîne série).

    Relation : le repère modifié i coïncide avec le repère standard i-1 ... mais
    translaté le long de son axe z. Concrètement, les paramètres a et alpha
    de la ligne i en standard deviennent a_{i} et alpha_{i} de la ligne i+1
    en modifiée. La dernière ligne modifiée reçoit (a, alpha) = (0, 0), et
    la transformation finale diffère d'une constante fixe (voir tests).

    Cette fonction est fournie pour illustrer le décalage d'indices. Elle ne
    reproduit PAS exactement la même pose d'effecteur sans le repère final fixe
    T_tool = Trans_x(a_n) Rot_x(alpha_n) appliqué à droite.
    """
    n = len(table_std)
    mod: list[DHRow] = []
    for i in range(n):
        theta, d, _a, _alpha = table_std[i]
        if i == 0:
            a_prev, alpha_prev = 0.0, 0.0
        else:
            a_prev, alpha_prev = table_std[i - 1][2], table_std[i - 1][3]
        mod.append((theta, d, a_prev, alpha_prev))
    return mod


def tool_from_standard_last_row(table_std: Sequence[DHRow]) -> np.ndarray:
    """Transformation fixe à appliquer à droite du produit modifié pour
    retrouver la pose de l'effecteur donnée par le tableau standard :
    Trans_x(a_n) Rot_x(alpha_n)."""
    _theta, _d, a_n, alpha_n = table_std[-1]
    ca, sa = np.cos(alpha_n), np.sin(alpha_n)
    return np.array(
        [
            [1.0, 0.0, 0.0, a_n],
            [0.0, ca, -sa, 0.0],
            [0.0, sa, ca, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def pose_to_xyz_rpy(T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Extrait la position (x, y, z) et les angles (roll, pitch, yaw) ZYX
    d'une matrice homogène. Utile pour comparer avec tf2_echo / URDF.

    Convention : R = Rz(yaw) Ry(pitch) Rx(roll). Singulier si |pitch| = pi/2.
    """
    R = T[:3, :3]
    p = T[:3, 3].copy()
    pitch = -np.arcsin(np.clip(R[2, 0], -1.0, 1.0))
    if abs(np.cos(pitch)) > 1e-9:
        roll = np.arctan2(R[2, 1], R[2, 2])
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:  # singularité : on fixe yaw = 0
        roll = np.arctan2(-R[1, 2], R[1, 1])
        yaw = 0.0
    return p, np.array([roll, pitch, yaw])

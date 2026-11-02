"""Animation 3D de la convention de Denavit-Hartenberg (rendu réaliste).

Toute la géométrie (repères, position de l'effecteur, trajectoire) est calculée
en Python avec dh.py. La page render/scene.html (Three.js) ne fait que dessiner
ce que Python lui envoie, image par image, puis Playwright capture chaque image
et ffmpeg assemble la vidéo.

Lancer :
    python render_animation.py            # vidéo complète -> dh_robot.mp4 + dh_robot.gif
    python render_animation.py --test     # quelques images de contrôle

Dépendances : numpy, playwright (Chromium), ffmpeg ; three.min.js r128 et les
polices Archivo / JetBrains Mono dans render/ (voir README).
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from playwright.sync_api import sync_playwright

from dh import dh_standard, forward_kinematics

FPS = 24
D2R = np.pi / 180
HERE = Path(__file__).parent
RENDER_DIR = HERE / "render"
REQUIRED = ["scene.html", "three.min.js",
            "fonts/archivo-latin-400-normal.woff2", "fonts/archivo-latin-700-normal.woff2",
            "fonts/archivo-latin-800-normal.woff2", "fonts/jetbrains-mono-latin-400-normal.woff2",
            "fonts/jetbrains-mono-latin-600-normal.woff2"]


def check_files():
    """Vérifie que le dossier render/ est complet avant de lancer le navigateur."""
    missing = [f for f in REQUIRED if not (RENDER_DIR / f).is_file()]
    if missing:
        sys.exit("Fichiers manquants dans " + str(RENDER_DIR) + " :\n  - " + "\n  - ".join(missing)
                 + "\nDécompresse render.zip à côté de render_animation.py (dossier render/).")
    if not shutil.which("ffmpeg"):
        sys.exit("ffmpeg est introuvable. Sous Windows : winget install Gyan.FFmpeg, puis rouvre le terminal.")

# Robot UR5 : paramètres DH standard (theta_offset, d, a, alpha)
UR5 = [
    (0.0, 0.089159, 0.0, np.pi / 2),
    (0.0, 0.0, -0.425, 0.0),
    (0.0, 0.0, -0.39225, 0.0),
    (0.0, 0.10915, 0.0, np.pi / 2),
    (0.0, 0.09465, 0.0, -np.pi / 2),
    (0.0, 0.0823, 0.0, 0.0),
]
TOOL_TIP = np.array([0.0, 0.0, 0.13, 1.0])  # extrémité des doigts dans le repère 6

SUBS = "₀₁₂₃₄₅₆₇₈₉"


def sub(n: int) -> str:
    return "".join(SUBS[int(c)] for c in str(n))


def num(v: float, d: int = 3) -> str:
    return f"{v:+.{d}f}".replace(".", ",").replace("-", "−")


def ease(t: float) -> float:
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)


def flat(T: np.ndarray) -> list[float]:
    return [round(float(v), 6) for v in T[:3, :].ravel()]


def chain(q: np.ndarray) -> list[np.ndarray]:
    """Repères T(0,0) … T(0,6) calculés avec dh.py."""
    return [np.eye(4)] + [forward_kinematics(UR5[:k], q[:k]) for k in range(1, 7)]


def pose(t: float) -> np.ndarray:
    """Mouvement de prise et dépose, effecteur toujours orienté vers le bas.

    Pour ce robot, l'outil reste vertical si q2 + q3 + q4 = −90°.
    """
    w = 2 * np.pi * t
    q1 = 50 * np.sin(w)
    q2 = -105 + 12 * np.sin(2 * w)
    q3 = 100 + 18 * np.sin(2 * w + 0.6)
    q4 = -90 - q2 - q3
    q6 = 60 * np.sin(w)
    return np.array([q1, q2, q3, q4, -90, q6]) * D2R


ROBOT_CAM = dict(target=[-0.12, 0.0, 0.47], elev=17, dist=2.3)
DEMO_CAM = dict(target=[0.18, 0.12, 0.43], elev=19, dist=2.2)


def robot_state(q, text, azim, frames_shown=(), trail=None, end=0.0):
    F = chain(q)
    s = {"cam": {**ROBOT_CAM, "azim": azim}, "robot": {"F": [flat(T) for T in F]}, "text": text}
    s["frames"] = [{"T": flat(F[k]), "L": 0.17, "dim": False} for k in frames_shown]
    if trail is not None and len(trail) > 1:
        s["tubes"] = [{"pts": [list(map(float, p)) for p in trail], "r": 0.0045}]
    if end:
        s["end"] = end
    return s, F


# ----------------------------------------------------------------------------
# Scènes
# ----------------------------------------------------------------------------
def scene_question(states):
    n = int(3.6 * FPS)
    for i in range(n):
        q = pose(0.06 * i / n)
        s, _ = robot_state(q, {
            "kick": "Robotique · modèle géométrique direct",
            "title": "Où se trouve l'effecteur ?",
            "sub": "Le robot ne le voit pas, mais le calcule à partir de ses angles articulaires.",
        }, azim=-112 + 8 * i / n)
        states.append(s)


def scene_four_transforms(states):
    th, d, a, al = 60 * D2R, 0.30, 0.40, 50 * D2R
    F0 = dh_standard(0, 0.06, 0, 0)  # repère i-1, légèrement au-dessus du sol
    names = ["Rot(z, θ)", "Trans(z, d)", "Trans(x, a)", "Rot(x, α)"]
    steps = [
        ("1 · Tourner de θ autour de z", "L'angle θ fait tourner le repère autour de l'axe de l'articulation."),
        ("2 · Monter de d le long de z", "La distance d fait glisser le repère le long de cet axe."),
        ("3 · Avancer de a le long de x", "La longueur a mène jusqu'à l'axe suivant : c'est la perpendiculaire commune."),
        ("4 · Incliner de α autour de x", "L'angle α aligne z sur l'axe de l'articulation suivante."),
    ]

    def foot(cur: int) -> str:
        parts = []
        for k, nm in enumerate(names):
            cls = "on" if k == cur else ("dim" if cur >= 0 and k > cur else "")
            parts.append(f'<span class="{cls}">{nm}</span>')
        return "T<sub>i</sub> = " + " · ".join(parts)

    def build(sv: float, text: dict, azim: float):
        f = [ease(sv - k) for k in range(4)]
        F1 = F0 @ dh_standard(th * f[0], 0, 0, 0)
        F2 = F1 @ dh_standard(0, d * f[1], 0, 0)
        F3 = F2 @ dh_standard(0, 0, a * f[2], 0)
        F4 = F3 @ dh_standard(0, 0, 0, al * f[3])
        o0, z0 = F0[:3, 3], F0[:3, 2]
        s = {"cam": {**DEMO_CAM, "azim": azim}, "text": text, "tubes": [], "labels": [], "dots": []}
        s["dashed"] = [[list(o0 - 0.05 * z0), list(o0 + 0.5 * z0)]]
        s["frames"] = [{"T": flat(F0), "L": 0.2, "dim": sv > 0}, {"T": flat(F4), "L": 0.24, "dim": False}]
        s["labels"] += [{"t": "x" + sub(0), "p": list(F0[:3, 3] + 0.24 * F0[:3, 0] - 0.06 * F0[:3, 1]), "c": "#d0352a"},
                        {"t": "z" + sub(0), "p": list(o0 + 0.55 * z0), "c": "#2b63e0"}]
        if f[0] > 0:
            R, t1 = 0.27, th * f[0]
            pts = [list((F0 @ [R * np.cos(u), R * np.sin(u), 0, 1])[:3]) for u in np.linspace(0, t1, 40)]
            s["tubes"].append({"pts": pts, "r": 0.006})
            s["labels"].append({"t": f"θ = {th / D2R:.0f}°", "p": list((F0 @ [1.45 * R * np.cos(t1 / 2), 1.45 * R * np.sin(t1 / 2), 0, 1])[:3])})
        if f[1] > 0:
            p, q = F1[:3, 3], F2[:3, 3]
            s["tubes"].append({"pts": [list(p), list((p + q) / 2), list(q)], "r": 0.007})
            s["labels"].append({"t": f"d = {d:.2f} m".replace(".", ","), "p": list((p + q) / 2 - 0.13 * F1[:3, 0] - 0.05 * F1[:3, 1])})
        if f[2] > 0:
            p, q = F2[:3, 3], F3[:3, 3]
            s["tubes"].append({"pts": [list(p), list((p + q) / 2), list(q)], "r": 0.007})
            s["labels"].append({"t": f"a = {a:.2f} m".replace(".", ","), "p": list((p + q) / 2 + 0.07 * F2[:3, 2])})
        if f[3] > 0:
            R, t1 = 0.2, al * f[3]
            pts = [list((F3 @ [0, -R * np.sin(u), R * np.cos(u), 1])[:3]) for u in np.linspace(0, t1, 40)]
            s["tubes"].append({"pts": pts, "r": 0.006})
            s["labels"].append({"t": f"α = {al / D2R:.0f}°", "p": list((F3 @ [0, -1.5 * R * np.sin(t1 / 2), 1.5 * R * np.cos(t1 / 2), 1])[:3])})
        if sv >= 4:
            o4, z4 = F4[:3, 3], F4[:3, 2]
            s["dashed"].append([list(o4 - 0.3 * z4), list(o4 + 0.45 * z4)])
        s["labels"] += [{"t": "x′", "p": list(F4[:3, 3] + 0.31 * F4[:3, 0]), "c": "#d0352a"},
                        {"t": "z′", "p": list(F4[:3, 3] + 0.31 * F4[:3, 2]), "c": "#2b63e0"}]
        return s

    kick = "Du repère i−1 au repère i"
    move, hold, intro, outro = int(1.4 * FPS), int(0.65 * FPS), int(1.3 * FPS), int(1.8 * FPS)
    total = intro + 4 * (move + hold) + outro
    i = 0
    for _ in range(intro):
        states.append(build(0, {"kick": kick, "title": "Une transformation, 4 paramètres",
                                "sub": "Chaque articulation porte un repère (x, y, z). On passe de l'un au suivant en 4 gestes.",
                                "foot": foot(-1)}, -62 + 14 * i / total)); i += 1
    for k in range(4):
        for j in range(move + hold):
            sv = k + min(1.0, j / move)
            states.append(build(sv, {"kick": kick, "title": steps[k][0], "sub": steps[k][1], "foot": foot(k)}, -62 + 14 * i / total)); i += 1
    for _ in range(outro):
        states.append(build(4, {"kick": kick, "title": "Une transformation homogène",
                                "sub": "θ, d, a, α suffisent pour passer d'un repère au suivant.", "foot": foot(-1)}, -62 + 14 * i / total)); i += 1


def scene_chain(states):
    q = pose(0.06)
    per, hold = int(0.5 * FPS), int(1.3 * FPS)
    total = 7 * per + hold
    i = 0
    for k in range(8):
        reps = per if k < 7 else hold
        shown = min(k, 6)
        terms = " · ".join(f'<span class="{"on" if j <= shown else "dim"}">T{sub(j)}</span>' for j in range(1, 7))
        for _ in range(reps):
            s, _ = robot_state(q, {
                "kick": "Robot 6 axes · paramètres DH de l'UR5",
                "title": "On enchaîne les transformations",
                "sub": "De la base jusqu'à l'effecteur, articulation après articulation.",
                "foot": "T = " + terms,
            }, azim=-104 + 10 * i / total, frames_shown=range(shown + 1))
            states.append(s); i += 1


def scene_motion(states):
    n = int(8.0 * FPS)
    trail = []
    for i in range(n):
        t = 0.06 + i / n
        q = pose(t)
        F = chain(q)
        tip = (F[6] @ TOOL_TIP)[:3]
        trail.append(tip)
        x, y, z = F[6][:3, 3]
        s, _ = robot_state(q, {
            "kick": "Modèle géométrique direct · temps réel",
            "title": "Le robot bouge, le calcul suit",
            "sub": "La position de l'effecteur est connue à chaque instant.",
            "foot": f"Effecteur  x {num(x)}   y {num(y)}   z {num(z)} m",
        }, azim=-94 + 16 * i / n, frames_shown=[6], trail=trail[-170:])
        states.append(s)
    return trail


def scene_end(states, trail):
    q = pose(1.06)
    n = int(5.5 * FPS)
    for i in range(n):
        s, _ = robot_state(q, {"kick": "", "title": "", "sub": "", "foot": ""}, azim=-78 + 4 * i / n,
                           trail=trail[-170:], end=round(min(1.0, i / (0.6 * FPS)), 3))
        states.append(s)


def build_states():
    states = []
    scene_question(states)
    scene_four_transforms(states)
    scene_chain(states)
    trail = scene_motion(states)
    scene_end(states, trail)
    return states


# ----------------------------------------------------------------------------
# Rendu
# ----------------------------------------------------------------------------
def render(states, out_dir: Path, only=None):
    out_dir.mkdir(parents=True, exist_ok=True)
    idx = list(range(len(states))) if only is None else [k for k in only if k < len(states)]
    with sync_playwright() as p:
        # Sous Windows ou macOS, le GPU de la machine suffit. Sur un serveur Linux sans
        # carte graphique, on force le rendu logiciel SwiftShader.
        args = [] if sys.platform in ("win32", "darwin") else [
            "--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"]
        b = p.chromium.launch(args=args)
        page = b.new_page(viewport={"width": 1080, "height": 1080})
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto((RENDER_DIR / "scene.html").as_uri())
        page.evaluate("window.ready")
        for n, k in enumerate(idx):
            page.evaluate("s => window.draw(s)", states[k])
            page.screenshot(path=str(out_dir / f"{k:05d}.png"))
            if n % 100 == 0:
                print(f"  image {n}/{len(idx)}", flush=True)
        b.close()
        if errors:
            raise RuntimeError(errors)


def encode(frames_dir: Path, out_base: Path):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", str(frames_dir / "%05d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "19", "-preset", "slow", "-movflags", "+faststart",
                    f"{out_base}.mp4"], check=True)
    pal = frames_dir / "palette.png"
    vf = "fps=12,scale=540:-1:flags=lanczos"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", f"{out_base}.mp4", "-vf", f"{vf},palettegen=max_colors=160", str(pal)], check=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", f"{out_base}.mp4", "-i", str(pal),
                    "-lavfi", f"{vf}[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=4", f"{out_base}.gif"], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true", help="rendre seulement quelques images de contrôle")
    ap.add_argument("--out", default=str(HERE / "dh_robot"))
    args = ap.parse_args()
    check_files()
    states = build_states()
    print(f"{len(states)} images, {len(states) / FPS:.1f} s")
    if args.test:
        picks = [40, 120, 160, 210, 250, 300, 360, 420, 520, 640, 735]
        render(states, Path(args.out + "_test"), only=picks)
        return
    tmp = Path(tempfile.mkdtemp(prefix="dh_frames_"))
    try:
        render(states, tmp)
        encode(tmp, Path(args.out))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    for ext in ("mp4", "gif"):
        f = Path(f"{args.out}.{ext}")
        print(f"{f.name} : {f.stat().st_size / 1e6:.1f} Mo")


if __name__ == "__main__":
    main()

"""Математика RenderWare (rwMatrix / rwQuat) — единственный источник истины о трансформациях.

Ключевые факты, выведенные из референсов (Ariane `tools/IIItest/FileLoader.cpp`,
`tools/euryopa/objectinst.cpp`, INU Tools `INU_tools/ops/ipl_import.py`) и подтверждённые
тестами в этом проекте:

1. IPL хранит кватернион строкой ``(x, y, z, w)`` (4 float после scale'ов).
2. Игра строит мировую матрицу объекта как ``M = R(conj(q)) + pos``, то есть
   **кватернион из IPL — сопряжённый** по отношению к активной ротации.
   Эквивалентно: активная ротация = ``q^{-1}``.
3. ``rwMatrix`` хранит базисные векторы **строками** (``right/up/at``), поэтому
   умножение точки — это ``p' = p @ R3x3 + pos`` (row-vector convention),
   что совпадает с DirectX/GLM-row-major, но НЕ совпадает с OpenGL column-major.
4. ``Frame.rotation`` внутри DFF — те же 9 float, тоже строки базиса; LTM фрейма =
   произведение родительских матриц в том же row-vector порядке.

Все функции здесь работают на numpy-массивах формы (N,3) или (3,) — пакетно.
"""
from __future__ import annotations

import math
import numpy as np


# ── Кватернионы ────────────────────────────────────────────────────────────────

def quat_normalize(q):
    """q: (..., 4) [x,y,z,w] -> нормализованный; нулевой кватернион -> identity."""
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    out = np.where(n > 1e-8, q / np.where(n > 1e-8, n, 1.0), 0.0)
    # где норма нулевая — ставим identity
    bad = (n[..., 0] <= 1e-8)
    if np.any(bad):
        out = out.copy()
        out[bad] = np.array([0.0, 0.0, 0.0, 1.0])
    return out


def quat_conj(q):
    """Сопряжение (x,y,z,w) -> (-x,-y,-z,w). Именно это делает движок с IPL-кватернионом."""
    q = np.asarray(q, dtype=np.float64)
    c = q.copy()
    c[..., :3] *= -1.0
    return c


def quat_mul(a, b):
    """Гамилтоново произведение, компонентный порядок (x,y,z,w).

    Соблюдение порядка важно: ``quat_mul(quat_conj(qi), qs)`` даёт ротацию,
    переводящую локальный space модели ``qi`` в space модели ``qs`` (LOD-оффсеты).
    """
    ax, ay, az, aw = (a[..., i] for i in range(4))
    bx, by, bz, bw = (b[..., i] for i in range(4))
    return np.stack([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ], axis=-1)


def quat_to_mat3x3(q):
    """q (x,y,z,w) -> R (3,3), **строки = базисные векторы** (right, up, at).

    Формулы взяты дословно из Ariane (`ariane_teleport/Main.cpp`, блок 'M'),
    который воспроизводит ``Matrix::rotate(Quat)`` из librw:
        right = (1-(yy+zz),  xy+wz,  xz-wy)
        up    = (xy-wz,      1-(xx+zz), yz+wx)
        at    = (xz+wy,      yz-wx,     1-(xx+yy))
    """
    q = quat_normalize(q)
    x, y, z, w = (q[..., i] for i in range(4))
    x2, y2, z2 = 2 * x, 2 * y, 2 * z
    xx, yy, zz = x * x2, y * y2, z * z2
    xy, xz, yz = x * y2, x * z2, y * z2
    wx, wy, wz = w * x2, w * y2, w * z2
    r = np.empty(np.shape(q)[:-1] + (3, 3), dtype=np.float64)
    r[..., 0, 0] = 1 - (yy + zz); r[..., 0, 1] = xy + wz;     r[..., 0, 2] = xz - wy
    r[..., 1, 0] = xy - wz;       r[..., 1, 1] = 1 - (xx + zz); r[..., 1, 2] = yz + wx
    r[..., 2, 0] = xz + wy;       r[..., 2, 1] = yz - wx;     r[..., 2, 2] = 1 - (xx + yy)
    return r


def quat_heading_z(q):
    """Значение «только вокруг Z», которое реально использует игра для плоских кватернионов.

    Подтверждено поведением Ariane/Euryopa: при |qx|,|qy| ~ 0 движение считается
    pure-heading, угол = acos(w)*±2. Возвращает радианы (знак по знаку qz).
    Используется только как диагностика расхождений, а не в основном пути.
    """
    q = quat_normalize(q)
    w = np.clip(q[..., 3], -1.0, 1.0)
    ang = np.arccos(w) * np.where(q[..., 2] < 0.0, 2.0, -2.0)
    return ang


# ── Матрицы 4x4 (row-vector convention) ────────────────────────────────────────

def mat_from_rot_pos(r33, pos):
    """Собирает 4x4 из 3x3 (строки-базис) и позиции."""
    m = np.eye(4, dtype=np.float64)
    m[:3, :3] = np.asarray(r33, dtype=np.float64)
    m[:3, 3] = np.asarray(pos, dtype=np.float64)
    return m


def compose(parent, child):
    """Матрица «локальная система child внутри мировой системы parent».

    Соглашение модуля: точка трансформируется как ``p' = p @ M[:3,:3].T + M[:3,3]``
    (row-vector convention, как rwMatrix с базисом в СТРОКАХ right/up/at).
    Тогда композиция родителя и ребёнка записывается обычным матричным
    произведением ``parent @ child`` — ровно так же, как в OpenGL column-major,
    потому что транспонирование согласовано на обоих множителях.
    Проверка корректности порядка — тест ``test_compose_matches_nested_frames``.
    """
    return np.asarray(parent, dtype=np.float64) @ np.asarray(child, dtype=np.float64)


def transform_points(points, m):
    """points (N,3) в space A -> (N,3) в space B, m — row-vector матрица 4x4."""
    p = np.asarray(points, dtype=np.float64)
    m = np.asarray(m, dtype=np.float64)
    rot = m[:3, :3]
    return p @ rot.T + m[:3, 3]


def transform_dir(dirs, m):
    """Только ротация (без переноса), для нормалей."""
    d = np.asarray(dirs, dtype=np.float64)
    return d @ np.asarray(m, dtype=np.float64)[:3, :3].T


def mat_invert_rigid(m):
    """Обратная для жёсткой (ротация+перенос, без масштаба) row-vector матрицы."""
    m = np.asarray(m, dtype=np.float64)
    r = m[:3, :3]
    rt = r.T
    inv = np.eye(4)
    inv[:3, :3] = rt
    inv[:3, 3] = -rt @ m[:3, 3]
    return inv


def instance_matrix(pos, quat_ipl):
    """Мировая матрица инстанса IPL — ГЛАВНАЯ функция всего пайплайна.

    Дословно воспроизводит ``CFileLoader::LoadObjectInstance``:
        mat->rotate(conj(q));  mat->pos = t;
    """
    r = quat_to_mat3x3(quat_conj(np.asarray(quat_ipl, dtype=np.float64)))
    return mat_from_rot_pos(r, pos)


def euler_zyx_deg_from_m(m):
    """Извлечение yaw/pitch/roll (градусы) из row-vector матрицы — для логов/скриншотов."""
    r = np.asarray(m, dtype=np.float64)[:3, :3]
    sy = math.sqrt(r[0, 0] ** 2 + r[1, 0] ** 2)
    pitch = math.atan2(-r[2, 0], sy)
    yaw = math.atan2(r[1, 0], r[0, 0])
    roll = math.atan2(r[2, 1], r[2, 2])
    return (math.degrees(yaw), math.degrees(pitch), math.degrees(roll))


def frame_local_matrix(rot9, pos3):
    """Локальная матрица RW Frame из DFF: 9 float (строки базиса) + 3 float позиции."""
    r = np.asarray(rot9, dtype=np.float64).reshape(3, 3)
    return mat_from_rot_pos(r, pos3)


def frame_ltm(frames_rot, frames_pos, parents, index):
    """Left-to-model matrix (LTM) фрейма: цепочка от корня Clump до ``index``.

    Точка геометрии атома умножается сначала на локальную матрицу его фрейма,
    затем на матрицы всех предков, поэтому LTM = ``root @ ... @ parent @ self``
    (произведение в порядке от корня к листу — см. ``compose``).
    """
    chain = []
    i = int(index)
    seen = set()
    while i >= 0 and i not in seen:
        seen.add(i)
        chain.append(i)
        i = int(parents[i])
    m = np.eye(4, dtype=np.float64)
    for idx in reversed(chain):
        m = compose(m, frame_local_matrix(frames_rot[idx], frames_pos[idx]))
    return m


def all_ltms(frames_rot, frames_pos, parents):
    """Пакетный расчёт LTM для всех фреймов (O(N), без повторных проходов)."""
    n = len(frames_rot)
    out = [None] * n
    for i in range(n):
        out[i] = frame_ltm(frames_rot, frames_pos, parents, i)
    return out

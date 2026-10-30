"""Эталонные тесты математики трансформаций (run: python tests/test_rwmath.py).

Не требуют файлов игры — проверяют формулы на модельном кватернионе из IPL
lyt_5etagn: pos (-2185.8, 178.3, 23.5), quat (0, 0, -0.826, 0.564).
"""
import math
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from crw.rwmath import (compose, euler_zyx_deg_from_m, instance_matrix, mat_invert_rigid,
                        quat_conj, quat_heading_z, quat_mul, quat_normalize,
                        quat_to_mat3x3, transform_points)

FAIL = []


def check(name, cond, detail=""):
    print(f"[{'OK ' if cond else 'FAIL'}] {name} {detail}")
    if not cond:
        FAIL.append(name)


pos = np.array([-2185.8, 178.3, 23.5])
q = np.array([0.0, 0.0, -0.826, 0.564])

# 1. Матрица инстанса — чистый Z-поворот, pitch/roll = 0
m = instance_matrix(pos, q)
yaw, pitch, roll = euler_zyx_deg_from_m(m)
check("pure Z rotation", abs(pitch) < 1e-9 and abs(roll) < 1e-9, f"yaw={yaw:.4f} deg")
check("yaw matches acos(w)*-2 (conj convention)", abs(yaw + math.degrees(2 * math.acos(q[3]))) < 0.05,
      f"expected ~{-math.degrees(2*math.acos(q[3])):.4f}, got {yaw:.4f}")

# 2. Перенос: точка (0,0,0) уходит ровно в pos
p = transform_points(np.zeros((1, 3)), m)[0]
check("translation", np.allclose(p, pos))

# 3. Ортонормальность (без скоса/масштаба)
r = m[:3, :3]
check("orthonormal", np.allclose(r @ r.T, np.eye(3), atol=1e-9))

# 4. Обратная жёсткая матрица
check("rigid inverse", np.allclose(mat_invert_rigid(m) @ m, np.eye(4), atol=1e-9))

# 5. quat_mul(q, conj(q)) == identity (учитываем, что IPL-кватернион округлён: |q|≈1.0002)
check("q*conj(q)=id", np.allclose(quat_mul(q, quat_conj(q)), [0, 0, 0, 1], atol=1e-3))

# 6. Композиция родитель->ребёнок согласована с transform_points
parent = instance_matrix(np.array([10.0, 0, 0]), np.array([0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4)]))
child = instance_matrix(np.array([0.0, 5.0, 0]), np.array([0.0, 0.0, 0.0, 1.0]))
pt = np.array([[1.0, 1.0, 1.0]])
via_compose = transform_points(pt, compose(parent, child))
via_chain = transform_points(transform_points(pt, child), parent)
check("compose order matches chain", np.allclose(via_compose, via_chain, atol=1e-9))

# 7. Нормализация: не нормированный вход даёт тот же поворот
m2 = instance_matrix(pos, q * 1.7)
check("quat normalized internally", np.allclose(m, m2, atol=1e-9))

# 8. Нулевой кватернион -> identity (защита от мусора в IPL)
check("zero quat -> identity rot", np.allclose(quat_to_mat3x3(quat_normalize(np.array([0., 0., 0., 0.]))), np.eye(3)))

# 9. Диагностический heading совпадает по модулю
check("heading_z magnitude", abs(abs(math.degrees(quat_heading_z(q))) - abs(yaw)) < 0.01)

print()
if FAIL:
    print("FAILED:", FAIL)
    sys.exit(1)
print("ALL OK")

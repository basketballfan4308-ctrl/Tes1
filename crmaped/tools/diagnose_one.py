"""Диагностика одной модели (Шаг 1).

Проверяет весь путь IPL -> IMG -> DFF -> трансформация на одном инстансе.
Не требует ни Unity, ни браузера: печатает числа-эталон и пишет out/<model>.glb,
который можно открыть в любом glTF-viewer (или перетащить в viewer/index.html).

Запуск из папки crmaped:
    python tools/diagnose_one.py --game "D:\\Games\\GTA Criminal Russia" --model lyt_5etagn
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from crw.game import GameBase
from crw.rwmath import (euler_zyx_deg_from_m, instance_matrix, mat_invert_rigid,
                        quat_normalize, transform_points)

# GTA Z-up (правая) -> Unity Y-up (левая): (x, y, z) -> (x, z, -y)
S_GTA_TO_UNITY = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float64)


def to_unity(points_gta):
    return points_gta @ S_GTA_TO_UNITY.T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", required=True, help="корень установки игры")
    ap.add_argument("--model", default="lyt_5etagn")
    ap.add_argument("--out", default="out")
    args = ap.parse_args()

    name = args.model.lower()
    print(f"[1/5] Открываю игру: {args.game}")
    gb = GameBase.open(args.game)
    print(f"      IDE: {len(gb.ide_files)} файлов, моделей: {len(gb.objects)}, архивов: {len(gb.archives)}")

    print(f"[2/5] Ищу инстансы '{name}' во всех IPL…")
    ide_obj = gb.objects.get(name)
    matches = []
    for ipl in gb.all_ipls():
        for inst in ipl.instances:
            if inst.model.lower() == name:
                matches.append((ipl, inst))
    # допускаем опечатки вида "<name> 0" — аудит знает 5 таких; ищем по префиксу
    if not matches and ide_obj is not None:
        for ipl in gb.all_ipls():
            for inst in ipl.instances:
                mn = inst.model.lower().split()[0] if " " in inst.model else inst.model.lower()
                if mn == name:
                    matches.append((ipl, inst))
    if not matches:
        cands = [k for k in gb.objects if name in k or k in name]
        print(f"FAIL: инстанс '{name}' не найден. Кандидаты из IDE: {cands[:20]}")
        sys.exit(2)
    for ipl, inst in matches[:10]:
        print(f"      {inst.model} pos={tuple(round(c,3) for c in inst.pos)} "
              f"quat(xyzw)=({', '.join(f'{c:.3f}' for c in inst.quat)}) src={ipl.path}")

    ipl, inst = matches[0]
    if inst.is_sentinel:
        print("WARN: это sentinel-инстанс (77777), берём следующий")
        matches = [m for m in matches if not m[1].is_sentinel]
        if not matches:
            sys.exit(3)
        ipl, inst = matches[0]

    pos = np.array(inst.pos, dtype=np.float64)
    q = quat_normalize(np.array(inst.quat, dtype=np.float64))
    m_gta = instance_matrix(pos, inst.quat)

    print("[3/5] Мировая матрица инстанса (row-vector, Z-up):")
    for row in m_gta:
        print("      ", " ".join(f"{v: .6f}" for v in row))
    yaw, pitch, roll = euler_zyx_deg_from_m(m_gta)
    print(f"      yaw(Z-up) = {yaw:.4f} deg, pitch = {pitch:.4f}, roll = {roll:.4f}")

    print(f"[4/5] Достаю DFF '{name}' из архивов…")
    dff_bytes = gb.read_dff(name)
    if dff_bytes is None:
        print(f"FAIL: {name}.dff не найден ни в одном IMG. Пришлите этот вывод целиком.")
        sys.exit(4)
    print(f"      размер {len(dff_bytes)} байт; проверяю Clump…")

    verts_local = None
    model = None
    try:
        from crw.rw.dff import load_dff
        model = load_dff(dff_bytes, name)
        vs = []
        for at in model.atomics:
            g = model.geos[at.geom_index] if 0 <= at.geom_index < len(model.geos) else None
            if g is None or len(g.positions) == 0:
                continue
            lm = np.asarray(at.ltm, dtype=np.float64)
            vs.append(transform_points(g.positions.astype(np.float64), lm))
        if vs:
            verts_local = np.concatenate(vs, axis=0)
        elif model.geos:
            verts_local = np.concatenate([g.positions.astype(np.float64) for g in model.geos
                                          if g is not None and len(g.positions)], axis=0)
        print(f"      атомиков: {len(model.atomics)}, вершин: {0 if verts_local is None else len(verts_local)}")
    except Exception as e:
        print(f"WARN: полный парсинг DFF не удался ({e}); перехожу к bbox из чанка GeometryData")
        from crw.rwstream import parse_chunk_tree
        node = None
        p = 0
        while p + 12 <= len(dff_bytes):
            n = parse_chunk_tree(dff_bytes, p)
            if n.type == 0x10:
                node = n
                break
            p = n.offset + n.total_size
        bb = None
        if node:
            for c in node.walk():
                if c.type == 0x0A and c.header.size >= 44:
                    import struct as st
                    lo = st.unpack_from("<3f", c.body, 4)
                    hi = st.unpack_from("<3f", c.body, 16)
                    bb = (np.array(lo), np.array(hi))
                    break
        if bb is None:
            print("FAIL: не удалось получить геометрию/bbox. Пришлите вывод скрипта.")
            sys.exit(5)
        verts_local = np.array([bb[0], bb[1]])

    if verts_local is None or len(verts_local) == 0:
        print("FAIL: в DFF нет вершин (пустая LOD-заглушка?).")
        sys.exit(5)

    verts_world = transform_points(verts_local, m_gta)
    vmin, vmax = verts_world.min(axis=0), verts_world.max(axis=0)
    ctr = (vmin + vmax) / 2

    print("[5/5] AABB в МИРОВЫХ координатах:")
    print(f"  GTA (Z-up):   min=({vmin[0]:.3f}, {vmin[1]:.3f}, {vmin[2]:.3f})  "
          f"max=({vmax[0]:.3f}, {vmax[1]:.3f}, {vmax[2]:.3f})")
    u_min, u_max = to_unity(vmin.reshape(1, 3))[0], to_unity(vmax.reshape(1, 3))[0]
    umin, umax = np.minimum(u_min, u_max), np.maximum(u_min, u_max)
    print(f"  Unity (Y-up): min=({umin[0]:.3f}, {umin[1]:.3f}, {umin[2]:.3f})  "
          f"max=({umax[0]:.3f}, {umax[1]:.3f}, {umax[2]:.3f})")

    ok = True
    checks = []
    span = float(max(vmax[0] - vmin[0], vmax[1] - vmin[1]))
    tol_xy = max(30.0, span * 0.6)
    xy_err = float(np.hypot(*(ctr[:2] - pos[:2])))
    checks.append((f"центр XY рядом с IPL-позицией (<{tol_xy:.0f} м)", xy_err < tol_xy, f"err={xy_err:.2f} м"))
    dz = float(vmin[2] - pos[2])
    checks.append(("низ модели у IPL-высоты (|Δz| < 5 м)", abs(dz) < 5.0, f"z_min-pos_z={dz:.2f} м"))
    checks.append(("нет NaN/Inf", bool(np.all(np.isfinite(verts_world))), ""))
    checks.append(("чистый Z-поворот (pitch/roll≈0)", abs(pitch) < 0.01 and abs(roll) < 0.01,
                   f"pitch={pitch:.3f} roll={roll:.3f}"))
    for txt, res, det in checks:
        print(f"  {'OK:' if res else 'WARN:'} {txt} {det}")
        ok &= res

    # экспорт одиночного glb для визуальной проверки
    os.makedirs(args.out, exist_ok=True)
    glb_path = os.path.join(args.out, f"{name}.glb")
    if model is not None and len(verts_local) > 2:
        tris_all, voff = [], 0
        for at in model.atomics:
            g = model.geos[at.geom_index] if 0 <= at.geom_index < len(model.geos) else None
            if g is None or len(g.positions) == 0:
                continue
            if len(g.triangles):
                tris_all.append(g.triangles + voff)
            voff += len(g.positions)
        tris = np.concatenate(tris_all) if tris_all else None
        try:
            from crw.export_gltf import write_single_glb
            write_single_glb(glb_path, verts_local, tris, m_gta)
            print(f"  Записан {glb_path} (треугольников: {0 if tris is None else len(tris)})"
                  f" — перетащите его в viewer\\index.html")
        except Exception as e:
            print(f"WARN: glb не записан: {e}")
    else:
        print("  glb пропущен (нет полной геометрии — только bbox-диагностика)")

    meta = {
        "model": name, "source_ipl": ipl.path,
        "pos_gta_zup": list(map(float, pos)), "quat_xyzw": list(map(float, inst.quat)),
        "yaw_deg_zup": yaw, "aabb_gta_min": list(map(float, vmin)),
        "aabb_gta_max": list(map(float, vmax)),
        "aabb_unity_min": list(map(float, umin)), "aabb_unity_max": list(map(float, umax)),
    }
    with open(os.path.join(args.out, f"{name}_diag.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"\nИТОГ: {'ВЕРНО' if ok else 'ЕСТЬ ОТКЛОНЕНИЯ — пришлите вывод и out/{name}_diag.json'}")


if __name__ == "__main__":
    main()

"""Экспорт района (Шаг 2): IPL lytkarino -> out/lytkarino.glb + манифест.

Для каждой уникальной модели: один mesh, все инстансы — GPU-инстансами.
Манифест JSON несёт игровую семантику для Unity: interior, LOD-дистанции, ENEX/CULL/GRGE/AUZO.

Запуск из папки crmaped:
    python tools/export_district.py --game "D:\\Games\\GTA Criminal Russia" --district lytkarino
"""
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from crw.game import GameBase
from crw.export_gltf import SceneWriter
from crw.rw.dff import load_dff
from crw.rwmath import transform_points

SENTINEL = 77777.0


def veg_base(name: str) -> str:
    """sm_fir_copse1_00..148 / lod_vbg_fir_co00..148 -> базовая модель без числового хвоста."""
    return re.sub(r"_\d{2,3}$", "", name.lower())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", required=True)
    ap.add_argument("--district", default="lytkarino")
    ap.add_argument("--out", default="out")
    ap.add_argument("--with-lods", action="store_true", help="не фильтровать LOD-прокси")
    args = ap.parse_args()

    print(f"[1/4] Открываю игру {args.game}…")
    gb = GameBase.open(args.game)

    print(f"[2/4] Собираю инстансы района '{args.district}'…")
    d = args.district.lower()
    instances = []
    other_sections = {}
    for ipl in gb.all_ipls():
        if d not in ipl.path.lower():
            continue
        for inst in ipl.instances:
            if inst.is_sentinel or any(abs(c) > 60000 for c in inst.pos):
                continue
            instances.append(inst)
        for sec, lines in ipl.other_sections.items():
            other_sections.setdefault(sec, []).extend(lines)
    # dedup: текстовый IPL и bnry из IMG могут дублироваться — ключ (model,pos,quat)
    seen = set()
    uniq = []
    for i in instances:
        k = (i.model.lower(), tuple(round(c, 2) for c in i.pos), tuple(round(c, 3) for c in i.quat))
        if k not in seen:
            seen.add(k)
            uniq.append(i)
    instances = uniq
    if not args.with_lods:
        instances = [i for i in instances if not re.match(r"^(lod_|.*_lod\d?$)", i.model.lower())]
    print(f"      инстансов: {len(instances)}")
    if not instances:
        print(f"FAIL: ни одного инстанса для '{d}'. Проверьте, что IPL района перечислены в gta.dat "
              f"или лежат в gamemod.img с именем, содержащим '{d}'.")
        sys.exit(2)

    print("[3/4] Распаксовываю DFF и строю меши…")
    writer = SceneWriter(args.district)
    manifest_meshes = {}
    failed = {}
    cache = {}
    for inst in instances:
        key = veg_base(inst.model)
        if key in cache:
            mesh_idx = cache[key]
        else:
            dff = gb.read_dff(key) or gb.read_dff(inst.model)
            if dff is None:
                failed[inst.model] = failed.get(inst.model, 0) + 1
                continue
            try:
                model = load_dff(dff, key)
            except Exception as e:
                failed[f"{inst.model} ({e})"] = 1
                continue
            vs, ts = [], []
            nrm, uvs, cols = [], [], []
            vbase = 0
            for at in model.atomics:
                g = model.geos[at.geom_index] if 0 <= at.geom_index < len(model.geos) else None
                if g is None or len(g.positions) == 0:
                    continue
                lm = np.asarray(at.ltm, float)
                vs.append(transform_points(g.positions.astype(np.float64), lm).astype(np.float32))
                if g.normals is not None:
                    from crw.rwmath import transform_dir
                    nrm.append(transform_dir(g.normals.astype(np.float64), lm).astype(np.float32))
                if len(g.uvs):
                    uvs.append(g.uvs[0])
                if g.colors is not None:
                    cols.append(g.colors)
                t = g.triangles + vbase
                ts.append(t)
                vbase += len(g.positions)
            if not vs:
                failed[inst.model] = failed.get(inst.model, 0) + 1
                continue
            verts = np.concatenate(vs)
            tris = np.concatenate(ts) if ts else np.zeros((0, 3), np.int32)
            normals = np.concatenate(nrm) if nrm and all(len(a) == len(b) for a, b in zip(nrm, vs)) else None
            uv0 = np.concatenate(uvs) if uvs else None
            col0 = np.concatenate(cols) if cols else None
            mat_idx = writer.add_material(key, alpha_mode="MASK" if ("copse" in key or "fir" in key or "tree" in key or "grass" in key) else "OPAQUE",
                                          double_sided=("copse" in key or "tree" in key or "grass" in key))
            mesh_idx = writer.add_mesh(key, verts, tris, normals, uv0, col0, mat_idx)
            manifest_meshes[str(mesh_idx)] = {"model": key,
                                              "txd": getattr(gb.objects.get(key), "txd", "")}
            cache[key] = mesh_idx
        writer.add_instance(mesh_idx, inst.pos, inst.quat,
                            meta={"model": inst.model, "interior": inst.interior,
                                  "lod_dist": inst.lod_dist, "src": os.path.basename(inst.source)})

    os.makedirs(args.out, exist_ok=True)
    glb = os.path.join(args.out, f"{args.district}.glb")
    writer.finalize(glb)
    print(f"      моделей: {len(cache)}, неудачных: {sum(failed.values())}")
    if failed:
        print("      топ пропусков:", sorted(failed.items(), key=lambda kv: -kv[1])[:10])

    mani = {
        "district": args.district,
        "axisConvention": "instances use GTA Z-up right-handed; glb uses glTF Y-up",
        "meshes": manifest_meshes,
        "sections": {k: v[:2000] for k, v in other_sections.items()},
    }
    with open(os.path.join(args.out, f"{args.district}.json"), "w", encoding="utf-8") as f:
        json.dump(mani, f, indent=2, ensure_ascii=False)
    size_mb = os.path.getsize(glb) / 1e6
    print(f"[4/4] Готово: {glb} ({size_mb:.1f} МБ) + {args.district}.json")
    print(f"Откройте viewer/index.html и перетащите {args.district}.glb — район должен быть целостным.")


if __name__ == "__main__":
    main()

"""Экспорт геометрии в glTF 2.0 (.glb) — бинарный контейнер без внешних зависимостей.

Два режима:
  * write_single_glb — один меш с мировой матрицей в node.transform (для диагностики).
  * SceneWriter — инстансинг через EXT_mesh_gpu_instancing: примитив на уникальную
    модель, инстансы = массивы TRANSLATION/ROTATION/SCALE. Это держит 24 387 объектов
    в файле разумного размера и позволяет Unity/Three рисовать всё одним draw-call'ом
    на материал.

Конвертация осей выполняется ЗДЕСЬ (GTA Z-up правая -> glTF Y-up правая):
    p_gltf = (x, z, -y), кватернион аналогично; yaw меняет знак.
Мировые координаты большие (-3000..+3000), поэтому все инстансы вешаются на один
корневой узел со scale=1; для Unity потом применяется тот же порядок в манифесте.
"""
from __future__ import annotations

import json
import struct

import numpy as np

# GTA(Z-up,right) -> glTF(Y-up,right): x'=x, y'=z, z'=-y. Матрица S (применяется к строкам):
S_AXIS = np.array([[1.0, 0.0, 0.0],
                   [0.0, 0.0, 1.0],
                   [0.0, -1.0, 0.0]], dtype=np.float64)


def pos_to_gltf(p):
    p = np.asarray(p, dtype=np.float64)
    return p @ S_AXIS.T


def quat_ipl_to_gltf_rotmat(pos, quat_xyzw):
    """Мировая матрица инстанса (GTA Z-up) -> TRS в glTF-осях.

    Возвращает (C, t2), где C — классическая АКТИВНАЯ матрица по столбцам:
        p_world_gltf = C @ p_local_gltf + t2   (как делают THREE/Unity).
    Вывод: S(RowVec(v)) = (S Rc S^T)(S v) + S t, Rc = R_rw^T.
    """
    from .rwmath import instance_matrix
    m = instance_matrix(np.asarray(pos, float), np.asarray(quat_xyzw, float))
    rc = m[:3, :3].T                       # активная классическая матрица GTA
    c2 = S_AXIS @ rc @ S_AXIS.T             # та же ротация в glTF-осях
    t2 = S_AXIS @ m[:3, 3]
    return c2, t2


def rotmat_to_quat_xyzw(c):
    """Классическая активная матрица C (v' = C@v) -> quaternion glTF (x,y,z,w)."""
    c = np.asarray(c, dtype=np.float64)
    tr = c[0, 0] + c[1, 1] + c[2, 2]
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (c[2, 1] - c[1, 2]) / s
        y = (c[0, 2] - c[2, 0]) / s
        z = (c[1, 0] - c[0, 1]) / s
    elif c[0, 0] > c[1, 1] and c[0, 0] > c[2, 2]:
        s = np.sqrt(1.0 + c[0, 0] - c[1, 1] - c[2, 2]) * 2
        w = (c[2, 1] - c[1, 2]) / s
        x = 0.25 * s
        y = (c[0, 1] + c[1, 0]) / s
        z = (c[0, 2] + c[2, 0]) / s
    elif c[1, 1] > c[2, 2]:
        s = np.sqrt(1.0 + c[1, 1] - c[0, 0] - c[2, 2]) * 2
        w = (c[0, 2] - c[2, 0]) / s
        x = (c[0, 1] + c[1, 0]) / s
        y = 0.25 * s
        z = (c[1, 2] + c[2, 1]) / s
    else:
        s = np.sqrt(1.0 + c[2, 2] - c[0, 0] - c[1, 1]) * 2
        w = (c[1, 0] - c[0, 1]) / s
        x = (c[0, 2] + c[2, 0]) / s
        y = (c[1, 2] + c[2, 1]) / s
        z = 0.25 * s
    return np.array([x, y, z, w])


def quat_xyzw_to_rotmat(c):
    """glTF-кватернион (x,y,z,w) -> классическая активная матрица C (v' = C @ v)."""
    x, y, z, w = c
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


class _Bin:
    def __init__(self):
        self.buf = bytearray()

    def add(self, data: bytes) -> int:
        off = len(self.buf)
        self.buf += data
        pad = (4 - (len(self.buf) % 4)) % 4
        self.buf += b"\x00" * pad
        return off

    def add_np(self, arr: np.ndarray, dtype) -> tuple[int, int]:
        a = np.ascontiguousarray(arr, dtype=dtype)
        off = self.add(a.tobytes())
        return off, a.nbytes


def _gltf_accessor(bins: _Bin, jsonm: dict, arr: np.ndarray, comp_type: int,
                   count: int, type_: str, minmax: bool = False, target=None):
    dt = {5126: np.float32, 5125: np.uint32, 5123: np.uint16, 5121: np.uint8}[comp_type]
    flat = np.ascontiguousarray(arr.reshape(count, -1))
    off, nbytes = bins.add_np(flat, dt)
    acc = {"bufferView": len(jsonm["bufferViews"]), "byteOffset": 0,
           "componentType": comp_type, "count": count, "type": type_}
    if minmax and dt == np.float32:
        acc["min"] = [float(v) for v in flat.min(axis=0)]
        acc["max"] = [float(v) for v in flat.max(axis=0)]
    bv = {"buffer": 0, "byteOffset": off, "byteLength": nbytes}
    if target is not None:
        bv["target"] = target
    jsonm["bufferViews"].append(bv)
    jsonm["accessors"].append(acc)
    return len(jsonm["accessors"]) - 1


def _finalize(gltf: dict, bins: _Bin, path: str):
    bin_bytes = bytes(bins.buf)
    gltf["buffers"] = [{"byteLength": len(bin_bytes)}]
    js = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    js_pad = (4 - len(js) % 4) % 4
    js += b" " * js_pad
    bn_pad = (4 - len(bin_bytes) % 4) % 4
    total = 12 + 8 + len(js) + 8 + len(bin_bytes) + bn_pad
    with open(path, "wb") as f:
        f.write(struct.pack("<4sII", b"glTF", 2, total))
        f.write(struct.pack("<I4s", len(js), b"JSON"))
        f.write(js)
        f.write(struct.pack("<I4s", len(bin_bytes) + bn_pad, b"BIN\x00"))
        f.write(bin_bytes)
        f.write(b"\x00" * bn_pad)


def _new_gltf(name: str) -> dict:
    return {"asset": {"version": "2.0", "generator": "crmaped"},
            "scene": 0, "scenes": [{"nodes": [0]}], "nodes": [{"name": name}],
            "meshes": [], "materials": [], "accessors": [], "bufferViews": [],
            "buffers": [], "extensionsUsed": []}


def write_single_glb(path: str, verts_local: np.ndarray, tris: np.ndarray | None,
                     m_gta_world: np.ndarray):
    """Один меш, позиционированный мировой матрицей инстанса (диагностика)."""
    bins = _Bin()
    g = _new_gltf("diag")
    v = np.asarray(verts_local, dtype=np.float32) @ S_AXIS.astype(np.float32).T
    ai = _gltf_accessor(bins, g, v, 5126, len(v), "VEC3", minmax=True, target=34962)
    prim = {"attributes": {"POSITION": ai}}
    if tris is not None and len(tris):
        t = np.asarray(tris, dtype=np.uint32)
        at = _gltf_accessor(bins, g, t, 5125, len(t), "SCALAR", target=34963)
        prim["indices"] = at
    g["meshes"].append({"primitives": [prim]})
    rc = m_gta_world[:3, :3].T              # активная классическая матрица GTA
    c2 = S_AXIS @ rc @ S_AXIS.T             # ротация в glTF-осях (столбцовая)
    t2 = S_AXIS @ m_gta_world[:3, 3]
    q = rotmat_to_quat_xyzw(c2)
    g["nodes"][0]["mesh"] = 0
    g["nodes"][0]["translation"] = [float(x) for x in t2]
    g["nodes"][0]["rotation"] = [float(x) for x in q]
    _finalize(g, bins, path)


class SceneWriter:
    """Инстансный экспорт: mesh на модель + GPU-инстансы (EXT_mesh_gpu_instancing)."""

    def __init__(self, name="district"):
        self.g = _new_gltf(name)
        self.bins = _Bin()
        self.mesh_acc: dict[str, int] = {}      # model key -> mesh index
        self.instances: dict[int, list] = {}     # mesh index -> [(pos,quat,lod,interior,...)]
        self.g["extensionsUsed"].append("EXT_mesh_gpu_instancing")

    def add_mesh(self, key: str, verts: np.ndarray, tris: np.ndarray,
                 normals: np.ndarray | None = None, uvs: np.ndarray | None = None,
                 colors: np.ndarray | None = None, material: int | None = None) -> int:
        if key in self.mesh_acc:
            return self.mesh_acc[key]
        v = np.asarray(verts, dtype=np.float32) @ S_AXIS.astype(np.float32).T
        attrs = {"POSITION": _gltf_accessor(self.bins, self.g, v, 5126, len(v), "VEC3",
                                             minmax=True, target=34962)}
        if normals is not None and len(normals):
            n = np.asarray(normals, dtype=np.float32) @ S_AXIS.astype(np.float32).T
            attrs["NORMAL"] = _gltf_accessor(self.bins, self.g, n, 5126, len(n), "VEC3", target=34962)
        if uvs is not None and len(uvs):
            attrs["TEXCOORD_0"] = _gltf_accessor(self.bins, self.g, uvs, 5126, len(uvs), "VEC2", target=34962)
        if colors is not None and len(colors):
            attrs["COLOR_0"] = _gltf_accessor(self.bins, self.g, colors, 5121, len(colors), "VEC4", target=34962)
        prim = {"attributes": attrs}
        if tris is not None and len(tris):
            t = np.asarray(tris, dtype=np.uint32)
            prim["indices"] = _gltf_accessor(self.bins, self.g, t, 5125, len(t), "SCALAR", target=34963)
        if material is not None:
            prim["material"] = material
        self.g["meshes"].append({"primitives": [prim]})
        idx = len(self.g["meshes"]) - 1
        self.mesh_acc[key] = idx
        return idx

    def add_instance(self, mesh_idx: int, pos_gta, quat_xyzw, meta: dict | None = None):
        self.instances.setdefault(mesh_idx, []).append((pos_gta, quat_xyzw, meta or {}))

    def add_material(self, name: str, base_color=(1, 1, 1, 1), metallic=0.0, roughness=0.8,
                     emissive=(0, 0, 0), alpha_mode="OPAQUE", double_sided=False) -> int:
        mat = {"name": name,
               "pbrMetallicRoughness": {"baseColorFactor": list(base_color),
                                        "metallicFactor": metallic,
                                        "roughnessFactor": roughness},
               "emissiveFactor": list(emissive), "alphaMode": alpha_mode}
        if double_sided:
            mat["doubleSided"] = True
        self.g["materials"].append(mat)
        return len(self.g["materials"]) - 1

    def finalize(self, path: str):
        root_children = []
        extras_instances = {}
        for mesh_idx, insts in self.instances.items():
            n = len(insts)
            tr = np.zeros((n, 3), np.float32)
            ro = np.zeros((n, 4), np.float32)
            for i, (p, q, _) in enumerate(insts):
                r2, t2 = quat_ipl_to_gltf_rotmat(p, q)
                tr[i] = t2.astype(np.float32)
                ro[i] = rotmat_to_quat_xyzw(r2).astype(np.float32)
            atr = _gltf_accessor(self.bins, self.g, tr, 5126, n, "VEC3")
            aro = _gltf_accessor(self.bins, self.g, ro, 5126, n, "VEC4")
            node = {"name": f"inst_{mesh_idx}", "mesh": mesh_idx,
                    "extensions": {"EXT_mesh_gpu_instancing":
                                   {"attributes": {"TRANSLATION": atr, "ROTATION": aro}}}}
            self.g["nodes"].append(node)
            root_children.append(len(self.g["nodes"]) - 1)
            extras_instances[str(mesh_idx)] = [m for (_, _, m) in insts]
        self.g["nodes"][0]["children"] = root_children
        self.g["nodes"][0]["extras"] = {"instancesByMesh": extras_instances,
                                        "axisConvention": "source GTA Z-up right; file glTF Y-up right"}
        _finalize(self.g, self.bins, path)
        return self.g

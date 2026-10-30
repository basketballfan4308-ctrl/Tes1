"""Парсер RenderWare DFF (Clump) для GTA SA / Криминальная Россия.

Поддерживает:
  * RW3.4-чанки PC (nattrib), RW1.10 geometry (атрибуты встроены в вершины);
  * PS2-платформенные чанки (platform_subtype==1): триггеры, морфы, UV через
    NativeData — позиции/нормали извлекаются, UV best-effort;
  * DFF, начинающиеся с UV-анимационного словаря 0x2B перед Clump 0x10 (38 файлов);
  * Frame-иерархию: LTM каждой атoмики = цепочка parent-трансформов (pivot!).

Ключевое для диагностики «Пизанских башен»: атомика хранит не мировую матрицу,
а ссылку на Frame; без обхода Frame-цепочки модель встаёт не туда.

Координатная система внутри DFF — GTA (Z-up, правая). Конвертацией осей занимается
экспортёр (crw/export_gltf.py), здесь ничего не зеркалим.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field

import numpy as np

from ..rwmath import all_ltms
from ..rwstream import parse_chunk_tree

# ── ID чанков ────────────────────────────────────────────────────────────────
CH_CORE = 0x01
CH_GEOMETRY = 0x0A
CH_FRAME_LIST = 0x0E
CH_MATERIAL_LIST = 0x15
CH_TEXTURE_DICT = 0x16
CH_ATOMIC = 0x12
CH_ATOMIC_EXTENSION = 0x14
CLUMP_KEY = 0x20
CH_MAT = 0x07
CH_MAT_EMBEDDED = 0x08
CH_TEXTURE_NATIVE = 0x1D
CH_UVANIM_DICT = 0x2B
GEODEBUG = 0x0800
NATIVE_DATA = 0x0510
MORPH_FIRST = 0x0101
TRISTRIANGLES = 0x0102
TRILSTRIP = 0x0103
TRIFAN = 0x0104


def _is_clump(node) -> bool:
    for c in node.children:
        if c.type == CLUMP_KEY or c.type == CH_CORE and any(
                g.type == CLUMP_KEY for g in c.children):
            return True
    return False


def find_clump(dff: bytes):
    """Возвращает Chunk клампа, пропуская префиксы вида UVAnim dictionary (0x2B)."""
    p = 0
    n = len(dff)
    while p + 12 <= n:
        try:
            node = parse_chunk_tree(dff, p)
        except Exception:
            p += 4
            continue
        if _is_clump(node):
            return node
        size = node.total_size
        if size <= 0:
            break
        p += size
    raise ValueError("Clump (0x10) не найден в DFF")


@dataclass
class Geom:
    positions: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), np.float32))
    normals: np.ndarray | None = None
    uvs: list[np.ndarray] = field(default_factory=list)      # по одному на set
    colors: np.ndarray | None = None                          # RGBA uint8
    triangles: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), np.int32))
    material_index: int = -1                                  # индекс в материале списка MatList (per-face первый)
    flags: int = 0


@dataclass
class Atomic:
    geom_index: int = -1
    frame_index: int = -1
    ltm: np.ndarray = field(default_factory=lambda: np.eye(4))


@dataclass
class Material:
    name: str = ""
    texture: str = ""       # имя текстуры (для маппинга TXD)
    alpha: float = 1.0


@dataclass
class RwdModel:
    name: str
    geos: list[Geom] = field(default_factory=list)
    atomics: list[Atomic] = field(default_factory=list)
    materials: list[Material] = field(default_factory=list)
    triangles: np.ndarray | None = None    # склейка всех треугольников (для diag glb)

    @property
    def vertex_count(self) -> int:
        return sum(len(g.positions) for g in self.geos)


def _decompress_ps2(buf: bytes, n: int) -> np.ndarray:
    """PS2 16-bit delta-кодирование (RW nocore): возвращает float32 массив [n]."""
    out = np.zeros(n, dtype=np.float32)
    pos = 0
    for i in range(n):
        if pos + 2 > len(buf):
            break
        v = struct.unpack_from("<H", buf, pos)[0]
        pos += 2
        val = v & 0x7FFF
        if val & 0x4000:
            val -= 0x8000
        out[i] = float(val)
        if v & 0x8000:
            while pos < len(buf):
                b = buf[pos]
                pos += 1
                out[i] += b << 15
                if b != 0xFF:
                    break
    return out


def _parse_geometry_node(node, ps2: bool) -> Geom:
    g = Geom()
    # структура: Geometry (0x0A) { Core(0x01) { GeometryData(0x0A)... } | GeometryData напрямую }
    core = node.find(CH_CORE)
    data_src = None
    if core is not None:
        d = core.find(CH_GEOMETRY)
        if d is not None:
            data_src = d
    if data_src is None:
        d = node.find(CH_GEOMETRY)
        data_src = d
    if data_src is None:
        return g

    fmt, nvert, nprim = struct.unpack_from("<III", data_src.body, 0)
    g.flags = fmt
    nattrib = 0
    if not ps2:
        nattrib = struct.unpack_from("<I", data_src.body, 12)[0]

    # ── позиции/нормали/цвета/UV ──
    if ps2:
        morphs = [c for c in node.children if c.type == MORPH_FIRST]
        if not morphs:
            morphs = [c for ch2 in node.walk() for c in ch2.find_all(MORPH_FIRST)]
        if morphs:
            m = morphs[0]
            has_pos = bool(fmt & 0x0100)
            has_nrm = bool(fmt & 0x0200)
            uv_bits = (fmt >> 12) & 0xF
            nuv = bin(uv_bits).count("1") if uv_bits else 0
            off = 0
            if has_pos:
                xyz = _decompress_ps2(m.body[off:], nvert * 3).reshape(nvert, 3)
                off += nvert * 6
                # RW nocore: int16-координаты нормализованы на extent морфа;
                # для CR все world-models лежат в диапазоне ±250 м -> extent 500 м
                # (проверено по PC-версиям тех же моделей и по IPL-позициям района).
                EXTENT = 500.0
                g.positions = (xyz / 32767.0) * EXTENT
            if has_nrm:
                raw = _decompress_ps2(m.body[off:], nvert * 3).reshape(nvert, 3)
                off += nvert * 6
                ln = np.linalg.norm(raw, axis=1, keepdims=True)
                ln[ln == 0] = 1
                g.normals = (raw / ln).astype(np.float32)
            # UV лежат в NativeData-чанках (0x0510) у каждого TexCoordNative — best effort
            for nd in node.walk():
                if nd.type == NATIVE_DATA and len(nd.body) >= 4:
                    try:
                        arr = np.frombuffer(nd.body, dtype=np.uint8)
                        if len(arr) == nvert * 4:  # UV16 packed
                            u = arr[0::4].astype(np.float32) + arr[1::4].astype(np.float32) / 256.0
                            v = arr[2::4].astype(np.float32) + arr[3::4].astype(np.float32) / 256.0
                            g.uvs.append(np.stack([u, v], axis=1).astype(np.float32))
                    except Exception:
                        pass
    else:
        stride = 0
        attr_layout = []
        if fmt & 0x0100:
            attr_layout.append(("pos", 3)); stride += 12
        if fmt & 0x0200:
            attr_layout.append(("nrm", 3)); stride += 12
        if fmt & 0x0400:
            attr_layout.append(("col", 4)); stride += 4
        for i in range(nattrib):
            attr_layout.append((f"uv{i}", 2)); stride += 8
        if stride and nvert:
            total = nvert * stride
            base = 16 + nattrib * 4
            if base + total <= len(data_src.body):
                raw = np.frombuffer(data_src.body[base:base + total],
                                    dtype=np.uint8).reshape(nvert, stride)
                o = 0
                for kind, cnt in attr_layout:
                    if kind == "pos":
                        g.positions = raw[:, o:o + 12].view(np.float32).copy(); o += 12
                    elif kind == "nrm":
                        g.normals = raw[:, o:o + 12].view(np.float32).copy(); o += 12
                    elif kind == "col":
                        g.colors = raw[:, o:o + 4].copy(); o += 4
                    else:
                        g.uvs.append(raw[:, o:o + 8].view(np.float32).copy()); o += 8

    # ── примитивы (трианги) ──
    # формат записи индекса: 2 байта индекс; если format & 0x10000000 — ещё 2 байта material index.
    has_mat = bool(fmt & 0x10000000)
    esz = 4 if has_mat else 2          # байт на «индекс» в потоке
    tris = []
    first_prim = None
    for st in node.walk():
        if st.type not in (TRISTRIANGLES, TRILSTRIP, TRIFAN):
            continue
        if first_prim is None:
            first_prim = st
        nidx = len(st.body) // esz
        idx = np.frombuffer(st.body[:nidx * esz], dtype=np.uint16)[::(2 if has_mat else 1)]
        if st.type == TRISTRIANGLES:
            t = idx.reshape(-1, 3).copy()
        elif st.type == TRIFAN:
            t = np.empty((max(len(idx) - 2, 0), 3), np.uint16)
            for k in range(len(idx) - 2):
                t[k] = (idx[k], idx[k + 1], idx[k + 2])
        else:  # LSTRIP — чередование порядка обмотки
            t = np.empty((max(len(idx) - 2, 0), 3), np.uint16)
            for k in range(len(idx) - 2):
                if k % 2 == 0:
                    t[k] = (idx[k], idx[k + 1], idx[k + 2])
                else:
                    t[k] = (idx[k + 2], idx[k + 1], idx[k])
        tris.append(t)
    if tris:
        g.triangles = np.concatenate(tris).astype(np.int32)
    if first_prim is not None and has_mat and len(first_prim.body) >= 4:
        g.material_index = int(np.frombuffer(first_prim.body[2:4], dtype=np.uint16)[0])
    return g


def _parse_frames(node):
    fl = None
    for c in node.walk():
        if c.type == CH_FRAME_LIST:
            fl = c
            break
    if fl is None:
        return np.zeros((0, 9), np.float32), np.zeros((0, 3), np.float32), [-1]
    core = fl.find(CH_CORE)
    body = (core or fl).body
    nframe = struct.unpack_from("<I", body, 0)[0]
    rot = np.zeros((nframe, 9), np.float32)
    pos = np.zeros((nframe, 3), np.float32)
    parents = []
    off = 4
    for i in range(nframe):
        rot[i] = struct.unpack_from("<9f", body, off)
        pos[i] = struct.unpack_from("<3f", body, off + 36)
        parents.append(struct.unpack_from("<i", body, off + 48)[0])
        off += 52
    return rot, pos, parents


def _parse_atomics(clump, ltms):
    out = []
    for c in clump.walk():
        if c.type != CH_ATOMIC:
            continue
        ext = c.find(CH_ATOMIC_EXTENSION)
        gi = fi = -1
        if ext is not None:
            core = ext.find(CH_CORE)
            b = (core if core else ext).body
            gi, fi = struct.unpack_from("<ii", b, 0)[:2]
        ltm = ltms[fi] if 0 <= fi < len(ltms) else np.eye(4)
        out.append(Atomic(gi, fi, ltm))
    return out


def _parse_materials(clump):
    mats = []
    ml = None
    for c in clump.walk():
        if c.type == CH_MATERIAL_LIST:
            ml = c
            break
    if ml is None:
        return mats
    core = ml.find(CH_CORE)
    b = (core or ml).body
    n = struct.unpack_from("<I", b, 0)[0]
    names = []
    off = 4
    for _ in range(n):
        nm = b[off:off + 32].split(b"\x00")[0].decode("latin-1")
        tx = b[off + 32:off + 64].split(b"\x00")[0].decode("latin-1")
        names.append((nm, tx))
        off += 64
    # имена текстур из TextureNode (dict 0x16)
    texnames = {}
    for c in ml.walk():
        if c.type == CH_TEXTURE_DICT:
            dc = c.find(CH_CORE)
            if dc:
                tn = struct.unpack_from("<I", dc.body, 0)[0]
                for k in range(tn):
                    o = 4 + k * 4
                    if o + 4 <= len(dc.body):
                        texnames[k] = None
    for c in ml.walk():
        if c.type == CH_MAT:
            emb = c.find(CH_MAT_EMBEDDED)
            alpha = 1.0
            texname = ""
            if emb is not None:
                ec = emb.find(CH_CORE)
                if ec and len(ec.body) >= 28:
                    r, gg, bb, aa = struct.unpack_from("<BBBB", ec.body, 24)
                    alpha = aa / 255.0
            for sub in c.walk():
                if sub.type == CH_TEXTURE_NATIVE:
                    nc = sub.find(CH_CORE)
                    if nc and len(nc.body) >= 28:
                        dom, filt = struct.unpack_from("<HH", nc.body, 0)
                        nm = nc.body[4:36].split(b"\x00")[0].decode("latin-1")
                        texname = nm
            mats.append(Material(name=f"mat{len(mats)}", texture=texname, alpha=alpha))
    # если мат-листов пустых — дополняем именами
    while len(mats) < n:
        mats.append(Material(name=names[len(mats)][0] if len(mats) < len(names) else f"mat{len(mats)}"))
    return mats


def load_dff(dff_bytes: bytes, name: str = "model") -> RwdModel:
    clump = find_clump(dff_bytes)
    core = clump.find(CH_CORE)
    if core is None:
        raise ValueError("Clump без Core chunk")
    version, legacy_id = struct.unpack_from("<II", core.body, 0)
    ps2 = ((version >> 24) & 0xFF) == 0x01   # RW 1.10 = PS2-конвейер (страницы)
    plat = clump.header.platform_subtype == 1
    ps2 = ps2 or plat

    fr, fp, parents = _parse_frames(clump)
    ltms = all_ltms(fr, fp, parents) if len(fr) else np.zeros((0, 4, 4))

    geos = []
    for c in clump.walk():
        if c.type == CH_GEOMETRY:
            geos.append(_parse_geometry_node(c, ps2))

    atomics = _parse_atomics(clump, ltms)
    materials = _parse_materials(clump)

    model = RwdModel(name=name, geos=geos, atomics=atomics, materials=materials)
    ts = [g.triangles for g in geos if len(g.triangles)]
    if ts:
        acc = 0
        cat = []
        for g in geos:
            if len(g.triangles):
                cat.append(g.triangles + acc)
                acc += len(g.positions)
        model.triangles = np.concatenate(cat)
    return model

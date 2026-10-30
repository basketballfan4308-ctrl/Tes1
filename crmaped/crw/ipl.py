"""IPL: текстовые (секция inst) и бинарные (магия bnry).

Бинарный IPL (подтверждено аудитом): магия b"bnry", заголовок 76 байт,
затем записи по 40 байт: pos(3f) quat(4f) modelId(u32) interior(u32) lod(u32).
Игровая загрузка (CFileLoader::LoadObjectInstance): M = R(conj(q)) + t.

Sentinel-координаты 77777 исключаются. Для текстовых IPL ключ — ИМЯ модели.
"""
from __future__ import annotations

import re
import struct
from dataclasses import dataclass, field

SENTINEL = 77777.0


@dataclass
class Instance:
    model: str          # имя модели (для bnry — после резолва IDE id->name)
    pos: tuple[float, float, float]
    quat: tuple[float, float, float, float]   # (x, y, z, w) как в файле
    interior: int = 0
    lod_dist: int = 0
    source: str = ""
    model_id: int = -1

    @property
    def is_sentinel(self) -> bool:
        return any(abs(c - SENTINEL) < 1.0 for c in self.pos)


@dataclass
class IplFile:
    path: str = ""
    instances: list[Instance] = field(default_factory=list)
    other_sections: dict[str, list[str]] = field(default_factory=dict)  # enex/cull/... строки как есть


# ── текстовый IPL ─────────────────────────────────────────────────────────────

def parse_ipl_text(text: str, path: str = "") -> IplFile:
    ipl = IplFile(path=path)
    text = re.sub(r"//.*", "", text)
    sec = None
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        if s == "inst":
            sec = "inst"
            continue
        if s == "end":
            sec = None
            continue
        if re.fullmatch(r"[A-Za-z_]\w*", s) and sec != "inst":
            sec = s.lower()
            ipl.other_sections.setdefault(sec, [])
            continue
        if sec == "inst":
            parts = s.split()
            # name x y z w? ... формат SA: name px py pz rx? нет: name x y z qw qx qy qz [interior lod]
            # GTA SA text IPL: modelName x y z w x y z (кватернион идёт W,X,Y,Z!) interior lodTime
            if len(parts) < 5:
                continue
            name = parts[0]
            try:
                nums = [float(p) for p in parts[1:9]] if len(parts) >= 9 else None
            except ValueError:
                nums = None
            if nums is None:
                continue
            px, py, pz = nums[0], nums[1], nums[2]
            qw, qx, qy, qz = nums[3], nums[4], nums[5], nums[6]
            interior = int(float(parts[9])) if len(parts) > 9 else 0
            lod = int(float(parts[10])) if len(parts) > 10 else 0
            ipl.instances.append(Instance(name, (px, py, pz), (qx, qy, qz, qw),
                                          interior, lod, path))
        elif sec:
            ipl.other_sections.setdefault(sec, []).append(s)
    return ipl


# ── бинарный IPL (bnry) ───────────────────────────────────────────────────────

BNRY_HEADER_SIZE = 76
BNRY_RECORD_SIZE = 40


def parse_ipl_binary(data: bytes, path: str = "",
                     id_to_name: dict[int, str] | None = None) -> IplFile:
    """id_to_name обязателен для резолва числовых ID моделей (из IDE)."""
    if data[:4] != b"bnry":
        raise ValueError(f"{path}: не bnry, а {data[:4]!r}")
    ipl = IplFile(path=path)
    body = data[BNRY_HEADER_SIZE:]
    n = len(body) // BNRY_RECORD_SIZE
    for i in range(n):
        rec = body[i * BNRY_RECORD_SIZE:(i + 1) * BNRY_RECORD_SIZE]
        px, py, pz, qx, qy, qz, qw, mid, interior, lod = struct.unpack("<7fIII", rec)
        # пустые/нулевые padding-записи в конце файла — не инстансы
        if mid == 0 and abs(px) < 1e-6 and abs(py) < 1e-6 and abs(pz) < 1e-6 \
                and abs(qw) < 1e-6:
            continue
        name = None
        if id_to_name:
            name = id_to_name.get(mid) or id_to_name.get(str(mid))
        if name is None:
            name = f"#{mid}"
        ipl.instances.append(Instance(name, (px, py, pz), (qx, qy, qz, qw),
                                      int(interior), int(lod), path, int(mid)))
    return ipl


def load_ipl_any(data_or_text, path="", id_to_name=None) -> IplFile:
    if isinstance(data_or_text, (bytes, bytearray)):
        if bytes(data_or_text[:4]) == b"bnry":
            return parse_ipl_binary(bytes(data_or_text), path, id_to_name)
        data_or_text = bytes(data_or_text).decode("latin-1")
    low = data_or_text.lstrip()[:8].lower()
    if low.startswith("bnry"):
        return parse_ipl_binary(data_or_text.encode("latin-1"), path, id_to_name)
    return parse_ipl_text(data_or_text, path)

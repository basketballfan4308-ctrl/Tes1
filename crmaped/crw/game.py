"""Игровая база: обход gta.dat, чтение IMG/IDE/IPL (в т.ч. из IMG), резолв моделей.

gta.dat содержит строки:
    model <path>.ide | <path>.ipl        (относительно папки data\)
    image  models\<x>.img
gta3.img и gta_int.img в gta.dat НЕ перечислены — добавляем их принудительно.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from .img import ImgArchive
from .ide import parse_ide, IdeFile
from .ipl import IplFile, load_ipl_any


@dataclass
class GameBase:
    root: str
    ide_files: list[IdeFile] = field(default_factory=list)
    objects: dict[str, object] = field(default_factory=dict)     # name.lower() -> IdeObject
    id_to_name: dict[int, str] = field(default_factory=dict)
    archives: dict[str, ImgArchive] = field(default_factory=dict)  # stem.lower() -> archive
    ipl_paths_disk: list[str] = field(default_factory=list)
    ipl_in_img: list[tuple[str, str]] = field(default_factory=list)  # (archive, entry)

    @classmethod
    def open(cls, root: str) -> "GameBase":
        gb = cls(root=root)
        dat = os.path.join(root, "data", "gta.dat")
        if not os.path.exists(dat):
            raise FileNotFoundError(f"Не найден {dat}")
        with open(dat, "r", encoding="latin-1") as f:
            for line in f:
                t = line.strip()
                low = t.lower()
                parts = t.split(None, 1)
                if len(parts) != 2:
                    continue
                cmd, arg = parts[0].lower(), parts[1].strip().strip('"')
                if cmd == "model":
                    p = os.path.normpath(os.path.join(root, "data", arg))
                    ext = p.lower().rsplit(".", 1)[-1]
                    if ext == "ide" and os.path.exists(p):
                        gb.add_ide_file(p)
                    elif ext == "ipl" and os.path.exists(p):
                        gb.ipl_paths_disk.append(p)
                elif cmd == "image":
                    p = os.path.normpath(os.path.join(root, arg))
                    if os.path.exists(p):
                        gb.add_archive(p)
        # принудительно: gta3.img / gta_int.img (не в gta.dat, но используются)
        for extra in ("models/gta3.img", "models/gta_int.img"):
            p = os.path.normpath(os.path.join(root, extra))
            if os.path.exists(p) and os.path.splitext(os.path.basename(p))[0].lower() not in gb.archives:
                gb.add_archive(p)
        return gb

    def add_archive(self, path: str):
        stem = os.path.splitext(os.path.basename(path))[0].lower()
        try:
            self.archives[stem] = ImgArchive(path)
        except Exception as e:
            print(f"WARN: архив {path} не читается: {e}")

    def add_ide_file(self, path: str):
        with open(path, "r", encoding="latin-1", errors="replace") as f:
            ide = parse_ide(f.read(), path)
        self.ide_files.append(ide)
        for o in ide.objects.values():
            self.objects.setdefault(o.name.lower(), o)
        for oid, lst in ide.by_id.items():
            self.id_to_name.setdefault(oid, lst[0].name)

    # ── IPL ────────────────────────────────────────────────────────────────
    def all_ipls(self) -> list[IplFile]:
        out = []
        for p in self.ipl_paths_disk:
            try:
                with open(p, "rb") as f:
                    data = f.read()
                out.append(load_ipl_any(data, p, self.id_to_name))
            except Exception as e:
                print(f"WARN: IPL {p}: {e}")
        for arch in self.archives.values():
            for e in list(arch._order):
                if e.ext == "ipl":
                    try:
                        data = arch.read(e)
                        out.append(load_ipl_any(data, f"{arch.path}:{e.name}", self.id_to_name))
                    except Exception as ex:
                        print(f"WARN: IPL {e.name} в {os.path.basename(arch.path)}: {ex}")
        return out

    # ── ресурсы ────────────────────────────────────────────────────────────
    def find_resource(self, stem: str, ext: str) -> tuple[ImgArchive, object] | None:
        key = f"{stem.lower()}.{ext}"
        alt = stem.lower()
        for arch in self.archives.values():
            e = arch.entries.get(key) or arch.find(alt)
            if e:
                return arch, e
        return None

    def read_dff(self, model_name: str) -> bytes | None:
        r = self.find_resource(model_name, "dff")
        return r[0].read(r[1]) if r else None

    def read_txd(self, txd_name: str) -> bytes | None:
        r = self.find_resource(txd_name, "txd")
        return r[0].read(r[1]) if r else None

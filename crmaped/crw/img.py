"""IMG-архивы GTA (формат VER2).

TOC:   b'VER2' + uint32 count, затем ``count`` записей по 32 байта:
    uint32 offset   — начало данных в СЕКТОРАХ по 2048 байт
    uint32 size     — размер данных в секторах
    char   name[24] — Latin-1, NUL-terminated

Нюанс для архивов > 2 ГБ (наш случай: gamemod.img = 1.3 ГБ, но запись может
лежать высоко): в SA/VC старший бит поля offset (0x8000_0000) не является
флагом расширения — это просто часть 32-битного номера сектора, и смещение
может достигать ~4 ТБ. Поэтому offset = off_sec * 2048 без маскирования, а
корректность проверяется границами файла (иначе читались бы мусорные байты).
"""
from __future__ import annotations

import os
import struct
from dataclasses import dataclass
from typing import Iterable, Optional

SECTOR = 2048


@dataclass(frozen=True)
class ImgEntry:
    name: str          # как в архиве (ASCII/Latin-1, без расширения обычно)
    offset: int        # байтовое смещение в .img
    size: int          # размер данных в байтах
    index: int = -1

    @property
    def ext(self) -> str:
        return self.name.rsplit(".", 1)[-1].lower() if "." in self.name else ""


class ImgArchive:
    def __init__(self, path: str):
        self.path = path
        self.entries: dict[str, ImgEntry] = {}
        self._order: list[ImgEntry] = []
        self._parse_toc()

    def _parse_toc(self):
        with open(self.path, "rb") as f:
            head = f.read(8)
            if head[:4] != b"VER2":
                raise ValueError(f"{self.path}: не VER2, а {head[:4]!r}")
            count = struct.unpack_from("<I", head, 4)[0]
            toc = f.read(count * 32)
            if len(toc) < count * 32:
                raise ValueError(f"{self.path}: усечённый TOC ({len(toc)} < {count*32})")
            filesize = os.path.getsize(self.path)
            for i in range(count):
                off_sec, size_sec, raw_name = struct.unpack_from("<II24s", toc, i * 32)
                name = raw_name.split(b"\x00", 1)[0].decode("latin-1").strip()
                if not name:
                    continue
                offset = off_sec * SECTOR
                size = size_sec * SECTOR
                # В SA/VER2 старший бит offset — НЕ флаг расширения: это часть
                # 32-битного номера сектора (архивы до ~4 ТБ). Маскировать его нельзя,
                # иначе все записи выше 2 ГБ читались бы мусором. Корректность
                # проверяем только границами файла ниже.
                e = ImgEntry(name, offset, size, i)
                if offset + size > filesize:
                    e = ImgEntry(name, offset, max(0, filesize - offset), i)
                self.entries[name.lower()] = e
                self._order.append(e)

    def __len__(self):
        return len(self._order)

    def names(self) -> Iterable[str]:
        return (e.name for e in self._order)

    def find(self, stem_or_name: str) -> Optional[ImgEntry]:
        """Ищет по полному имени или по основе без расширения (регистронезависимо)."""
        key = stem_or_name.lower()
        e = self.entries.get(key)
        if e:
            return e
        for cand in self._order:
            c = cand.name.lower()
            if c == key or c.rsplit(".", 1)[0] == key:
                return cand
        return None

    def read(self, entry: ImgEntry) -> bytes:
        with open(self.path, "rb") as f:
            f.seek(entry.offset)
            return f.read(entry.size)

    def read_stream(self, entry: ImgEntry, n: int) -> bytes:
        """Читает только n байт (для больших DFF не тянем весь секторный хвост)."""
        with open(self.path, "rb") as f:
            f.seek(entry.offset)
            return f.read(min(n, entry.size))

"""Формат RenderWare: чтение потока чанков (RW 3.4, little-endian).

Структура чанка (12 байт):
    uint32 type
    uint32 size        — размер ПОСЛЕДУЮЩИХ данных (без этого заголовка)
    uint32 version     — library id: (ver<<26)|(subtype<<24)|(subver<<14)|build
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import Iterator, Optional


CHUNK_HEADER_FMT = "<III"
CHUNK_HEADER_SIZE = 12


@dataclass
class ChunkHeader:
    type: int
    size: int
    version: int

    @property
    def rw_version(self) -> int:
        return (self.version >> 26) & 0x3F

    @property
    def platform_subtype(self) -> int:
        """0 = PS2/PC generic, 1 = Xbox, 2 = GameCube(?), 3 = PS2 native geometry."""
        return (self.version >> 24) & 0x3

    @property
    def build(self) -> int:
        return self.version & 0x3FFF


@dataclass
class Chunk:
    """Чанк с материалом: заголовок + сырое тело + распарсенные дочерние чанки."""
    header: ChunkHeader
    offset: int                 # смещение заголовка в исходном буфере
    body: bytes                 # данные тела (size байт)
    children: list["Chunk"] = field(default_factory=list)

    @property
    def type(self) -> int:
        return self.header.type

    @property
    def total_size(self) -> int:
        return CHUNK_HEADER_SIZE + self.header.size

    def find(self, ctype: int) -> Optional["Chunk"]:
        for c in self.children:
            if c.type == ctype:
                return c
        return None

    def find_all(self, ctype: int) -> list["Chunk"]:
        return [c for c in self.children if c.type == ctype]

    def walk(self) -> Iterator["Chunk"]:
        yield self
        for c in self.children:
            yield from c.walk()


def read_header(buf: bytes, pos: int) -> ChunkHeader:
    t, s, v = struct.unpack_from(CHUNK_HEADER_FMT, buf, pos)
    return ChunkHeader(t, s, v)


def parse_chunk_tree(buf: bytes, pos: int = 0, end: Optional[int] = None,
                     depth: int = 0, max_depth: int = 64) -> Chunk:
    """Читает ОДИН чанк по адресу ``pos`` и рекурсивно разбирает его тело на дочерние.

    Возвращает Chunk (дочерние чанки лежат в ``children``).
    Тело листа (struct/string) остаётся в ``body`` нетронутым.
    """
    if end is None:
        end = len(buf)
    hdr = read_header(buf, pos)
    body_start = pos + CHUNK_HEADER_SIZE
    body_end = body_start + hdr.size
    if body_end > end:
        raise ValueError(f"chunk 0x{hdr.type:X} at {pos}: body_end {body_end} > end {end}")
    node = Chunk(hdr, pos, buf[body_start:body_end])
    if depth < max_depth:
        for child in iter_chunks(buf, body_start, body_end):
            node.children.append(parse_chunk_tree(buf, child.offset, end=body_end,
                                                  depth=depth + 1, max_depth=max_depth))
    return node


def iter_chunks(buf: bytes, start: int, end: int) -> Iterator[Chunk]:
    """Линейный обход последовательности чанков в [start, end) — только заголовки+границы.

    Именно так идёт RW-загрузчик игры: пока не упёрся в end. Это важно для DFF,
    где перед Clump (0x10) лежит UV-анимационный словарь (0x2B) — его нельзя
    «пропустить», надо корректно проскочить по размеру.
    """
    pos = start
    while pos + CHUNK_HEADER_SIZE <= end:
        hdr = read_header(buf, pos)
        body_start = pos + CHUNK_HEADER_SIZE
        body_end = body_start + hdr.size
        if body_end > end:
            break
        yield Chunk(hdr, pos, buf[body_start:body_end])
        pos = body_end


def dump_tree(node: Chunk, indent: int = 0, limit: int = 200) -> str:
    lines = []

    def rec(n: Chunk, d: int):
        if len(lines) >= limit:
            return
        flag = " *" if not n.children else ""
        lines.append(f"{'  ' * d}0x{n.type:08X} size={n.header.size} lib=0x{n.header.version:08X}{flag}")
        for c in n.children:
            rec(c, d + 1)

    rec(node, indent)
    return "\n".join(lines)

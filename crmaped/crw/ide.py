"""Идефикаторы GTA (IDE): секции objs / tobj / txdp / (anim).

Формат строки objs/tobj:
    id, modelName, textureDict, atomicModelCount, worldModelCount[, flags]
txdp:
    txdId, name
Комментарии: // до конца строки; секции в фигурных скобках: { ... }
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class IdeObject:
    id: int
    name: str
    txd: str
    n_atomic: int = 0
    n_world: int = 0
    flags: int = 0
    kind: str = "objs"   # objs|tobj

    @property
    def is_tobj(self) -> bool:
        return self.kind == "tobj"


@dataclass
class IdeFile:
    path: str = ""
    objects: dict[str, IdeObject] = field(default_factory=dict)   # key = name.lower()
    by_id: dict[int, list[IdeObject]] = field(default_factory=dict)
    txds: dict[int, str] = field(default_factory=dict)

    def get(self, name: str) -> IdeObject | None:
        return self.objects.get(name.lower())

    def resolve_model(self, name_or_id: str) -> IdeObject | None:
        """Ключ — ИМЯ модели (48 расхождений ID между IPL и IDE — известные опечатки)."""
        o = self.get(name_or_id)
        if o:
            return o
        m = re.fullmatch(r"\d+", name_or_id.strip())
        if m:
            lst = self.by_id.get(int(m.group()))
            if lst:
                return lst[0]
        return None


SEC_RE = re.compile(r"^(\w+)\s*\{", re.MULTILINE)


def parse_ide(text: str, path: str = "") -> IdeFile:
    ide = IdeFile(path=path)
    # убираем комментарии
    text = re.sub(r"//.*", "", text)
    # находим секции: имя { ... }
    for m in SEC_RE.finditer(text):
        sec = m.group(1).lower()
        start = m.end()
        depth = 1
        i = start
        while i < len(text) and depth:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        body = text[start:i - 1]
        for line in body.splitlines():
            line = line.strip().rstrip(",")
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            if sec in ("objs", "tobj"):
                if len(parts) < 3 or not parts[0].isdigit():
                    continue
                try:
                    oid = int(parts[0])
                except ValueError:
                    continue
                o = IdeObject(
                    id=oid, name=parts[1], txd=parts[2],
                    n_atomic=int(parts[3]) if len(parts) > 3 and parts[3].isdigit() else 0,
                    n_world=int(parts[4]) if len(parts) > 4 and parts[4].isdigit() else 0,
                    flags=int(parts[5]) if len(parts) > 5 and parts[5].isdigit() else 0,
                    kind=sec,
                )
                ide.objects.setdefault(o.name.lower(), o)
                ide.by_id.setdefault(o.id, []).append(o)
            elif sec == "txdp":
                if len(parts) >= 2 and parts[0].isdigit():
                    ide.txds[int(parts[0])] = parts[1]
    return ide

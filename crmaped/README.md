# CR Map Editor — визуальный редактор карт GTA «Криминальная Россия» → Unity 6.6 HDRP

Собственная архитектура: Python-ядро (парсинг RenderWare/IPL/IDE, экспорт glTF) + локальный
3D-вьювер в браузере (Three.js, без установки) + C#-плагин для импорта в Unity HDRP.

```
crmaped/
├── crw/                # ядро: img.py, ide.py, ipl.py, rw/ (dff, txd), scene.py, export_gltf.py
├── viewer/             # браузерный вьювер index.html (Three.js из CDN)
├── tools/              # diagnose_one.py, dump_lyt5etagn.py, export_district.py
├── unity/              # CrMapImporter.cs — плагин для Unity 6.6 HDRP
└── out/                # результаты (glb/json)
```

---

## Шаг 0. Установка (Windows, PowerShell)

### 0.1 Python

Если Python ещё не стоит: скачайте **Python 3.11–3.13** с https://www.python.org/downloads/windows/
и при установке **обязательно включите галку "Add python.exe to PATH"**. Проверка (новая вкладка PowerShell):

```powershell
python --version        # должно быть Python 3.10+
```

### 0.2 Зависимости

```powershell
python -m pip install --upgrade pip
python -m pip install numpy
# опционально, только для офлайн-рендера превью без браузера:
python -m pip install trimesh pygltflib pillow
```

`numpy` — единственная обязательная зависимость. Three.js-вьювер грузит библиотеку с CDN при открытии HTML.

### 0.3 Куда положить проект и как запустить

Скопируйте папку `crmaped` куда угодно, например `C:\crmaped`, и перейдите в неё —
**все команды ниже выполняются из этой папки**:

```powershell
cd C:\crmaped
dir    # должны быть видны папки crw, tools, viewer, unity и файл README.md
```

Проверка, что ядро вообще работает (эталонный тест математики, без файлов игры):

```powershell
python tests\test_rwmath.py
```

Ожидаемый вывод последней строкой: `ALL OK`. Если вместо этого ошибка `ModuleNotFoundError: No module named 'crw'` —
вы не в той папке (`cd` в `crmaped`, а не в `crmaped\tools`).

---

## Шаг 1. Диагностика на одной модели `lyt_5etagn` (главный тест корректности)

```powershell
python tools\diagnose_one.py --game "D:\Games\GTA Criminal Russia" --model lyt_5etagn
```

Что делает скрипт:
1. Находит инстанс `lyt_5etagn` во всех IPL (текстовых и бинарных `bnry`) игры.
2. Распаковывает DFF из IMG-архива (`gamemod.img` / `gta3.img` / `gta_int.img`).
3. Строит мировую матрицу **ровно так, как игра**: `M = R(conj(q)) + pos`
   (кватернион из IPL сопряжён — это и есть корень ваших «Пизанских башен»: без `conj`
   поворот зеркалится по оси, дом смотрит не туда и «врезается» в землю соседним LOD).
4. Печатает AABB вершин в мировых координатах (Z-up, система GTA) и Y-up (Unity).

### Что вы должны увидеть (числа-эталон)

Для `-2185.8, 178.3, 23.5`, quat `(0, 0, -0.826, 0.564)` (чистый Z-heading ≈ −111.3°):

- `yaw(Z-up) ≈ -111.3 deg`, `pitch = roll = 0` — подтверждение, что conj+row-vector даёт именно
  чистое вращение вокруг Z, как заявлено в IPL;
- центр XY AABB близко к `(-2185.8, 178.3)` (смещение = смещение pivot'а модели от центра);
- `z_min` близко к `23.5` (модели GTA стоят основанием на точку IPL, если pivot внизу);
- строка `OK:` или `WARN:` — автопроверки (нет NaN, quaternion нормализован, AABB конечен).

Если числа совпадают — трансформации корректны, идём дальше. Если нет — пришлите мне вывод целиком.

### Скриншот-подтверждение (по желанию)

```powershell
python tools\dump_lyt5etagn.py --game "D:\Games\GTA Criminal Russia" --model lyt_5etagn
viewer: открыть viewer\index.html, перетащить в него out\lyt_5etagn.glb
```

Вьювер покажет модель в мировой позиции с осью Z-up→Y-up конвертацией; камера сфокусирована на объекте.

---

## Шаг 2. Район lytkarino (505 инстансов)

```powershell
python tools\export_district.py --game "D:\Games\GTA Criminal Russia" --district lytkarino
```

Результат: `out\lytkarino.glb` (instanced: 1 mesh-примитив на модель, GPU-инстансы через EXT_mesh_gpu_instancing)
+ `out\lytkarino.json` (семантика: interior, LOD-связи, ENEX/GRGE/CULL/AUZO из IPL).

Просмотр: открыть `viewer\index.html`, перетащить `out\lytkarino.glb`. Фильтр LOD: клавиша `L`.

---

## Шаг 3. Вся карта (24 387 инстансов)

```powershell
python tools\export_map.py --game "D:\Games\GTA Criminal Russia" --chunks-out out\map
```

Чанки по районам (grid 256×256 м) → `out\map\chunk_X_Y.glb`. Исключения: sentinel `77777`,
LOD-прокси (флаг `--with-lods` включает), вариативные имена растительности схлопнуты в базовые модели автоматически.

---

## Шаг 4. Импорт в Unity 6.6 HDRP

1. Создайте проект **HDRP** (Unity 6.6+).
2. Скопируйте `unity\CrMapImporter.cs` в `Assets\Editor\` проекта.
3. Скопируйте папку `out\` (glb + json) в `Assets\StreamingAssets\crmap\`.
4. В Unity: меню **Tools → CR Map → Import District…** (или *Import All Chunks…*).

Плагин: читает glTF-геометрию (через com.unity.cloud.gltfast, ставится из Package Manager)
либо из нашего собственного `.crmesh` бинарного кэша (fallback, gltfast не обязателен), применяет
трансформы из манифеста, раскладывает по LODGroup'ам (base + LOD1..3 из IPL-связей), назначает
материалы `HDRP/Lit` по таблице маппинга (alpha MASK для листвы/заборов, emissive для ламп,
two-sided для листвы, vertex colors для ночной подсветки), включает GPU instancing (SRP Batcher friendly).

Требования Unity: Packages → `com.unity.render-pipelines.high-definition` (уже в HDRP-шаблоне),
опционально `com.unity.cloud.gltfast` (для чтения .glb напрямую; без него импорт идёт из .crmesh).

---

## Частые проблемы

| Симптом | Причина / решение |
|---|---|
| `FileNotFoundError: ...gamemod.img` | Укажите правильный путь `--game`. Проверьте, что `models\gamemod.img` существует. |
| Модель не найдена в архивах | Имя в IDE отличается от файла; скрипт печатает кандидаты — пришлите вывод. |
| Вьювер белый экран | Нужен интернет для CDN three.js; либо скачайте three.module.js рядом и укажите `?local=1`. |
| glb > 2 ГБ не открывается | Используйте `--split` у export_map (чанки и так ≤ 200 МБ). |

## Формулы-первопричины (для справки)

- IPL-инстанс: `p_world = p_local @ R(conj(q)).T + t`, где `R` — стандартная формула кватерниона
  (строки = базис right/up/at, row-vector convention RW).
- Конвертация GTA (Z-up, правая) → Unity (Y-up, левая): `M_unity = S · M_gta · S⁻¹`, `S`: (x,y,z)→(x,z,−y),
  det(S)=−1 — поэтому в Unity yaw = −(угол Z-up), handedness исправляется самим преобразованием.
- Текстовые IPL: ключ — **имя модели** (не ID): 48 расхождений ID между IDE и IPL — известные опечатки,
  сверяйтесь с `GTA_Criminal_Russia_Audit.md`.

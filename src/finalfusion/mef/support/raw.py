"""Metadata-minimal RAW TIFF discovery, Bayer packing, and TIFF output."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
import re
from typing import Iterable

import numpy as np
import tifffile


SUPPORTED_BAYER_PATTERNS = ("RGGB", "BGGR", "GRBG", "GBRG")
_PATTERN_RE = re.compile(r"(?<![A-Z])(RGGB|BGGR|GRBG|GBRG)(?![A-Z])", re.IGNORECASE)
_SCENE_RE = re.compile(r"^scene_\d+$", re.IGNORECASE)


class RawDataError(RuntimeError):
    """Raised when paired RAW files cannot satisfy the pipeline contract."""


@dataclass(frozen=True)
class RawPair:
    scene: str
    under_path: str
    over_path: str
    bayer_pattern: str
    height: int
    width: int
    dtype: str

    @property
    def packed_height(self) -> int:
        return self.height // 2

    @property
    def packed_width(self) -> int:
        return self.width // 2

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict) -> "RawPair":
        return cls(**value)


def parse_bayer_pattern(path: str | Path) -> str:
    """Read one of the four Bayer patterns from a RAW filename, not EXIF."""
    matches = _PATTERN_RE.findall(Path(path).name.upper())
    if len(matches) != 1:
        raise RawDataError(
            "Expected exactly one Bayer pattern token in filename: {}".format(path)
        )
    return matches[0]


def _single_tiff(directory: Path) -> Path:
    files = sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in {".tif", ".tiff"}
    )
    if len(files) != 1:
        raise RawDataError("Expected one TIFF in {}, found {}".format(directory, len(files)))
    return files[0]


def _scene_exposure_dir(scene_dir: Path, exposure: str) -> tuple[Path, bool]:
    """Resolve flat scene data or DatasetMEF's nested DNG component layout."""
    flat = scene_dir / exposure
    if flat.is_dir():
        return flat, False
    nested = scene_dir / "dng_components" / exposure / "raw" / "raw_unprocessed"
    if nested.is_dir():
        return nested, True
    raise RawDataError(
        "Missing {} RAW directory in {} (expected {}/ or "
        "dng_components/{}/raw/raw_unprocessed/)".format(exposure, scene_dir, exposure, exposure)
    )


def _flat_dng_path(scene_dir: Path, exposure: str) -> Path | None:
    """Return the flat ``<exposure>.dng`` path when present, else None."""
    path = scene_dir / (exposure + ".dng")
    return path if path.is_file() else None


def _bayer_from_rawpy(raw_pattern) -> str:
    """Map a rawpy 2x2 CFA pattern (0=R,1=G,2=B,3=G2) to RGGB/BGGR/GRBG/GBRG."""
    chars = {0: "R", 1: "G", 2: "B", 3: "G"}
    pattern = "".join(chars[int(value)] for value in np.asarray(raw_pattern).reshape(-1))
    if pattern not in SUPPORTED_BAYER_PATTERNS:
        raise RawDataError("Unsupported rawpy CFA pattern: {}".format(pattern))
    return pattern


def _inspect_dng(path: Path) -> tuple[int, int, np.dtype, str]:
    """Read a ProRAW DNG via rawpy and return (height, width, dtype, bayer)."""
    import rawpy
    with rawpy.imread(str(path)) as raw:
        mosaic = raw.raw_image
        pattern = _bayer_from_rawpy(raw.raw_pattern)
    height, width = mosaic.shape
    if height % 2 or width % 2:
        raise RawDataError("RAW DNG dimensions must be even for Bayer packing: {}".format(path))
    if mosaic.dtype.kind != "u":
        raise RawDataError("RAW DNG must have an unsigned integer dtype: {}".format(path))
    return int(height), int(width), np.dtype(mosaic.dtype), pattern


def _inspect_tiff(path: Path) -> tuple[int, int, np.dtype]:
    with tifffile.TiffFile(path) as handle:
        if len(handle.pages) != 1:
            raise RawDataError("Only single-page RAW TIFFs are supported: {}".format(path))
        page = handle.pages[0]
        shape = tuple(page.shape)
        dtype = np.dtype(page.dtype)
    if len(shape) != 2:
        raise RawDataError("Expected a 2-D Bayer mosaic in {}, got {}".format(path, shape))
    height, width = shape
    if height % 2 or width % 2:
        raise RawDataError("RAW dimensions must be even for Bayer packing: {}".format(path))
    if dtype.kind != "u":
        raise RawDataError("RAW TIFF must have an unsigned integer dtype: {}".format(path))
    return int(height), int(width), dtype


def discover_pairs(data_root: str | Path) -> list[RawPair]:
    """Discover flat or nested DatasetMEF scene pairs and validate RAW files."""
    root = Path(data_root).expanduser().resolve()
    if not root.is_dir():
        raise RawDataError("Data root does not exist: {}".format(root))

    pairs: list[RawPair] = []
    for scene_dir in sorted(
        path for path in root.iterdir() if path.is_dir() and _SCENE_RE.fullmatch(path.name)
    ):
        under_dng = _flat_dng_path(scene_dir, "under")
        over_dng = _flat_dng_path(scene_dir, "over")
        if under_dng is not None and over_dng is not None:
            under_h, under_w, under_dtype, under_pattern = _inspect_dng(under_dng)
            over_h, over_w, over_dtype, over_pattern = _inspect_dng(over_dng)
            if under_pattern != over_pattern:
                raise RawDataError(
                    "Under/over Bayer patterns differ for {}: {} vs {}".format(
                        scene_dir.name, under_pattern, over_pattern
                    )
                )
            if (under_h, under_w, under_dtype) != (over_h, over_w, over_dtype):
                raise RawDataError("Under/over shapes or dtypes differ for {}".format(scene_dir.name))
            pairs.append(
                RawPair(
                    scene=scene_dir.name,
                    under_path=str(under_dng),
                    over_path=str(over_dng),
                    bayer_pattern=under_pattern,
                    height=under_h,
                    width=under_w,
                    dtype=under_dtype.name,
                )
            )
            continue
        try:
            under_dir, under_nested = _scene_exposure_dir(scene_dir, "under")
            over_dir, over_nested = _scene_exposure_dir(scene_dir, "over")
        except RawDataError:
            continue
        under_path = _single_tiff(under_dir)
        over_path = _single_tiff(over_dir)
        try:
            under_pattern = parse_bayer_pattern(under_path)
        except RawDataError:
            if not under_nested:
                raise
            under_pattern = "BGGR"
            print("Using DatasetMEF fallback Bayer pattern BGGR for {}".format(scene_dir.name))
        try:
            over_pattern = parse_bayer_pattern(over_path)
        except RawDataError:
            if not over_nested:
                raise
            over_pattern = "BGGR"
        if under_pattern != over_pattern:
            raise RawDataError(
                "Under/over Bayer patterns differ for {}: {} vs {}".format(
                    scene_dir.name, under_pattern, over_pattern
                )
            )
        under_h, under_w, under_dtype = _inspect_tiff(under_path)
        over_h, over_w, over_dtype = _inspect_tiff(over_path)
        if (under_h, under_w, under_dtype) != (over_h, over_w, over_dtype):
            raise RawDataError(
                "Under/over shapes or dtypes differ for {}".format(scene_dir.name)
            )
        pairs.append(
            RawPair(
                scene=scene_dir.name,
                under_path=str(under_path),
                over_path=str(over_path),
                bayer_pattern=under_pattern,
                height=under_h,
                width=under_w,
                dtype=under_dtype.name,
            )
        )
    if not pairs:
        raise RawDataError("No scene directories were found in {}".format(root))
    return pairs


def bayer_channel_positions(pattern: str) -> dict[str, tuple[int, int]]:
    """Return canonical R, Gr, B, Gb locations in a named 2x2 Bayer tile."""
    pattern = pattern.upper()
    if pattern not in SUPPORTED_BAYER_PATTERNS:
        raise RawDataError("Unsupported Bayer pattern: {}".format(pattern))
    tile = np.asarray(list(pattern)).reshape(2, 2)
    r_row, r_col = np.argwhere(tile == "R")[0]
    b_row, b_col = np.argwhere(tile == "B")[0]
    green_positions = [tuple(position) for position in np.argwhere(tile == "G")]
    gr_positions = [position for position in green_positions if position[0] == r_row]
    gb_positions = [position for position in green_positions if position[0] == b_row]
    if len(gr_positions) != 1 or len(gb_positions) != 1:
        raise RawDataError("Cannot assign Gr/Gb positions for {}".format(pattern))
    return {
        "R": (int(r_row), int(r_col)),
        "Gr": (int(gr_positions[0][0]), int(gr_positions[0][1])),
        "B": (int(b_row), int(b_col)),
        "Gb": (int(gb_positions[0][0]), int(gb_positions[0][1])),
    }


def pack_bayer(mosaic: np.ndarray, pattern: str) -> np.ndarray:
    """Pack a Bayer mosaic to canonical [R, Gr, B, Gb] channel order."""
    if mosaic.ndim != 2 or mosaic.shape[0] % 2 or mosaic.shape[1] % 2:
        raise RawDataError("Expected an even-size 2-D Bayer mosaic")
    positions = bayer_channel_positions(pattern)
    return np.ascontiguousarray(
        np.stack(
            [
                mosaic[row::2, col::2]
                for row, col in (positions["R"], positions["Gr"], positions["B"], positions["Gb"])
            ],
            axis=0,
        )
    )


def unpack_bayer(packed: np.ndarray, pattern: str) -> np.ndarray:
    """Restore canonical [R, Gr, B, Gb] data to its original Bayer mosaic."""
    if packed.ndim != 3 or packed.shape[0] != 4:
        raise RawDataError("Expected packed RAW with shape [4, H, W]")
    positions = bayer_channel_positions(pattern)
    output = np.empty((packed.shape[1] * 2, packed.shape[2] * 2), dtype=packed.dtype)
    for channel, name in enumerate(("R", "Gr", "B", "Gb")):
        row, col = positions[name]
        output[row::2, col::2] = packed[channel]
    return output


def normalize_raw(raw_dn: np.ndarray, black_level: float, white_level: float) -> np.ndarray:
    if not white_level > black_level:
        raise ValueError("white_level must be greater than black_level")
    result = (raw_dn.astype(np.float32) - float(black_level)) / (float(white_level) - float(black_level))
    return np.clip(result, 0.0, 1.0)


def denormalize_raw(raw_linear: np.ndarray, black_level: float, white_level: float) -> np.ndarray:
    if not white_level > black_level:
        raise ValueError("white_level must be greater than black_level")
    values = np.clip(raw_linear, 0.0, 1.0) * (float(white_level) - float(black_level)) + float(black_level)
    return np.rint(values).astype(np.uint16)


def read_raw_mosaic(path: str | Path) -> np.ndarray:
    path = Path(path)
    if path.suffix.lower() == ".dng":
        import rawpy
        with rawpy.imread(str(path)) as raw:
            data = raw.raw_image.copy()
    else:
        data = tifffile.imread(path)
    if data.ndim != 2:
        raise RawDataError("Expected 2-D RAW TIFF: {}".format(path))
    return data


def write_raw_tiff(path: str | Path, mosaic_uint16: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(path, mosaic_uint16, photometric="minisblack", metadata=None)


def write_pair_manifest(path: str | Path, pairs: Iterable[RawPair]) -> None:
    content = {"schema_version": 1, "pairs": [pair.to_dict() for pair in pairs]}
    Path(path).write_text(json.dumps(content, indent=2) + "\n", encoding="utf-8")


def read_pair_manifest(path: str | Path) -> list[RawPair]:
    content = json.loads(Path(path).read_text(encoding="utf-8"))
    return [RawPair.from_dict(value) for value in content["pairs"]]

# Generic native data loader

One prebuilt library handles all runtime variant configurations. End users do
not need CMake, a C++ compiler, generated headers or a variant-specific DLL.
Use the library for your operating system/architecture from the same Trainer
revision; place it beside `nnue_dataset.py`. Runtime ABI 1 is checked on import.
The library is CPU-only and does not link to Python, PyTorch or CUDA.

```sh
python train.py train.bin val.bin --variant-config configs/metal-chess.json \
  --data-loader prp --record-bytes 7
```

The default is the bundled `variant.json`. `configs/metal-chess.json` uses the
user-supplied Metal Chess piece values. All existing training options remain.
PRP defaults to no smart or random skipping; sequential retains smart skipping
and random skipping of 3. Explicit skipping options override these defaults.
Record width 72 is PackedSfen; other widths in 7..144 use compact records.
Formats remain headerless, so select the correct JSON and record width.

The agreed JSON fields are `RANKS`, `FILES`, `KING_SQUARES`, `PIECE_TYPES`,
`PIECE_COUNT`, `USE_POCKETS` and `PIECE_VALUES`. Keys in `PIECE_VALUES` are
one-based dense indices represented as JSON strings. Python converts them to
integers. `SQUARES`, `PIECES` and pocket capacity are derived internally.
`DATA_SIZE` is no longer a compile-time variant parameter. Legacy 72-byte
PackedSfen decoding always uses its 512-bit position payload; larger old
PackedSfen profiles are not selected by `--record-bytes 136` (which means compact).

This initial version preserves `PIECE_COUNT`'s existing roles and feature
padding, the eight material buckets, King-square mapping conventions, and
all existing perspective/factorization rules. It does not introduce arbitrary
King maps, new one-bucket semantics, compact feature transport or `.npz` input.
In particular, `KING_SQUARES=1` retains the existing no-reference-King convention.
The compact record profile still requires two explicit NNUE Kings.

The record's dense type ordering must match its producer, exactly as with the
old generated configuration. Geometry/count metadata cannot infer piece names
from a headerless record. Reference-square mapping retains the existing full
board and nine-square-palace conventions rather than accepting new maps.

## Configuration, ownership and ABI

`variant.py` is now library code exposing an immutable `VariantConfig`; it is
not a generated configuration file. Do not overwrite it with an old generator's
`trainer_config` output. Convert those outputs separately:

```sh
python scripts/convert_variant_config.py old/variant.py old/variant.h new-variant.json
```

JSON is parsed and validated once. Native context creation copies its numeric
settings into an immutable shared configuration and prepares square-orientation
and King-map tables. Streams and feature extractors retain their own references,
so contexts can coexist, and closing a Python configuration handle cannot
invalidate a running stream. Each feature worker reuses a runtime-sized record
slab, avoiding allocations per decoded position.

The C ABI provides version, configuration create/destroy, record count, stream
create/fetch/destroy, batch release and error-reporting functions. Feature-family
selection is compiled and dispatched once per stream. No JSON or Python piece
loops occur during record processing. Worker errors are returned to Python as
exceptions, including legacy sequential-stream errors.

The output batch layout, dtypes and ownership remain unchanged. Native arrays
are copied before their native batch is released. CPU and CUDA feature-transformer
paths are unchanged. `PIECE_VALUES` only seeds Python PSQT initialization.

## Checkpoints and serialization

New checkpoints and `.pt` models retain the variant configuration. With no
explicit `--variant-config`, `train.py` inherits it when resuming; `serialize.py`
also inherits it for checkpoint/model input. Explicit incompatible settings
are rejected. Initialization values are preserved but do not define feature
index identity. Keep the same `--features` selection when resuming/exporting.

Older checkpoints/models lack this metadata: supply their matching JSON (the
bundled default applies if omitted). Legacy `.pt` feature objects are replaced
with the matching runtime-configured feature definitions on load. Shape checks
reject mismatched old model dimensions. Importing a raw `.nnue` requires the
matching configuration because that format is unchanged.

## Development builds and validation

CMake remains a developer/CI build dependency for now. Migration to the FSF Make
system is explicitly deferred, as requested. Platform-specific wheel packaging
is also a separate distribution improvement; the current build produces native
libraries and configuration bundles.

A future optional specialised build could consume the same JSON at build time,
embedding dimensions and feature offsets as constants. Its value would be tested
against this generic loader rather than assumed. It is not part of this version.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_PRP_TESTS=ON -DCMAKE_INSTALL_PREFIX=.
cmake --build build --config Release --parallel 2 --target install
ctest --test-dir build -C Release --output-on-failure
python -m unittest discover -s tests -p 'test_prp_*.py' -v
python -m unittest discover -s tests -p test_record_bytes.py -v
python -m unittest discover -s tests -p test_generic_loader.py -v
python -m unittest discover -s tests -p test_score_targets.py -v
```

CI builds/tests Windows, Linux and macOS libraries and uploads artifacts. The
native library can also be bundled without changing the Trainer sources:

```sh
python scripts/package_data_loader.py training_data_loader.dll generic-loader-windows-x64.zip
```

`tests/compare_loaders.py` compares the same generic binary with independently
compiled `prp_data_loader` references across multiple profiles. Python tests
also exercise interleaved contexts, large coordinates/types, native errors,
checkpoint mismatch checks, and all record widths. No fixed global maximum-board
allocation is imposed on ordinary small-board streams. Decoder and signed-int32
feature-ID limits are checked before unsupported data is processed.

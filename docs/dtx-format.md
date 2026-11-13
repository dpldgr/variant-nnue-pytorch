# DTX v1 distance-training format

DTX (Distance to x) is a header-described, fixed-width, mixed-material training
format. Version 1 initially exports **mate distance**, with up to **four pieces**
per record, from `mate-table`. Existing `.bin` formats and defaults are unchanged.

## Use

Select the format with the export filename extension (case-insensitive):

```text
load metal-chess.ini
setoption name UCI_Variant value metal-chess
mate-table pieces {K,k,N} export metal-training.dtx
```

After the job reports completion, further exports to that filename append records:

```text
mate-table pieces {K,k,Q,R} export metal-training.dtx
```

Wait for each asynchronous job's `mate-table complete` message before issuing the
next build. Every material uses the same file-wide stride. Do not specify
`entrysize` for DTX; that option remains exclusive to `.bin` exports.

Train using the matching generic loader (ABI 2):

```powershell
python train.py ..\metal-training.dtx ..\metal-training.dtx --variant-config configs\metal-chess.json --data-loader prp --score-target linear-recip --threads 2 --num-workers 4 --gpus 1 --max_epochs 200 --limit_val_batches 0
```

No `--record-bytes` is needed for DTX. The existing option/default applies to
`.bin`; DTX always reads its stride from its header. Sequential, global PRP and
buffered PRP readers are supported. Header bytes are excluded from all record
counts and permutations. Each reader caches its header and a label lookup table
once, then uses `payload_offset + record_index * record_bytes` for access.

Keep the matching `variant.json`/`--variant-config`: DTX does not change NNUE
feature construction, dense type ordering, bucket normalization or score-target
functions. The header's four-piece **storage capacity** is independent of the
NNUE schema's `PIECE_COUNT` (32 for the supplied Metal Chess schema). The reader
checks actual decoded piece counts against that model limit. Geometry and type
count must match; the JSON's dense type semantics must match the header catalogue.

## Physical layout

One header, followed by exactly `record_count * record_bytes` payload bytes.
There are no per-material sections, compression, continuation records, or
per-record headers. Integer header fields are little-endian. Record fields are
packed least-significant-bit first. Unused slots and padding must be zero.

The first 64 bytes are:

| Offset | Width | Meaning |
|---:|---:|---|
| 0 | 8 bytes | Magic `DTXTRN\r\n` |
| 8 | uint16 | Version, 1 |
| 10 | uint16 | Fixed prefix size, 64 |
| 12 | uint32 | Total header bytes / payload offset, a multiple of 8 |
| 16 | uint32 | Fixed record stride in bytes |
| 20 | uint16 | Board files |
| 22 | uint16 | Board ranks |
| 24 | uint16 | Number of non-King types |
| 26 | uint16 | Piece-slot capacity including the two Kings |
| 28 | uint8 | Square field bits |
| 29 | uint8 | Supporting-piece type bits |
| 30 | uint8 | Signed distance field bits |
| 31 | uint8 | Metric: 1=mate; 2=capture and 3=zeroing reserved for future producers |
| 32 | uint32 | Variant identifier length in bytes |
| 36 | uint32 | Piece catalogue length in bytes |
| 40 | uint64 | Record count, cumulative across completed appends |
| 48 | uint32 | Flags, currently zero |
| 52 | uint8 | Outcome encoding, 1=minimum signed value means draw; zero means terminal loss |
| 53 | uint8 | Bit order, 0=LSB first |
| 54 | uint8 | Colour width, 1 |
| 55 | uint8 | Side-to-move width, 1 |
| 56 | 8 bytes | Reserved, zero |

Metadata follows: the variant identifier bytes, then `non_king_types + 1` piece
symbols, each prefixed by a little-endian uint16 byte length. Symbols use the
variant's White spelling. The first symbol names the implicit King; the rest
list non-King types in dense encoding order. Zero padding rounds the header up
to an eight-byte boundary. The current Metal Chess header occupies **96 bytes**.
Headers are limited to 1 MiB. Unsupported versions/conventions and inconsistent
lengths are rejected.

The catalogue describes identities, not one material set. Pieces' individual
types and colours remain inside every record, so two-, three- and four-piece
materials can be mixed and shuffled within this one file.

## Metal Chess record: 44 useful bits, six bytes

| Bits | Meaning |
|---|---|
| 0–5 | White King square |
| 6–11 | Black King square |
| 12–17 | Supporting piece 1 square |
| 18–20 | Supporting piece 1 type |
| 21 | Supporting piece 1 colour |
| 22–27 | Supporting piece 2 square |
| 28–30 | Supporting piece 2 type |
| 31 | Supporting piece 2 colour |
| 32 | Side to move |
| 33–43 | Signed 11-bit distance/outcome code |
| 44–47 | Zero padding |

Squares are dense rank-major indices, a1=0 through h8=63. Colours and side to move
use White=0, Black=1. Supporting-piece wire types are Empty=0, Pawn=1, Queen=2,
Gold Rook=3, Silver Bishop=4, Iron Knight=5. Pawns therefore remain representable,
although the current fixed-material solver does not generate promotion-aware
pawn tables. The two Kings have implicit identities and colours.

An empty supporting slot has zero square, type and colour. Nonempty slots precede
empty ones. Exported supports are sorted by colour, dense type and square, so
input material-list order does not affect bytes. Slot capacity supplies the piece
count; no count field is stored. No evaluation score, separate WDL field,
promotion event/flag, en-passant state, game ply or move is stored.

For other geometries/catalogues, square width is `max(1, ceil(log2(cells)))` and
type width is `max(1, ceil(log2(non_king_types + 1)))`. The exporter uses capacity
4 and distance width 11. Readers use the declared widths/stride and validate
that they can encode the declared schema. Extra trailing padding is permitted
only when zero.

## Labels

The distance field uses two's complement with its minimum reserved for a draw:

| Signed 11-bit value | Meaning from side to move |
|---:|---|
| -1024 | Draw |
| 0 | Already mated / terminal loss |
| 1…1023 | Win in that many plies |
| -1…-1023 | Loss in that many plies |

The reserved draw is the spare "negative zero" discussed in the design; this is
an integer sentinel, not a floating-point zero. Other widths reserve the same
minimum-signed-value pattern. The initial compatible reader accepts 1–15
signed distance bits. It rejects non-mate metrics until their training meaning
is implemented. Metric IDs reserve space; they do not add capture/zeroing
solvers or extinction rules to `mate-table`.

A native LUT initialized once maps each code to the existing score/result tensors:
draw -> (0,0); win -> (32000-distance,+1); loss -> (-32000+distance,-1).
The chosen Python score-target LUT and model semantics remain unchanged. No
nonlinear mapping is computed per record in the native decoder.

DTX v1 cannot distinguish a terminal win at distance zero from a terminal loss,
so the exporter rejects stalemate-win variants and terminal-win-zero records.
It also rejects distances above 1023; it never clamps or truncates them. Each
record requires two present Kings. Current solver restrictions remain in force.

## Append and failure semantics

An existing destination must have exactly matching variant, catalogue and layout
metadata. Its length must agree with the header's count before any append starts.
A zero-length destination is initialized as a new DTX file. The count is updated
when an export successfully finishes. Normal failures and unfinished writer
lifetimes roll the append back to its original payload length and record count;
an orderly UCI stop can finish a valid partial export, like the existing bin path.
Only one exporter should write a given DTX file at a time. Train/read after the
export completes. An abrupt process termination can leave a length/count
mismatch; readers reject it rather than guessing or silently using partial data.

The header identifies the schema and variant name; it does not embed the full
variant rules or certify that two differently configured rulesets with the same
name are equivalent. Retain the variant definition with the training experiment.

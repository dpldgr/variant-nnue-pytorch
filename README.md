# Variant NNUE trainer

This is the chess variant NNUE training code for [Fairy-Stockfish](https://github.com/ianfab/Fairy-Stockfish). See the documentation in the [wiki](https://github.com/fairy-stockfish/variant-nnue-pytorch/wiki) for more details on the training process and [join our discord](https://discord.gg/FYUGgmCFB4) to ask questions. This project is derived from the [trainer for standard chess](https://github.com/glinscott/nnue-pytorch) used for official Stockfish.

# Setup

A [CUDA capable GPU](https://developer.nvidia.com/cuda-gpus) is strongly recommended.
Note: you don't need to download the CUDA Toolkit

Training on CPU only is also possible (see below), but much slower. In that case CuPy is not required,
and you can use requirements-CPU.txt instead of requirements.txt.

#### Package dependencies

At least Python 3.9 needed.

```
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

PyTorch with CUDA 11.8 will be automatically installed, along with the matching CuPy version
If your GPU is CUDA 12.8 capable you can use requirements-CUDA128.txt instead of requirements.txt

#### Build the fast DataLoader
This requires a C++17 compiler and cmake.

Windows:
```
compile_data_loader.bat
```

Linux/Mac:
```
sh compile_data_loader.bat
```

# Train a network

```
source env/bin/activate
python train.py train_data.bin val_data.bin
```

## Resuming from a checkpoint
```
python train.py --resume_from_checkpoint <path> ...
```

## Score targets

`--score-target` selects how the training score is converted to the evaluation
target. It is independent of `--data-loader`, and works with both readers:

```bash
python train.py train_data.bin val_data.bin --data-loader prp \
    --score-target linear-invlog --score-target-scale 64 \
    --no-smart-fen-skipping --random-fen-skipping 0
```

New runs default to `sigmoid`, the unchanged `sigmoid(score / 410)` expression.
The other modes expect game-theoretic records: a draw is score 0, and a decisive
score is signed `(32000 - DTM)`, with DTM in plies. Scores outside +/-32000 are
treated as terminal results (DTM 0). They are not centipawn target mappings.

For all DTM modes, the winning target starts at 1 and approaches 0.6; the losing
target is its complement, starting at 0 and approaching 0.4. Draws stay at 0.5.
Shorter wins and longer survival therefore remain preferable under the existing
engine evaluation convention. The prediction sigmoid, MSE loss, `--lambda`
result blending, NNUE architecture and exported format are unchanged.

Let `x = DTM / T`, with `--score-target-scale T` defaulting to **64 plies**.
These are the report's curves, scaled to the same initial winning-target slope
of `-0.2/T`:

| Mode | Winning target |
|---|---|
| `sigmoid` | Existing `sigmoid(score / 410)`; T is unused |
| `tanh` | `1 - 0.4*tanh(x/2)` |
| `a-law` | `1 - 0.4*C_A(z)`; bounded-input A-law below |
| `mu-law` | `1 - 0.4*C_mu(z)`; bounded-input mu-law below |
| `linear-invlog` | `1 - 0.2*x` for x <= 1, then `0.6 + 0.2/(1+log(x))` |
| `linear-exp` | `1 - 0.2*x` for x <= 1, then `0.6 + 0.2*exp(-(x-1))` |
| `linear-recip` | `1 - 0.2*x` for x <= 1, then `0.6 + 0.2/x` |

The A-law mode uses `A = 87.6`, `z = x/(x + 2*A/(1+log(A)))`, and
`C_A(z) = A*z/(1+log(A))` for z <= 1/A, otherwise
`C_A(z) = (1+log(A*z))/(1+log(A))`. The mu-law mode uses `mu = 255`,
`z = x/(x + 2*mu/log(1+mu))`, and `C_mu(z) = log(1+mu*z)/log(1+mu)`.
These bounded-input adaptations do not require a maximum DTM for normalization.
The three `linear-*` curves have matching values and slopes at x=1. A-law is
exactly linear in its initial z interval, approximately linear in DTM.

Each DTM mapping is precomputed once on initialization into a 65,536-entry,
float32 lookup table (256 KiB) covering the current signed-16-bit score format.
The buffer moves to the training device with the model and is not part of the
learned weights or checkpoint state_dict. Training batches perform only lookup;
the default sigmoid keeps its original expression. The continuous curve code
is separate from this record-format adapter for future score representations.

Checkpoints and `.pt` models retain the chosen mapping and scale. Omit the flags
when resuming to inherit them, or pass flags explicitly to change the objective.
Legacy checkpoints with no target settings default to sigmoid unless overridden.
An inference-only `.nnue` does not store training-target settings.

Only Python changed for this feature; no data-loader library rebuild is needed.
Run `python -m unittest discover -s tests -p test_score_targets.py -v` for the
target, loss, checkpoint and serialization tests.

## Training record order

`--data-loader sequential` is the default and preserves the existing reader,
cyclic batching, filtering options and `--epoch-size` behavior (20,000,000 by
default). Validation continues to use that sequential pipeline in both modes.

To shuffle records as they are read, without rewriting the dataset:

```bash
python train.py train_data.bin val_data.bin --data-loader prp \
    --no-smart-fen-skipping --random-fen-skipping 0
```

Rebuild the existing data-loader library before selecting `prp`; there is no
additional library. PRP uses a six-round Feistel permutation with cycle walking
to visit every source record exactly once per epoch. `--seed` and the epoch
number determine the order. Feature workers may finish out of order, but their
batches are delivered in permutation order, independent of `--num-workers`.

One PRP epoch is one complete dataset traversal. The final batch can be smaller;
it never wraps into the next epoch. Omit `--epoch-size` for automatic counting,
or supply the actual record count. Other values are rejected in PRP mode.
Filtering/skipping remain available and run after record selection: skipped
records are visited but not trained on. Their existing defaults are unchanged,
including random skipping of 3. Trainer limits such as `--limit_train_batches`
can deliberately interrupt a traversal.

PRP epochs are finite iterable datasets whose batch count depends on filtering;
the trainer may therefore display an unknown batch total. Fractional batch limits
that require a known total are not supported by Lightning for this loader; use
an integer batch limit for debugging. This first implementation supports a single
trainer process (CPU or GPU), with native worker threads for feature extraction.

Epoch-boundary checkpoints save the next permutation epoch and its configuration.
Resume with the same dataset contents, seed, batch size, features and filtering
settings. `--num-workers` may change. Mid-epoch checkpoints are rejected on resume
instead of silently skipping the remainder. A sequential checkpoint has no PRP
state; `--resume-from-model` can initialize weights for a new PRP run instead.

Record ordering uses logical indices through `IndexedTrainingDataSource`.
The initial `.bin` backend owns file sizing, fixed-width offsets and decoding,
using the record layout compiled into the existing library. Future variable-size
or compressed backends can implement indexed lookup without changing the PRP,
epoch or batching layers. Malformed, empty and unsupported files are rejected.

Native and Python regression tests:

```bash
cmake -S . -B build -DBUILD_PRP_TESTS=ON
cmake --build build --parallel 2 --target install
ctest --test-dir build --output-on-failure
python -m unittest discover -s tests -p 'test_prp_*.py' -v
```

## Training on GPU
```
python train.py --gpus 1 ...
```

## Training on CPU
```
python train.py --accelerator cpu --batch-size 1024 --threads 8 ...
```
Without a GPU (or without CuPy) the feature transformer falls back to a pure PyTorch implementation.

`scripts/cpu_smoke_test.sh <path to variant-nnue-tools stockfish>` runs the whole workflow (data generation, CPU training,
serialization, loading the net in the engine) on a tiny data set in about a minute. CI runs it on every push.
## Feature set selection
By default the trainer uses a factorized HalfKAv2 feature set (named "HalfKAv2^")
If you wish to change the feature set used then you can use the `--features=NAME` option. For the list of available features see `--help`
The default is:
```
python train.py ... --features="HalfKAv2^"
```

# Logging

```
pip install tensorboard
tensorboard --logdir=logs
```
Then, go to http://localhost:6006/


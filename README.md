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


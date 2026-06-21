# Continuous-Attractor Snow Forecasting

The model now represents regional snow state with the continuous coordinate

```text
(predicted next-day coverage, predicted coverage change)
```

A learnable two-dimensional anchor grid stores latent memory tokens. Bilinear
interpolation produces a continuous memory for arbitrary coordinates, while
the `nearest` mode provides a discrete-memory ablation. A shared Transformer
block performs self-attention, cross-attention to the retrieved memory, and an
explicit contraction of the pooled latent state toward the memory manifold.
The reported manifold energy is the latent squared distance to that memory.

## Pixel-Aligned Context

Each resized snow pixel is bound to:

- its pixel-center longitude and latitude from the source GeoTIFF bounds;
- interpolated elevation, slope, aspect, and terrain-radiation fields;
- nine NASA POWER variables for the last input day, trailing seven-day mean,
  target day, and target-minus-last change.

The weather and terrain fields are interpolated from the archived 5x5 point
grid. They are pixel-aligned conditioning fields, not direct pixel-resolution
measurements. Dynamic normalization is fitted only on training dates.

## Commands

```powershell
C:\Users\asus\.conda\envs\pytorch\python.exe train.py --smoke-test
C:\Users\asus\.conda\envs\pytorch\python.exe diagnose_manifold.py
C:\Users\asus\.conda\envs\pytorch\python.exe run_paper_experiments.py
```

The experiment matrix includes continuous memory, nearest-neighbor memory,
shared Transformer without memory, no manifold loss, no spatial context, no
attractor, no energy, no coverage guidance, and context U-Net variants.

Do not describe the spatial weather channels as observations at MODIS pixel
resolution. Do not claim formal Lyapunov stability; the implementation
provides empirical contraction and perturbation-recovery diagnostics.

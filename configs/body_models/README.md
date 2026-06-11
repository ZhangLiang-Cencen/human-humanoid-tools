# Body model weights (SMPL / SMPL-H / SMPL-X)

MPI-licensed weights are **not** shipped with this repository. After registering at the official sites, place files here:

```
configs/body_models/
├── smpl/
│   ├── SMPL_NEUTRAL.npz   # or .pkl
│   ├── SMPL_MALE.npz
│   └── SMPL_FEMALE.npz
├── smplh/
│   ├── SMPLH_NEUTRAL.npz
│   └── …
└── smplx/
    ├── SMPLX_NEUTRAL.npz
    └── …
```

Download links (non-commercial research license):

- [SMPL](https://smpl.is.tue.mpg.de)
- [SMPL+H](https://mano.is.tue.mpg.de)
- [SMPL-X](https://smpl-x.is.tue.mpg.de)

`hhtools` searches this directory **before** `~/.cache/hhtools/body_models/`. Override with `HHTOOLS_BODY_MODELS` if needed.

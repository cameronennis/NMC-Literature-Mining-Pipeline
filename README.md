# Solid-State NMC Power — project root

This repository holds the **Voltage Tracer** tool. Everything heavy or
machine-local (paper PDFs, run outputs, old versions) is kept out of the repo.

## Layout

```
Solid State NMC Power/            <- git repository root
├── Voltage Tracer/              <- the tool (tracked in git)
│   ├── voltage_tracer/          <- the Python package
│   ├── pyproject.toml
│   └── README.md                <- how the tool works + how to run it
└── Local Files (not in repo)/   <- NOT tracked (see .gitignore)
    ├── Paper PDFs (Top 21)/     <- the paper collection
    ├── Test Runs/               <- pipeline outputs from real runs
    ├── Debug Traces/            <- debug overlay images
    ├── Data Findings/           <- dashboard + spreadsheet
    └── Old Versions/            <- superseded code, kept for reference
```

## What's in git vs. local

- **Tracked:** the `Voltage Tracer/` code, this README, and `.gitignore`.
- **Not tracked:** the entire `Local Files (not in repo)/` folder, virtual
  environments (`.venv/`), caches, and OS files. To keep any new file local but
  off the repo, just drop it inside `Local Files (not in repo)/`.

## Running the tool

See [Voltage Tracer/README.md](Voltage%20Tracer/README.md) for full usage. In
short, from the `Voltage Tracer/` folder:

```bash
.venv/bin/voltage-tracer --pdf-dir "<your PDF folder>" --out-dir "<your output folder>"
```

Point `--pdf-dir` at the papers (e.g. `../Local Files (not in repo)/Paper PDFs (Top 21)/pdf`)
and `--out-dir` anywhere you like — putting run outputs under
`Local Files (not in repo)/` keeps them out of the repo automatically.

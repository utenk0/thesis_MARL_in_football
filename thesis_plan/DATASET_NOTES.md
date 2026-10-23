# Dataset Notes

Stage 2 uses exactly two sources: Dataset A (BASSEK) and the Metrica Sports
sample data. Raw files are stored under `data/raw/` and excluded from Git.

## Dataset A (BASSEK et al., 2025)

- **Title:** *An integrated dataset of spatiotemporal and event data in elite
  soccer*
- **Authors:** Manuel Bassek, Robert Rein, Hendrik Weber, and Daniel Memmert
- **Dataset source and DOI:** https://doi.org/10.6084/m9.figshare.28196177
- **Related article DOI:** https://doi.org/10.1038/s41597-025-04505-y
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0),
  https://creativecommons.org/licenses/by/4.0/
- **Citation requirement:** cite the dataset/article, credit the authors,
  link the license, and state whether the data was modified.
- **Matches:** seven official matches from the German Bundesliga first and
  second division.
- **Project path:** `data/raw/dataset_a/datasetA.zip`
- **Archive checksum (SHA-256):**
  `d4b57ed3308edb511d353a0d2d36c7ee99f7599c5b786147cf27b7683e0a8df5`
- **Tracking format:** seven DFL
  `positions_raw_observed_*.xml` files. Frames are timestamped at 40 ms
  intervals (25 Hz) and include players, referees, and the ball.
- **Event format:** seven DFL `events_raw_*.xml` files with timestamped,
  player/team-linked event records.
- **Metadata format:** seven DFL `matchinformation_*.xml` files describing
  matches, teams, players, and a 105 m by 68 m pitch.
- **Coordinate system:** tracking positions are centred metres; the pitch
  bounds are approximately `x ∈ [-52.5, 52.5]` and `y ∈ [-34, 34]`. Event
  locations use pitch metres from `0..105` and `0..68` and are normalized by
  the project converter.

The archive was checked locally and contains one metadata, one event, and one
position XML file for each of the seven matches.

## Metrica Sports Sample Data

- **Title:** Metrica Sports Sample Data
- **Source:** https://github.com/metrica-sports/sample-data
- **Project path:** `data/raw/metrica_sample/`
- **Source revision copied:** `e706dd506b360d69d9d123d5b8026e7294b13996`
- **License/usage terms:** the upstream repository does not provide a separate
  standard license file. Its README asks users to use the data responsibly and
  acknowledge Metrica Sports for public use. Those upstream terms should be
  retained and reviewed before redistribution.
- **Citation requirement:** acknowledge “Metrica Sports Sample Data” and link
  https://github.com/metrica-sports/sample-data in the thesis and any public
  output.
- **Matches:** three anonymized sample games.
- **Tracking format:** Games 1 and 2 provide separate home/away CSV tracking
  files; Game 3 provides FIFA EPTS tracking text plus XML metadata.
- **Event format:** Games 1 and 2 provide synchronized event CSV files; Game 3
  provides JSON events.
- **Coordinate system:** Games 1 and 2 use normalized coordinates from `0` to
  `1` on each axis, with `(0, 0)` at top-left and `(0.5, 0.5)` at midfield, on
  a 105 m by 68 m pitch. Game 3 uses its FIFA EPTS representation and must be
  interpreted through its metadata.

The project-local copy contains all nine game data files, the upstream README,
and the Metrica event-definition document.

## Storage policy

Raw datasets are intentionally ignored by Git because they are large. Dataset
A occupies about 2.4 GB and Metrica about 175 MB locally. Derived JSONL, NumPy,
and transition files belong in `data/processed/`, not beside the raw sources.
The tracked `data/dataset_sources.yaml` file provides a machine-readable record
of the local paths, versions, and Dataset A checksum.

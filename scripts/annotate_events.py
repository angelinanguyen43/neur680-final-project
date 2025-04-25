#!/usr/bin/env python
"""
annotate_events.py
Add a `feat_row` column to every *_events.tsv for BOLD5000.

Key idea
--------
File-names coming from the scanner are unique once you look at the
**basename** (e.g. `COCO_train2014_000000123.jpg`,
`n01440764_01234.JPEG`, `airplanecabin1.jpg`).  Your feature table
already has one row per basename, so we map directly:

    basename  →  stimulus_index row
"""
from pathlib import Path
import pandas as pd
from bids import BIDSLayout

from config import RAW, FEAT_DIR, LOGGER


def main() -> None:
    stim_df = pd.read_csv(FEAT_DIR / "stimulus_index.csv")
    id2row  = {Path(p).name: idx for idx, p in enumerate(stim_df["stimulus_id"])}

    layout  = BIDSLayout(RAW, validate=False)
    events  = layout.get(suffix="events", extension=".tsv")

    for ev in events:
        tsv = Path(ev.path)
        df  = pd.read_csv(tsv, sep="\t")

        if "stim_file" not in df.columns:        # localiser runs, etc.
            continue

        df["feat_row"] = df["stim_file"].map(lambda s: id2row.get(Path(s).name))

        out = tsv.with_name(
            tsv.stem.replace("_events", "_events_with_feat_row") + ".tsv"
        )
        df.to_csv(out, sep="\t", index=False)
        LOGGER.info("wrote %s", out.relative_to(RAW.parent))


if __name__ == "__main__":
    main()

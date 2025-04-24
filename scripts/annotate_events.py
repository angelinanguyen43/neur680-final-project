#!/usr/bin/env python
"""
annotate_events.py
Add a `feat_row` column to every *_events.tsv for BOLD5000.

Match on *file name* (stimulus_id) so it works regardless of absolute paths.
"""
from pathlib import Path
import pandas as pd
from bids import BIDSLayout

from config import RAW, FEAT_DIR, LOGGER

def main() -> None:
    stim_df   = pd.read_csv(FEAT_DIR / "stimulus_index.csv")
    id2row    = dict(zip(stim_df["stimulus_id"], stim_df.index))

    layout    = BIDSLayout(RAW, validate=False)
    events    = layout.get(suffix="events", extension=".tsv")

    for ev in events:
        tsv = Path(ev.path)
        df  = pd.read_csv(tsv, sep="\t")

        if "stim_file" not in df.columns:          # non-scene runs → skip
            continue

        df["feat_row"] = df["stim_file"].map(id2row)

        out = tsv.with_name(
            tsv.stem.replace("_events", "_events_with_feat_row") + ".tsv"
        )
        df.to_csv(out, sep="\t", index=False)
        LOGGER.info("wrote %s", out.relative_to(RAW.parent))

if __name__ == "__main__":
    main()

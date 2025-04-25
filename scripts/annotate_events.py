#!/usr/bin/env python
"""
annotate_events.py – Add a `feat_row` column to every *_events.tsv.

Maps each stimulus file’s **basename** to its row index in
features/stimulus_index.csv.  Non-task runs are left untouched.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
from bids import BIDSLayout
from config import RAW, FEAT_DIR, LOGGER


def main() -> None:
    stim_df = pd.read_csv(FEAT_DIR / "stimulus_index.csv")
    id2row  = {Path(p).name: i for i, p in enumerate(stim_df["stimulus_id"])}

    layout = BIDSLayout(RAW, validate=False)
    for ev in layout.get(suffix="events", extension=".tsv"):
        df = pd.read_csv(ev.path, sep="\t")
        if "stim_file" not in df.columns:
            continue
        df["feat_row"] = df["stim_file"].map(lambda s: id2row.get(Path(s).name))
        out = Path(ev.path).with_name(
            Path(ev.path).stem.replace("_events", "_events_with_feat_row") + ".tsv")
        df.to_csv(out, sep="\t", index=False)
        LOGGER.info("wrote %s", out.relative_to(RAW.parent))


if __name__ == "__main__":
    main()

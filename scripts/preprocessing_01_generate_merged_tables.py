"""
Generate merged trials table for included EIDs.

Input:
    - Filtered recordings file produced by the QC pipeline (BWM_LL_release_afterQC_df.csv)
Output:
    - ibl_included_eids_trials_table2025_full.csv
"""

# %%
import logging
from pathlib import Path
import pandas as pd
from one.api import ONE
import config as C
from scripts.utils.data_utils import load_filtered_recordings, add_age_group
from scripts.utils.behavior_utils import create_trials_table

log = logging.getLogger(__name__)


def main():
    one = ONE()

    # --- Load filtered sessions produced by the QC pipeline
    recordings_filtered = load_filtered_recordings(filename="BWM_LL_release_afterQC_df.csv")
    eids = recordings_filtered["eid"].astype(str).unique().tolist()
    log.info(f"{len(eids)} sessions remaining after QC")

    # --- Build merged trials table from the included sessions
    trials_table, err_list = create_trials_table(eids, one) #DONE: test a few eids

    # Optional: basic annotations (e.g., age group)
    trials_table = add_age_group(trials_table) 

    # --- Save
    outpath = C.DATAPATH / "ibl_included_eids_trials_table2025_full.csv"
    outpath.parent.mkdir(parents=True, exist_ok=True)
    trials_table.to_csv(outpath, index=False)

    # --- Logging: summary + errors (if any)
    n_mice = trials_table["mouse_name"].nunique() if "mouse_name" in trials_table.columns else "NA"
    n_trials = len(trials_table)
    log.info(f"New trials table saved: {outpath.resolve()}")
    log.info(f"{n_mice} mice, {n_trials} trials")

    if err_list:
        log.warning(f"Encountered {len(err_list)} errors while fetching trials (showing up to 5):")
        for e in err_list[:5]:
            log.warning(f"  - {e}")


if __name__ == "__main__":
    from scripts.utils.io import setup_logging

    setup_logging()
    main()


# info/

| file | what it is |
|---|---|
| `electrode_to_row_P{2,3,4}.json` | each array's map from electrode id (as named in the spike-sorting files) to its row in the NS5 signal, and the channel counts whose map was verified |
| `splits_summary.csv` | recordings and recording days per patient, split and group (the recordings that were cached and used) |
| `splits.csv` | **not distributed**: one row per recording (name, date, files, split, group). It names clinical recordings. `python -m utils.split` builds it from local data and asserts the split rules; its hash is stored with every run and every score |

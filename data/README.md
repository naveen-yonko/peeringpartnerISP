# Data Guide

This project intentionally does not store large datasets in Git.
Use this folder to keep local raw and processed files.

## Folder Layout

- data/raw/: downloaded source files
- data/processed/: generated intermediate/final CSV files

## Not Committed to Git

The repository ignores large data directories so pushes to GitHub do not fail:

- data/raw/
- data/processed/

## Rebuild Pipeline

Run all commands from the repository root.

1) Install dependencies

```bash
pip install -r requirements.txt
```

2) Download CAIDA AS Rank JSON

```bash
python src/data_collection/fetch_as_rank.py
```

Output:
- data/raw/caida_as_rank.json

3) Download PeeringDB dump JSON

```bash
python src/data_collection/fetch_peeringdb.py
```

Output:
- data/raw/peeringdb_2_dump_2024_06_01.json

4) Get AS relationship file and extract it

The extractor expects this input file:
- data/raw/20240601.as-rel2.txt.bz2

If not present, download it from CAIDA:
- https://publicdata.caida.org/datasets/as-relationships/serial-2/

Then extract:

```bash
python src/data_collection/extract_asrel.py
```

Output:
- data/raw/20240601.as-rel2.txt

5) Build pair dataset

```bash
python src/preprocessing/build_pairs.py
```

Output:
- data/processed/as_pairs.csv

6) Add engineered features

```bash
python src/preprocessing/feature_engineering.py
```

Output:
- data/processed/as_pairs_features.csv

7) Prepare final ML dataset

```bash
python src/preprocessing/prepare_dataset.py
```

Output:
- data/processed/final_dataset.csv

## Notes

- Keep large files local and out of Git history.
- If a push fails with GH001 again, check whether large files were committed in earlier commits.
- Use Git LFS only if you intentionally want versioned large files in the repository.

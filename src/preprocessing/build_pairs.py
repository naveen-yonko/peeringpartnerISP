import os
import json
import pandas as pd

os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
print("Working directory:", os.getcwd())

CAIDA_AS_RANK_PATH = os.path.join("data", "raw", "caida_as_rank.json")
PEERINGDB_PATH     = os.path.join("data", "raw", "peeringdb_2_dump_2024_06_01.json")
AS_REL_PATH        = os.path.join("data", "raw", "20240601.as-rel2.txt")
OUTPUT_PATH        = os.path.join("data", "processed", "as_pairs.csv")

def load_caida_as_rank():
    print("Loading CAIDA AS Rank...")
    with open(CAIDA_AS_RANK_PATH, "r") as f:
        data = json.load(f)

    rows = []
    for entry in data:
        row = {
            "asn"          : entry.get("asn"),
            "asnName"      : entry.get("asnName"),
            "rank"         : entry.get("rank"),
            "cliqueMember" : entry.get("cliqueMember"),
            "seen"         : entry.get("seen"),
            "longitude"    : entry.get("longitude"),
            "latitude"     : entry.get("latitude"),
            "source"       : entry.get("source"),
            "customer"     : entry.get("asnDegree", {}).get("customer"),
            "peer"         : entry.get("asnDegree", {}).get("peer"),
            "provider"     : entry.get("asnDegree", {}).get("provider"),
            "total"        : entry.get("asnDegree", {}).get("total"),
            "NumberASNs"   : entry.get("cone", {}).get("numberAsns"),
            "NumberPrefix" : entry.get("cone", {}).get("numberPrefixes"),
            "NumberAddrs"  : entry.get("cone", {}).get("numberAddresses"),
            "Country"      : entry.get("organization", {}).get("country", {}).get("iso") if entry.get("organization") else None,
            "Org"          : entry.get("organization", {}).get("orgName") if entry.get("organization") else None,
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df["asn"] = pd.to_numeric(df["asn"], errors="coerce").dropna()
    df         = df.dropna(subset=["asn"])
    df["asn"]  = df["asn"].astype(int)
    print(f" CAIDA AS Rank: {len(df)} ASes loaded")
    return df


def load_peeringdb():
    print("Loading PeeringDB...")
    with open(PEERINGDB_PATH, "r") as f:
        data = json.load(f)

    net_data = data.get("net", {}).get("data", [])
    df       = pd.DataFrame(net_data)
    df["asn"] = pd.to_numeric(df["asn"], errors="coerce")
    df         = df.dropna(subset=["asn"])
    df["asn"]  = df["asn"].astype(int)
    print(f"✅ PeeringDB: {len(df)} ASes loaded")
    return df


def load_as_relationships():
    print("Loading AS Relationships...")
    rows = []
    with open(AS_REL_PATH, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("#"):
                continue
            parts = line.split("|")
            if len(parts) >= 3:
                rows.append({
                    "asn1"         : int(parts[0]),
                    "asn2"         : int(parts[1]),
                    "relationship" : int(parts[2])
                })

    df = pd.DataFrame(rows)
    print(f" AS Relationships: {len(df)} pairs loaded")
    return df


def build_pairs(df_caida, df_peeringdb, df_asrel):
    print("\nBuilding AS pairs...")

    common_asns = set(df_caida["asn"]) & set(df_peeringdb["asn"])
    print(f"  Common ASes (CAIDA ∩ PeeringDB): {len(common_asns)}")

    df_caida     = df_caida[df_caida["asn"].isin(common_asns)].copy()
    df_peeringdb = df_peeringdb[df_peeringdb["asn"].isin(common_asns)].copy()

    df_caida_renamed     = df_caida.add_prefix("caida_")
    df_peeringdb_renamed = df_peeringdb.add_prefix("pdb_")

    df_caida_renamed     = df_caida_renamed.rename(columns={"caida_asn": "asn"})
    df_peeringdb_renamed = df_peeringdb_renamed.rename(columns={"pdb_asn": "asn"})

    df_asrel_filtered = df_asrel[
        df_asrel["asn1"].isin(common_asns) &
        df_asrel["asn2"].isin(common_asns)
    ].copy()

    print(f"  AS pairs among common ASes: {len(df_asrel_filtered)}")

    df_asrel_filtered["label"] = (df_asrel_filtered["relationship"] == 0).astype(int)

    peering     = df_asrel_filtered[df_asrel_filtered["label"] == 1]
    non_peering = df_asrel_filtered[df_asrel_filtered["label"] == 0]
    print(f"  Peering pairs:     {len(peering)}")
    print(f"  Non-peering pairs: {len(non_peering)}")

    df_merged = df_asrel_filtered.merge(
        df_caida_renamed,
        left_on="asn1",
        right_on="asn",
        how="inner"
    ).drop(columns=["asn"])

    df_merged = df_merged.merge(
        df_peeringdb_renamed,
        left_on="asn1",
        right_on="asn",
        how="inner"
    ).drop(columns=["asn"])

    feature_cols = [c for c in df_merged.columns
                    if c not in ["asn1", "asn2", "relationship", "label"]]
    df_merged    = df_merged.rename(columns={c: f"{c}_1" for c in feature_cols})

    df_merged = df_merged.merge(
        df_caida_renamed,
        left_on="asn2",
        right_on="asn",
        how="inner"
    ).drop(columns=["asn"])

    df_merged = df_merged.merge(
        df_peeringdb_renamed,
        left_on="asn2",
        right_on="asn",
        how="inner"
    ).drop(columns=["asn"])

    feature_cols = [c for c in df_merged.columns
                    if c not in ["asn1", "asn2", "relationship", "label"]
                    and not c.endswith("_1")]
    df_merged    = df_merged.rename(columns={c: f"{c}_2" for c in feature_cols})

    print(f"\n Final dataset shape: {df_merged.shape}")
    print(f"   Columns: {len(df_merged.columns)}")
    print(f"   Sample:\n{df_merged.head(2).to_string()}")

    return df_merged


if __name__ == "__main__":
    print("Script started!\n")

    df_caida     = load_caida_as_rank()
    df_peeringdb = load_peeringdb()
    df_asrel     = load_as_relationships()

    df_pairs = build_pairs(df_caida, df_peeringdb, df_asrel)

    os.makedirs(os.path.join("data", "processed"), exist_ok=True)
    df_pairs.to_csv(OUTPUT_PATH, index=False)
    print(f"\n Saved to: {OUTPUT_PATH}")

    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    print(f"Total pairs    : {len(df_pairs)}")
    print(f"Total features : {len(df_pairs.columns)}")
    print(f"Peering (1)    : {(df_pairs['label']==1).sum()}")
    print(f"Non-Peering (0): {(df_pairs['label']==0).sum()}")
    print("\n Step 2b complete!")
    print("Script finished!")
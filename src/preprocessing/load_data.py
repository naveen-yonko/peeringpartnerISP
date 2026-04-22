import os
import json
import pandas as pd

# Fix working directory
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
print("Working directory:", os.getcwd())

# ── Config ───────────────────────────────────────────────────────────────────
CAIDA_AS_RANK_PATH = os.path.join("data", "raw", "caida_as_rank.json")
PEERINGDB_PATH     = os.path.join("data", "raw", "peeringdb_2_dump_2024_06_01.json")
AS_REL_PATH        = os.path.join("data", "raw", "20240601.as-rel2.txt")

# ── 1. Load CAIDA AS Rank ────────────────────────────────────────────────────
def load_caida_as_rank():
    print("\n" + "="*50)
    print("Loading CAIDA AS Rank...")
    print("="*50)

    with open(CAIDA_AS_RANK_PATH, "r") as f:
        data = json.load(f)

    # Flatten nested fields
    rows = []
    for entry in data:
        row = {
            "asn"           : entry.get("asn"),
            "asnName"       : entry.get("asnName"),
            "rank"          : entry.get("rank"),
            "cliqueMember"  : entry.get("cliqueMember"),
            "seen"          : entry.get("seen"),
            "longitude"     : entry.get("longitude"),
            "latitude"      : entry.get("latitude"),
            "source"        : entry.get("source"),
            # asnDegree fields
            "customer"      : entry.get("asnDegree", {}).get("customer"),
            "peer"          : entry.get("asnDegree", {}).get("peer"),
            "provider"      : entry.get("asnDegree", {}).get("provider"),
            "total"         : entry.get("asnDegree", {}).get("total"),
            # cone fields
            "NumberASNs"    : entry.get("cone", {}).get("numberAsns"),
            "NumberPrefix"  : entry.get("cone", {}).get("numberPrefixes"),
            "NumberAddrs"   : entry.get("cone", {}).get("numberAddresses"),
            # organization fields
            "Country"       : entry.get("organization", {}).get("country", {}).get("iso") if entry.get("organization") else None,
            "Org"           : entry.get("organization", {}).get("orgName") if entry.get("organization") else None,
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    print(f"✅ Loaded {len(df)} ASes")
    print(f"   Columns: {list(df.columns)}")
    print(f"   Sample:\n{df.head(3).to_string()}")
    print(f"\n   Missing values:\n{df.isnull().sum()}")

    return df


# ── 2. Load PeeringDB ────────────────────────────────────────────────────────
def load_peeringdb():
    print("\n" + "="*50)
    print("Loading PeeringDB...")
    print("="*50)

    with open(PEERINGDB_PATH, "r") as f:
        data = json.load(f)

    # PeeringDB has a nested structure — network data is under "net"
    print(f"   Top level keys: {list(data.keys())}")

    net_data = data.get("net", {}).get("data", [])
    print(f"   Number of network entries: {len(net_data)}")

    df = pd.DataFrame(net_data)

    print(f"✅ Loaded {len(df)} ASes")
    print(f"   Columns: {list(df.columns)}")
    print(f"   Sample:\n{df.head(3).to_string()}")
    print(f"\n   Missing values:\n{df.isnull().sum()}")

    return df


# ── 3. Load AS Relationships ─────────────────────────────────────────────────
def load_as_relationships():
    print("\n" + "="*50)
    print("Loading AS Relationships...")
    print("="*50)

    rows = []
    with open(AS_REL_PATH, "r") as f:
        for line in f:
            line = line.strip()
            # Skip comment lines
            if line.startswith("#"):
                continue
            parts = line.split("|")
            if len(parts) >= 3:
                rows.append({
                    "asn1"         : int(parts[0]),
                    "asn2"         : int(parts[1]),
                    "relationship" : int(parts[2])  # 0 = peering, -1 = provider-customer
                })

    df = pd.DataFrame(rows)

    # relationship = 0 means peering
    peering     = df[df["relationship"] == 0]
    non_peering = df[df["relationship"] != 0]

    print(f"✅ Loaded {len(df)} AS relationships")
    print(f"   Peering pairs (0):         {len(peering)}")
    print(f"   Non-peering pairs (-1):    {len(non_peering)}")
    print(f"   Sample:\n{df.head(5).to_string()}")

    return df


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Script started!")

    df_caida    = load_caida_as_rank()
    df_peeringdb = load_peeringdb()
    df_asrel    = load_as_relationships()

    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    print(f"CAIDA AS Rank ASes    : {len(df_caida)}")
    print(f"PeeringDB ASes        : {len(df_peeringdb)}")
    print(f"AS Relationships      : {len(df_asrel)}")

    # Find common ASes between CAIDA and PeeringDB
    caida_asns    = set(df_caida["asn"].dropna().astype(int))
    peeringdb_asns = set(df_peeringdb["asn"].dropna().astype(int))
    common_asns   = caida_asns & peeringdb_asns

    print(f"\nCommon ASes (CAIDA ∩ PeeringDB): {len(common_asns)}")
    print(f"   (Paper had 24,475 common ASes)")

    print("\n✅ All data loaded successfully!")
    print("Script finished!")
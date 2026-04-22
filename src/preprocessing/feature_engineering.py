import os
import json
import pandas as pd
import numpy as np
from collections import defaultdict

# Fix working directory
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
print("Working directory:", os.getcwd())

# ── Config ───────────────────────────────────────────────────────────────────
AS_REL_PATH    = os.path.join("data", "raw", "20240601.as-rel2.txt")
PAIRS_PATH     = os.path.join("data", "processed", "as_pairs.csv")
OUTPUT_PATH    = os.path.join("data", "processed", "as_pairs_features.csv")
PEERINGDB_PATH = os.path.join("data", "raw", "peeringdb_2_dump_2024_06_01.json")

# ── 1. Build Customer Cones ──────────────────────────────────────────────────
def build_customer_cones():
    print("Building customer cones...")
    provider_to_customers = defaultdict(set)

    with open(AS_REL_PATH, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("#"):
                continue
            parts = line.split("|")
            if len(parts) >= 3:
                asn1         = int(parts[0])
                asn2         = int(parts[1])
                relationship = int(parts[2])
                if relationship == -1:
                    provider_to_customers[asn1].add(asn2)

    def get_cone(asn):
        cone    = set()
        visited = set()
        queue   = list(provider_to_customers.get(asn, []))
        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            cone.add(current)
            queue.extend(provider_to_customers.get(current, []))
        return cone

    print("  Computing cones for all ASes (this may take a minute)...")
    all_asns = set(provider_to_customers.keys())
    for customers in provider_to_customers.values():
        all_asns.update(customers)

    cones = {}
    for i, asn in enumerate(all_asns):
        cones[asn] = get_cone(asn)
        if i % 10000 == 0:
            print(f"  Processed {i} / {len(all_asns)} ASes...")

    print(f"✅ Built cones for {len(cones)} ASes")
    return cones


# ── 2. Compute Cone Overlap ──────────────────────────────────────────────────
def compute_cone_overlap(asn1, asn2, cones):
    cone1 = cones.get(asn1, set())
    cone2 = cones.get(asn2, set())
    return len(cone1 & cone2)


# ── 3. Load IXP memberships from PeeringDB ───────────────────────────────────
def load_ixp_memberships():
    """
    Build a dict: asn -> set of IXP lan IDs it is present at
    This is the correct way to find shared PoPs between two ASes
    """
    print("Loading IXP memberships from PeeringDB...")
    with open(PEERINGDB_PATH, "r") as f:
        data = json.load(f)

    # netixlan = which networks are at which IXP LANs
    netixlan_data = data.get("netixlan", {}).get("data", [])
    print(f"  Found {len(netixlan_data)} netixlan entries")

    # Also load ix_count per ASN for PoP count (P1, P2)
    net_data = data.get("net", {}).get("data", [])
    asn_to_ixcount = {}
    for entry in net_data:
        asn = entry.get("asn")
        if asn is not None:
            asn_to_ixcount[int(asn)] = entry.get("ix_count", 0)

    # Build asn -> set of ixlan_ids
    asn_to_ixlans = defaultdict(set)
    for entry in netixlan_data:
        asn      = entry.get("asn")
        ixlan_id = entry.get("ixlan_id")
        if asn is not None and ixlan_id is not None:
            asn_to_ixlans[int(asn)].add(ixlan_id)

    print(f"✅ Loaded IXP memberships for {len(asn_to_ixlans)} ASes")
    return asn_to_ixlans, asn_to_ixcount


# ── 4. Compute Affinity Score ────────────────────────────────────────────────
def compute_affinity_score(asn1, asn2, asn_to_ixlans, asn_to_ixcount):
    """
    Formula from paper:
        P1  = number of PoPs for AS1 (ix_count)
        P2  = number of PoPs for AS2 (ix_count)
        P0  = number of shared PoPs (common IXP memberships)

        alpha_1->2 = (P2 - P0) / ((P1 - P0) + (P2 - P0) + P0)
        alpha_2->1 = (P1 - P0) / ((P1 - P0) + (P2 - P0) + P0)
        affinity   = sqrt(alpha_1->2 * alpha_2->1)
    """
    p1 = asn_to_ixcount.get(asn1, 0)
    p2 = asn_to_ixcount.get(asn2, 0)

    # Actual shared PoPs using IXP membership overlap
    ixlans1 = asn_to_ixlans.get(asn1, set())
    ixlans2 = asn_to_ixlans.get(asn2, set())
    p0      = len(ixlans1 & ixlans2)   # real shared IXPs

    denominator = (p1 - p0) + (p2 - p0) + p0
    if denominator <= 0:
        return 0.0

    alpha_1_2 = (p2 - p0) / denominator
    alpha_2_1 = (p1 - p0) / denominator

    affinity = np.sqrt(max(alpha_1_2, 0) * max(alpha_2_1, 0))
    return round(affinity, 6)


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Script started!\n")

    # Load existing pairs
    print("Loading pairs dataset...")
    df = pd.read_csv(PAIRS_PATH, low_memory=False)
    print(f" Loaded {len(df)} pairs")

    # Build customer cones
    cones = build_customer_cones()

    # Load IXP memberships
    asn_to_ixlans, asn_to_ixcount = load_ixp_memberships()

    # Compute features
    print("\nComputing Cone Overlap and Affinity Score...")
    print("(This will take a few minutes...)")

    cone_overlaps   = []
    affinity_scores = []

    for i, row in df.iterrows():
        asn1 = int(row["asn1"])
        asn2 = int(row["asn2"])

        cone_overlaps.append(
            compute_cone_overlap(asn1, asn2, cones)
        )
        affinity_scores.append(
            compute_affinity_score(asn1, asn2, asn_to_ixlans, asn_to_ixcount)
        )

        if i % 50000 == 0:
            print(f"  Processed {i} / {len(df)} pairs...")

    df["ConeOverlap"]   = cone_overlaps
    df["AffinityScore"] = affinity_scores

    print(f"\n Features computed!")
    print(f"   ConeOverlap   — mean: {df['ConeOverlap'].mean():.2f},   max: {df['ConeOverlap'].max()}")
    print(f"   AffinityScore — mean: {df['AffinityScore'].mean():.4f}, max: {df['AffinityScore'].max():.4f}")

    # Quick sanity check
    non_zero_affinity = (df["AffinityScore"] > 0).sum()
    print(f"   Non-zero AffinityScores: {non_zero_affinity} / {len(df)}")

    # Save
    os.makedirs(os.path.join("data", "processed"), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\n Saved to: {OUTPUT_PATH}")
    print(f"   Shape: {df.shape}")

    print("\n Step 2c complete!")
    print("Script finished!")
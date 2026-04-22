import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Fix working directory
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
print("Working directory:", os.getcwd())

# ── Config ───────────────────────────────────────────────────────────────────
INPUT_PATH   = os.path.join("data", "processed", "as_pairs_features.csv")
OUTPUT_PATH  = os.path.join("data", "processed", "final_dataset.csv")

# Features to DROP per AS (these are the 15 low-importance features from paper)
FEATURES_TO_DROP = [
    "cliqueMember",
    "seen",
    "status",
    "policy_ratio",
    "info_unicast",
    "rir_status",
    "policy_general",
    "rir_status_updated",
    "route_server",
    "status_dashboard",
    "info_multicast",
    "policy_contracts",
    "info_never_via_route_servers",
    "info_ipv6",
    # extra cols not useful for ML
    "relationship",
    "asnName",
    "name",
    "name_long",
    "notes",
    "website",
    "policy_url",
    "looking_glass",
    "aka",
    "social_media",
    "Org",
    "irr_as_set",
    "info_types",
    "info_scope",
    "info_type",
    "info_ratio",
    "allow_ixp_update",
    "poc_updated",
    "netixlan_updated",
    "netfac_updated",
    "updated",
    "created",
    "org_id",
    "source",
    "policy_locations",
    "status_dashboard",
    "rir_status_updated",
]

def drop_features(df):
    """Drop low importance features for both ASN1 and ASN2"""
    print("Dropping low importance features...")

    cols_to_drop = []
    for col in df.columns:
        # Strip _1 or _2 suffix to get base feature name
        base = col
        if col.endswith("_1") or col.endswith("_2"):
            base = col[:-2]

        # Strip caida_ or pdb_ prefix
        base = base.replace("caida_", "").replace("pdb_", "")

        if base in FEATURES_TO_DROP:
            cols_to_drop.append(col)

    cols_to_drop = list(set(cols_to_drop))
    print(f"  Dropping {len(cols_to_drop)} columns...")
    df = df.drop(columns=cols_to_drop, errors="ignore")
    print(f"  Remaining columns: {len(df.columns)}")
    return df


def encode_categoricals(df):
    """Encode categorical/boolean columns to numeric"""
    print("\nEncoding categorical columns...")

    for col in df.columns:
        if col in ["asn1", "asn2", "label", "ConeOverlap", "AffinityScore"]:
            continue

        # Boolean columns
        if df[col].dtype == bool:
            df[col] = df[col].astype(int)
            continue

        # Object columns — apply label encoding
        if df[col].dtype == object:
            df[col] = df[col].astype(str).fillna("unknown")
            le       = LabelEncoder()
            df[col]  = le.fit_transform(df[col])

    print("✅ Encoding complete")
    return df


def handle_missing_values(df):
    """Fill missing values"""
    print("\nHandling missing values...")

    before = df.isnull().sum().sum()

    # Fill numeric columns with 0
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)

    # Fill object columns with "unknown"
    object_cols = df.select_dtypes(include=["object"]).columns
    df[object_cols] = df[object_cols].fillna("unknown")

    after = df.isnull().sum().sum()
    print(f"  Missing values before: {before}")
    print(f"  Missing values after:  {after}")
    return df


def print_summary(df):
    print("\n" + "="*50)
    print("FINAL DATASET SUMMARY")
    print("="*50)
    print(f"Total pairs     : {len(df)}")
    print(f"Total features  : {len(df.columns) - 3}")  # exclude asn1, asn2, label
    print(f"Peering (1)     : {(df['label']==1).sum()} ({(df['label']==1).mean()*100:.1f}%)")
    print(f"Non-Peering (0) : {(df['label']==0).sum()} ({(df['label']==0).mean()*100:.1f}%)")
    print(f"\nFeature columns:")
    feat_cols = [c for c in df.columns if c not in ["asn1", "asn2", "label"]]
    for i, col in enumerate(feat_cols):
        print(f"  {i+1:3}. {col}")


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Script started!\n")

    # Install sklearn if needed
    try:
        from sklearn.preprocessing import LabelEncoder
    except ImportError:
        print("Installing scikit-learn...")
        os.system("pip install scikit-learn")
        from sklearn.preprocessing import LabelEncoder

    # Load
    print("Loading features dataset...")
    df = pd.read_csv(INPUT_PATH, low_memory=False)
    print(f"✅ Loaded {len(df)} pairs with {len(df.columns)} columns")

    # Clean
    df = drop_features(df)
    df = handle_missing_values(df)
    df = encode_categoricals(df)

    # Print summary
    print_summary(df)

    # Save
    os.makedirs(os.path.join("data", "processed"), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\n💾 Saved to: {OUTPUT_PATH}")
    print(f"   Shape: {df.shape}")

    print("\n✅ Step 2d complete! Preprocessing done!")
    print("Script finished!")
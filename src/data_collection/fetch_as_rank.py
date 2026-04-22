import os
import json
import time
import urllib3

# Fix working directory
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
print("Working directory:", os.getcwd())

import requests

# Disable SSL warning
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ── Config ──────────────────────────────────────────────────────────────────
OUTPUT_PATH = os.path.join("data", "raw", "caida_as_rank.json")
API_URL     = "https://api.asrank.caida.org/v2/graphql"

QUERY = """
query getASRank($offset: Int!) {
  asns(offset: $offset, first: 1000) {
    totalCount
    pageInfo {
      hasNextPage
    }
    edges {
      node {
        asn
        asnName
        rank
        organization { orgId orgName country { iso name } }
        cliqueMember
        seen
        longitude
        latitude
        source
        asnDegree {
          customer
          peer
          provider
          total
        }
        cone {
          numberAsns
          numberPrefixes
          numberAddresses
        }
      }
    }
  }
}
"""

def load_existing_data():
    """Load already downloaded ASes so we can resume"""
    if os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH, "r") as f:
            existing = json.load(f)
        print(f"Found existing data: {len(existing)} ASes already downloaded!")
        return existing
    return []

def fetch_all_asns():
    # ── Resume from existing data ────────────────────────────────────────────
    all_asns = load_existing_data()

    # Calculate which offset to start from
    start_offset = (len(all_asns) // 1000) * 1000
    print(f"  Resuming from offset {start_offset}...")

    # Remove possibly incomplete last batch and re-fetch it
    all_asns = all_asns[:start_offset]
    offset   = start_offset

    max_retries = 5

    print("Continuing CAIDA AS Rank download...")

    while True:
        retries = 0
        success = False

        while retries < max_retries:
            try:
                print(f"  Fetching offset {offset}...")

                response = requests.post(
                    API_URL,
                    json={"query": QUERY, "variables": {"offset": offset}},
                    headers={"Content-Type": "application/json"},
                    timeout=60,
                    verify=False
                )

                print(f"  Status code: {response.status_code}")

                if response.status_code != 200:
                    print(f"  Error: {response.text}")
                    retries += 1
                    time.sleep(10)
                    continue

                data = response.json()

                if "errors" in data:
                    print(f"  GraphQL errors: {data['errors']}")
                    retries += 1
                    time.sleep(10)
                    continue

                if "data" not in data or "asns" not in data["data"]:
                    print(f"  Unexpected response: {json.dumps(data)[:300]}")
                    retries += 1
                    time.sleep(10)
                    continue

                asns_data   = data["data"]["asns"]
                edges       = asns_data["edges"]
                has_next    = asns_data["pageInfo"]["hasNextPage"]
                total_count = asns_data["totalCount"]

                for edge in edges:
                    all_asns.append(edge["node"])

                print(f"Fetched {len(all_asns)} / {total_count} ASes...")

                # Save progress every 5000 records
                if len(all_asns) % 5000 == 0:
                    save_data(all_asns)
                    print(f"Progress saved at {len(all_asns)} ASes...")

                if not has_next:
                    print("  No more pages!")
                    return all_asns

                offset  += 1000
                success  = True
                time.sleep(0.5)
                break

            except requests.exceptions.ConnectionError as e:
                print(f"  Connection error (retry {retries+1}/{max_retries}): {e}")
                retries += 1
                time.sleep(15)  # wait longer before retry
            except requests.exceptions.Timeout:
                print(f"  Timeout (retry {retries+1}/{max_retries})...")
                retries += 1
                time.sleep(15)
            except Exception as e:
                print(f"  Unexpected error: {type(e).__name__}: {e}")
                retries += 1
                time.sleep(10)

        if not success:
            # Save what we have before giving up
            save_data(all_asns)
            print(f"Failed after {max_retries} retries at offset {offset}.")
            print(f"Saved {len(all_asns)} ASes. Run again to resume!")
            break

    print(f"\nDone! Total ASes fetched: {len(all_asns)}")
    return all_asns


def save_data(data):
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    print("Script started!")
    asns = fetch_all_asns()
    if asns:
        save_data(asns)
        print(f"\Final total: {len(asns)} ASes saved!")
    else:
        print("No data fetched. Check errors above.")
    print("Script finished!")
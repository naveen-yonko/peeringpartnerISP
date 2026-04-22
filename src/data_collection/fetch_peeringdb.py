import os
import urllib3
import json
import time

# Fix working directory
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
print("Working directory:", os.getcwd())

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

import requests

# ── Config ───────────────────────────────────────────────────────────────────
OUTPUT_PATH = os.path.join("data", "raw", "peeringdb_2_dump_2024_06_01.json")
URL         = "https://publicdata.caida.org/datasets/peeringdb/2024/06/peeringdb_2_dump_2024_06_01.json"
MAX_RETRIES = 20       # keep retrying up to 20 times
CHUNK_SIZE  = 512 * 1024  # 512 KB chunks

def download_peeringdb():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    # Check if already fully downloaded
    if os.path.exists(OUTPUT_PATH):
        try:
            with open(OUTPUT_PATH, "r") as f:
                data = json.load(f)
            print(f" File already fully downloaded and valid!")
            print(f"   Keys: {list(data.keys())[:5]}")
            return
        except:
            print(" Existing file is incomplete. Will resume download...")

    attempt = 0

    while attempt < MAX_RETRIES:
        attempt += 1

        # Check how much we already downloaded (for resume)
        downloaded = 0
        if os.path.exists(OUTPUT_PATH):
            downloaded = os.path.getsize(OUTPUT_PATH)
            print(f"\n▶ Attempt {attempt}: Resuming from {downloaded/1024/1024:.1f} MB...")
        else:
            print(f"\n▶ Attempt {attempt}: Starting fresh download...")

        try:
            # Set Range header to resume from where we left off
            headers = {"Range": f"bytes={downloaded}-"}

            response = requests.get(
                URL,
                headers=headers,
                stream=True,
                verify=False,
                timeout=60
            )

            print(f"   Status code: {response.status_code}")

            # 206 = partial content (resume), 200 = full download
            if response.status_code not in (200, 206):
                print(f"Bad status: {response.text[:200]}")
                time.sleep(10)
                continue

            # If server doesn't support resume (returns 200 instead of 206)
            # we need to start fresh
            if response.status_code == 200 and downloaded > 0:
                print("Server doesn't support resume. Restarting...")
                downloaded = 0
                open(OUTPUT_PATH, "wb").close()

            total_size = int(response.headers.get("content-length", 0)) + downloaded
            print(f"   Total size: {total_size/1024/1024:.1f} MB")

            # Append to existing file if resuming, else write fresh
            mode = "ab" if downloaded > 0 else "wb"

            with open(OUTPUT_PATH, mode) as f:
                for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        percent     = (downloaded / total_size * 100) if total_size else 0
                        print(f"   Downloaded: {downloaded/1024/1024:.1f} MB / {total_size/1024/1024:.1f} MB ({percent:.1f}%)")

            # Download finished — validate JSON
            print("\n   Validating JSON...")
            with open(OUTPUT_PATH, "r") as f:
                data = json.load(f)
            print(f"Download complete and valid!")
            print(f"   Keys: {list(data.keys())[:5]}")
            print(f"   Saved to: {OUTPUT_PATH}")
            return   # ← success, exit function

        except requests.exceptions.ConnectionError as e:
            print(f"Connection dropped: {e}")
            print(f"   Waiting 15 seconds before retry...")
            time.sleep(15)

        except requests.exceptions.Timeout:
            print(f"Timeout. Waiting 15 seconds before retry...")
            time.sleep(15)

        except json.JSONDecodeError:
            print(f"JSON invalid — file may be incomplete. Retrying...")
            time.sleep(10)

        except Exception as e:
            print(f"Unexpected error: {type(e).__name__}: {e}")
            time.sleep(10)

    print(f"\Failed after {MAX_RETRIES} attempts.")
    print(f"   {OUTPUT_PATH} may be partially downloaded.")


if __name__ == "__main__":
    print("Script started!")
    download_peeringdb()
    print("Script finished!")
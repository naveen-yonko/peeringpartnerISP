import os
import bz2
import shutil

# Fix working directory
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
print("Working directory:", os.getcwd())

# ── Config ───────────────────────────────────────────────────────────────────
INPUT_PATH  = os.path.join("data", "raw", "20240601.as-rel2.txt.bz2")
OUTPUT_PATH = os.path.join("data", "raw", "20240601.as-rel2.txt")

def extract_bz2():
    # Skip if already extracted
    if os.path.exists(OUTPUT_PATH):
        size = os.path.getsize(OUTPUT_PATH)
        print(f"Already extracted! Size: {size/1024/1024:.1f} MB")
        print(f"   Path: {OUTPUT_PATH}")
        return

    print(f"Extracting {INPUT_PATH}...")

    try:
        with bz2.open(INPUT_PATH, "rb") as f_in:
            with open(OUTPUT_PATH, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

        size = os.path.getsize(OUTPUT_PATH)
        print(f" Extraction complete!")
        print(f"   Saved to: {OUTPUT_PATH}")
        print(f"   Size: {size/1024/1024:.1f} MB")

        # Preview first 5 lines
        print("\n   First 5 lines preview:")
        with open(OUTPUT_PATH, "r") as f:
            for i, line in enumerate(f):
                if i >= 5:
                    break
                print(f"   {line.strip()}")

    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")


if __name__ == "__main__":
    print("Script started!")
    extract_bz2()
    print("Script finished!")
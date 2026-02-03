import os
import shutil
import random
from PIL import Image
import numpy as np
from tqdm import tqdm

# --- CONFIGURATION ---
BASE_DIR = r"C:\Users\Cado\Desktop\cado\fyp\Datasets\FYP_Lymph_Project"
SRC_REAL_ROOT = os.path.join(BASE_DIR, "02_Processed_Data", "Real")
SRC_FAKE_ROOT = os.path.join(BASE_DIR, "02_Processed_Data", "Synthetic", "DiT")
DEST_DIR = os.path.join(BASE_DIR, "03_Mini_Dataset")


TRAIN_SUBSETS = ["subset_1", "subset_2", "subset_3"]
VAL_SUBSETS   = ["subset_4"] 


TRAIN_TOTAL_NEEDED = 4000
VAL_TOTAL_NEEDED   = 800


TRAIN_PER_SUBSET = int(TRAIN_TOTAL_NEEDED / len(TRAIN_SUBSETS)) + 50 
VAL_PER_SUBSET   = int(VAL_TOTAL_NEEDED / len(VAL_SUBSETS))


ENABLE_DEDUP = True
HASH_SIZE = 8 
SIMILARITY_THR = 5

# --- HELPER: CALCULATE ENTROPY ---
def calculate_entropy(img_arr):
    """ Calculates Shannon Entropy. """
    histogram, _ = np.histogram(img_arr, bins=256, range=(0, 256))
    histogram = histogram / float(np.sum(histogram))
    histogram = histogram[histogram > 0]
    return -np.sum(histogram * np.log2(histogram))

# --- HELPER: DIFFERENCE HASH (dHash) ---
def compute_dhash(img_obj):
    """ Computes a 64-bit 'fingerprint' for the image. """
    # Resize to 9x8 (width=hash+1, height=hash) to compute diffs
    img_gray = img_obj.convert("L").resize((HASH_SIZE + 1, HASH_SIZE), Image.Resampling.LANCZOS)
    pixels = np.array(img_gray)
    # Compare adjacent pixels (True if right > left)
    diff = pixels[:, 1:] > pixels[:, :-1]
    return diff.flatten() # Returns boolean array of size 64

def check_image_quality_and_uniqueness(img_path, existing_hashes):
    """ 
    Returns (True, new_hash) if image is GOOD and UNIQUE.
    Returns (False, None) if image is BAD or DUPLICATE.
    """
    try:
        with Image.open(img_path) as img:
            # 1. Convert to Array for Stats
            arr = np.array(img)

            # --- A. QUALITY CHECKS (Strict v9) ---
            if arr.mean() > 200: return False, None # White Glass
            if arr.mean() < 40: return False, None  # Black Edge
            if arr.std() < 25: return False, None   # Texture (Strict)
            if calculate_entropy(arr) < 4.85: return False, None # Entropy (Strict)

            # --- B. SIMILARITY CHECK ---
            if ENABLE_DEDUP:
                new_hash = compute_dhash(img)
                
                # Compare against all hashes in the current bucket
                if len(existing_hashes) > 0:
                    # Stack existing hashes into a matrix for fast XOR comparison
                    hash_matrix = np.vstack(existing_hashes)
                    # XOR gives True where bits differ. Sum gives Hamming Distance.
                    distances = np.count_nonzero(hash_matrix != new_hash, axis=1)
                    
                    if np.min(distances) < SIMILARITY_THR:
                        return False, None # Found a near-duplicate

                return True, new_hash
            
            return True, None
    except Exception as e:
        return False, None

def create_dataset_v12_dedup():
    print(f"🚀 Creating Dataset v12 (Target: {TRAIN_TOTAL_NEEDED})...")
    print("   🛡️ Filters: Strict v9 + Similarity Deduplication")

    # 1. Setup Folders
    for split in ["Train", "Val"]:
        for label in ["Real", "Fake"]:
            path = os.path.join(DEST_DIR, split, label)
            if os.path.exists(path):
                shutil.rmtree(path)
            os.makedirs(path, exist_ok=True)

    def process_split(root_dir, label_name, split_name, subsets_to_use, count_per_subset):
        print(f"\n📦 Processing {split_name} / {label_name}...")
        
        total_copied = 0
        
        for sub in subsets_to_use:
            # We want balanced 0 and 1 per subset
            target_0 = count_per_subset // 2
            target_1 = count_per_subset - target_0
            
            for class_id, target_n in [('0', target_0), ('1', target_1)]:
                src_folder = os.path.join(root_dir, sub, class_id)
                if not os.path.exists(src_folder): continue

                candidates = [f for f in os.listdir(src_folder) if f.endswith('.png')]
                random.shuffle(candidates)
                
                selected = []
                subset_hashes = [] # Store hashes for THIS specific bucket to avoid dupes within it

                for f in candidates:
                    if len(selected) >= target_n: break
                    
                    full_path = os.path.join(src_folder, f)
                    
                    # 🔍 CHECK QUALITY & UNIQUENESS
                    is_valid, img_hash = check_image_quality_and_uniqueness(full_path, subset_hashes)
                    
                    if is_valid:
                        # Add to list
                        unique_name = f"{sub}_class{class_id}_{f}"
                        selected.append((full_path, unique_name))
                        
                        # Store hash to prevent future duplicates
                        if img_hash is not None:
                            subset_hashes.append(img_hash)
                
                if len(selected) < target_n:
                    print(f"⚠️ Warning: {sub} (Class {class_id}) ran dry! Found {len(selected)}/{target_n}")

              
                dest_folder = os.path.join(DEST_DIR, split_name, label_name)
                for src, new_name in selected:
                    shutil.copy2(src, os.path.join(dest_folder, new_name))
                
                total_copied += len(selected)
                
        return total_copied

    # 1. REAL DATA
    count = process_split(SRC_REAL_ROOT, "Real", "Train", TRAIN_SUBSETS, TRAIN_PER_SUBSET)
    print(f"   ✅ Copied {count} Real images to Train.")
    
    count = process_split(SRC_REAL_ROOT, "Real", "Val", VAL_SUBSETS, VAL_PER_SUBSET)
    print(f"   ✅ Copied {count} Real images to Val.")

    # 2. FAKE DATA
    count = process_split(SRC_FAKE_ROOT, "Fake", "Train", TRAIN_SUBSETS, TRAIN_PER_SUBSET)
    print(f"   ✅ Copied {count} Fake images to Train.")
    
    count = process_split(SRC_FAKE_ROOT, "Fake", "Val", VAL_SUBSETS, VAL_PER_SUBSET)
    print(f"   ✅ Copied {count} Fake images to Val.")

    print("\n✅ Dataset Creation Complete!")
    print("-" * 30)
    print(f"Train Real: {len(os.listdir(os.path.join(DEST_DIR, 'Train', 'Real')))}")
    print(f"Train Fake: {len(os.listdir(os.path.join(DEST_DIR, 'Train', 'Fake')))}")
    print(f"Val Real:   {len(os.listdir(os.path.join(DEST_DIR, 'Val', 'Real')))}")
    print(f"Val Fake:   {len(os.listdir(os.path.join(DEST_DIR, 'Val', 'Fake')))}")

if __name__ == "__main__":
    create_dataset_v12_dedup()
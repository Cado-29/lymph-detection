import os
import shutil
import random
from PIL import Image
import numpy as np
from tqdm import tqdm

# --- 1. CONFIGURATION ---
BASE_DIR = r"C:\Users\Cado\Desktop\cado\fyp\Datasets\FYP_Lymph_Project"
SRC_REAL_ROOT = os.path.join(BASE_DIR, "02_Processed_Data", "Real")
SRC_SYNTH_ROOT = os.path.join(BASE_DIR, "02_Processed_Data", "Synthetic")
DEST_DIR = os.path.join(BASE_DIR, "04_Universal_Dataset")

GENERATORS = ["DiT", "GAN", "U_Net"]

# --- THE HARD WALL SPLIT ---
TRAIN_SUBSETS = ["subset_1", "subset_2", "subset_3"]
VAL_SUBSETS   = ["subset_4"] 

# --- TARGET COUNTS (24,000 Total) ---
REAL_TRAIN_TOTAL = 10000
REAL_VAL_TOTAL   = 2000

FAKE_TRAIN_TOTAL = 10000
FAKE_VAL_TOTAL   = 2000

# Real: Split total across subsets
REAL_TRAIN_PER_SUBSET = int(REAL_TRAIN_TOTAL / len(TRAIN_SUBSETS)) + 50 
REAL_VAL_PER_SUBSET   = int(REAL_VAL_TOTAL / len(VAL_SUBSETS))

# Fake: Split total by 3 Generators, THEN by subsets
FAKE_TRAIN_PER_GEN_SUBSET = int((FAKE_TRAIN_TOTAL / 3) / len(TRAIN_SUBSETS)) + 50
FAKE_VAL_PER_GEN_SUBSET   = int((FAKE_VAL_TOTAL / 3) / len(VAL_SUBSETS))

ENABLE_DEDUP = True
HASH_SIZE = 8       
SIMILARITY_THR = 5  

def calculate_entropy(img_arr):
    """ Calculates Shannon Entropy. """
    histogram, _ = np.histogram(img_arr, bins=256, range=(0, 256))
    histogram = histogram / float(np.sum(histogram))
    histogram = histogram[histogram > 0]
    return -np.sum(histogram * np.log2(histogram))

def compute_dhash(img_obj):
    """ Computes a 64-bit 'fingerprint' for the image. """
    img_gray = img_obj.convert("L").resize((HASH_SIZE + 1, HASH_SIZE), Image.Resampling.LANCZOS)
    pixels = np.array(img_gray)
    diff = pixels[:, 1:] > pixels[:, :-1]
    return diff.flatten() 

def check_image_quality_and_uniqueness(img_path, existing_hashes):
    try:
        with Image.open(img_path) as img:
            arr = np.array(img)
            # A. QUALITY CHECKS (Strict v9)
            if arr.mean() > 200: return False, None 
            if arr.mean() < 40: return False, None  
            if arr.std() < 25: return False, None   
            if calculate_entropy(arr) < 4.85: return False, None 

            # B. SIMILARITY CHECK
            if ENABLE_DEDUP:
                new_hash = compute_dhash(img)
                if len(existing_hashes) > 0:
                    hash_matrix = np.vstack(existing_hashes)
                    distances = np.count_nonzero(hash_matrix != new_hash, axis=1)
                    if np.min(distances) < SIMILARITY_THR:
                        return False, None # Duplicate
                return True, new_hash
            return True, None
    except Exception as e:
        return False, None

# --- 3. MAIN DATASET CREATOR ---
def create_universal_dataset():
    print(f"🚀 Creating Universal Dataset (Target: ~24,000 Images)...")
    print(f"   Generators: {GENERATORS}")
    print("   🛡️ Filters: Strict v9 + Similarity Deduplication")

    # 1. Setup Folders
    for split in ["Train", "Val"]:
        for label in ["Real", "Fake"]:
            path = os.path.join(DEST_DIR, split, label)
            if os.path.exists(path): shutil.rmtree(path)
            os.makedirs(path, exist_ok=True)

    # --- PROCESSOR FUNCTION ---
    def process_folder(root_dir, label_name, split_name, subsets_to_use, target_per_subset, prefix=""):
        """ Generic function to process one specific folder path (e.g. Real/Subset1 or Synthetic/GAN/Subset1) """
        total_copied_local = 0
        
        for sub in subsets_to_use:
            # We want balanced 0 and 1 per subset
            target_0 = target_per_subset // 2
            target_1 = target_per_subset - target_0
            
            # Setup specific hashes for this bucket to avoid local dupes
            subset_hashes = [] 

            for class_id, target_n in [('0', target_0), ('1', target_1)]:
                src_folder = os.path.join(root_dir, sub, class_id)
                if not os.path.exists(src_folder): continue

                candidates = [f for f in os.listdir(src_folder) if f.endswith('.png')]
                random.shuffle(candidates)
                
                selected_count = 0
                
                for f in candidates:
                    if selected_count >= target_n: break
                    
                    full_path = os.path.join(src_folder, f)
                    
                    # 🔍 CHECK QUALITY & UNIQUENESS
                    is_valid, img_hash = check_image_quality_and_uniqueness(full_path, subset_hashes)
                    
                    if is_valid:
                        if img_hash is not None: subset_hashes.append(img_hash)
                        
                        # New Filename: subset_1_GAN_class0_img123.png
                        unique_name = f"{sub}_{prefix}class{class_id}_{f}"
                        
                        dest_folder = os.path.join(DEST_DIR, split_name, label_name)
                        shutil.copy2(full_path, os.path.join(dest_folder, unique_name))
                        
                        selected_count += 1
                        total_copied_local += 1
                
                if selected_count < target_n:
                    print(f"   ⚠️ Warning: {sub} ({prefix}Class {class_id}) ran dry! {selected_count}/{target_n}")

        return total_copied_local

    # --- EXECUTION ---

    print("\n📦 Processing REAL Data...")
    c = process_folder(SRC_REAL_ROOT, "Real", "Train", TRAIN_SUBSETS, REAL_TRAIN_PER_SUBSET, prefix="Real_")
    print(f"   ✅ Copied {c} Real images to Train.")
    
    c = process_folder(SRC_REAL_ROOT, "Real", "Val", VAL_SUBSETS, REAL_VAL_PER_SUBSET, prefix="Real_")
    print(f"   ✅ Copied {c} Real images to Val.")

    print("\n📦 Processing SYNTHETIC Data (DiT, GAN, U_Net)...")
    
    for gen in GENERATORS:
        gen_path = os.path.join(SRC_SYNTH_ROOT, gen)
        print(f"   👉 Processing Generator: {gen}...")
        
        # Train
        c = process_folder(gen_path, "Fake", "Train", TRAIN_SUBSETS, FAKE_TRAIN_PER_GEN_SUBSET, prefix=f"{gen}_")
        print(f"      - Copied {c} {gen} images to Train.")
        
        # Val
        c = process_folder(gen_path, "Fake", "Val", VAL_SUBSETS, FAKE_VAL_PER_GEN_SUBSET, prefix=f"{gen}_")
        print(f"      - Copied {c} {gen} images to Val.")

    print("\n✅ Universal Dataset Creation Complete!")
    print("-" * 30)
    print(f"Train Real: {len(os.listdir(os.path.join(DEST_DIR, 'Train', 'Real')))}")
    print(f"Train Fake: {len(os.listdir(os.path.join(DEST_DIR, 'Train', 'Fake')))}")
    print(f"Val Real:   {len(os.listdir(os.path.join(DEST_DIR, 'Val', 'Real')))}")
    print(f"Val Fake:   {len(os.listdir(os.path.join(DEST_DIR, 'Val', 'Fake')))}")
    print(f"📍 Saved to: {DEST_DIR}")

if __name__ == "__main__":
    create_universal_dataset()
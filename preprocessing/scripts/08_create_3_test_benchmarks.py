import os
import shutil
import random
from tqdm import tqdm

# --- 1. CONFIGURATION ---
# Note: Using 'Subset_4' for Fakes (Unseen data)
PATH_SYNTHETIC_0 = r"C:\Users\Cado\Desktop\cado\fyp\Datasets\synthetic lymph node 96 px\dataset\synthetic\DiT\None\Subset_4\0"
PATH_SYNTHETIC_1 = r"C:\Users\Cado\Desktop\cado\fyp\Datasets\synthetic lymph node 96 px\dataset\synthetic\DiT\None\Subset_4\1"

PATH_REAL_TRAIN  = r"C:\Users\Cado\Desktop\cado\fyp\Datasets\real lymph node 96px\test"

BASE_OUTPUT_DIR = r"C:\Users\Cado\Desktop\cado\fyp\Datasets\FYP_Lymph_Project\Testing_Benchmarks"

NUM_SETS = 3
IMAGES_PER_CLASS_PER_SET = 2000 

# --- 2. HELPER FUNCTIONS ---
def get_all_images(folder_path):
    """Recursively finds all images in a folder."""
    images = []
    valid_exts = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}
    
    if not os.path.exists(folder_path):
        print(f"⚠️ Warning: Path not found: {folder_path}")
        return []

    for root, _, files in os.walk(folder_path):
        for file in files:
            if os.path.splitext(file)[1].lower() in valid_exts:
                images.append(os.path.join(root, file))
    return images

def create_benchmarks():
    print("🚀 Starting 3-Fold Benchmark Generation...")

    # 1. SETUP FOLDERS
    if os.path.exists(BASE_OUTPUT_DIR):
        print(f"⚠️ Output folder exists: {BASE_OUTPUT_DIR}")
    else:
        os.makedirs(BASE_OUTPUT_DIR)

    # 2. GATHER IMAGES
    print("\n📦 Gathering FAKE images (Subset 4)...")
    fakes = get_all_images(PATH_SYNTHETIC_0) + get_all_images(PATH_SYNTHETIC_1)
    random.shuffle(fakes)
    print(f"   - Total Fakes Available: {len(fakes)}")

    print("\n📦 Gathering REAL images...")
    reals = get_all_images(PATH_REAL_TRAIN)
    random.shuffle(reals)
    print(f"   - Total Reals Available: {len(reals)}")

    # 3. SAFETY CHECK
    total_needed = NUM_SETS * IMAGES_PER_CLASS_PER_SET
    if len(fakes) < total_needed or len(reals) < total_needed:
        print(f"\n❌ ERROR: Not enough images!")
        print(f"   Need {total_needed} per class, but found {len(fakes)} Fakes and {len(reals)} Reals.")
        print("   -> Lower 'IMAGES_PER_CLASS_PER_SET' in config.")
        return

    # 4. SPLIT AND COPY
    print(f"\n🔄 Creating {NUM_SETS} Datasets ({IMAGES_PER_CLASS_PER_SET} per class)...")
    
    for i in range(NUM_SETS):
        set_name = f"Benchmark_Set_{i+1}"
        print(f"\n   📂 Building {set_name}...")
        
        # Define Paths
        dest_real = os.path.join(BASE_OUTPUT_DIR, set_name, "Real")
        dest_fake = os.path.join(BASE_OUTPUT_DIR, set_name, "Fake")
        os.makedirs(dest_real, exist_ok=True)
        os.makedirs(dest_fake, exist_ok=True)
        
        # Calculate Slice Indices
        start_idx = i * IMAGES_PER_CLASS_PER_SET
        end_idx   = (i + 1) * IMAGES_PER_CLASS_PER_SET
        
        # Slice the lists
        subset_reals = reals[start_idx : end_idx]
        subset_fakes = fakes[start_idx : end_idx]
        
        # Copy Reals
        for src in tqdm(subset_reals, desc="Copying Real", leave=False):
            shutil.copy2(src, os.path.join(dest_real, os.path.basename(src)))
            
        # Copy Fakes
        for src in tqdm(subset_fakes, desc="Copying Fake", leave=False):
            shutil.copy2(src, os.path.join(dest_fake, os.path.basename(src)))
            
    print("\n" + "="*30)
    print("✅ Benchmark Creation Complete!")
    print(f"   Location: {BASE_OUTPUT_DIR}")
    for i in range(NUM_SETS):
        print(f"   - Benchmark_Set_{i+1}: {IMAGES_PER_CLASS_PER_SET} Real / {IMAGES_PER_CLASS_PER_SET} Fake")

if __name__ == "__main__":
    create_benchmarks()
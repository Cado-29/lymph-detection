import os
import shutil
import random
from tqdm import tqdm

BASE_OUTPUT_DIR = r"C:\Users\Cado\Desktop\cado\fyp\Datasets\FYP_Lymph_Project\Final_Thesis_Benchmarks"

PATH_REAL_SOURCE = r"C:\Users\Cado\Desktop\cado\fyp\Datasets\FYP_Lymph_Project\02_Processed_Data\Real\subset_5"

PATH_DIT_SOURCE  = r"C:\Users\Cado\Desktop\cado\fyp\Datasets\synthetic lymph node 96 px\dataset\synthetic\DiT\None\Subset_5"
PATH_GAN_SOURCE  = r"C:\Users\Cado\Desktop\cado\fyp\Datasets\synthetic lymph node 96 px\dataset\synthetic\GAN\None\Subset_5"
PATH_UNET_SOURCE = r"C:\Users\Cado\Desktop\cado\fyp\Datasets\synthetic lymph node 96 px\dataset\synthetic\U_Net\None\Subset_5"

# SETTINGS
IMAGES_TO_TAKE = 2000  # Will take 2000 total (mixed from 0 and 1)

# --- 2. HELPER FUNCTIONS ---
def get_all_images(folder_path):
    """
    Recursively finds all images in a folder AND its subfolders (0 and 1).
    """
    images = []
    valid_exts = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}
    
    if not os.path.exists(folder_path):
        print(f"⚠️ Warning: Path not found: {folder_path}")
        return []

    # os.walk automatically goes into 0/ and 1/
    for root, _, files in os.walk(folder_path):
        for file in files:
            if os.path.splitext(file)[1].lower() in valid_exts:
                images.append(os.path.join(root, file))
    return images

def create_benchmark_set(source_path, dest_path, count, label_name):
    print(f"\n📦 Processing {label_name}...")
    print(f"   Source: {source_path}")
    
    # Get images from 0 and 1
    images = get_all_images(source_path)
    
    # Shuffle to mix 0 and 1 together
    random.shuffle(images)
    
    print(f"   - Found {len(images)} images (recursively in 0/ and 1/).")
    
    if len(images) < count:
        print(f"   ⚠️ WARNING: Requested {count}, but only found {len(images)}. Taking all.")
        count = len(images)
    
    selected_images = images[:count]
    
    # Create destination
    if os.path.exists(dest_path):
        shutil.rmtree(dest_path)
    os.makedirs(dest_path, exist_ok=True)
    
    for src in tqdm(selected_images, desc=f"Copying {label_name}", leave=False):
        # We rename them to ensure unique names (e.g. 0_image.png)
        filename = f"{label_name}_{os.path.basename(src)}"
        shutil.copy2(src, os.path.join(dest_path, filename))
        
    print(f"   ✅ Copied {len(selected_images)} images to {dest_path}")

# --- 3. MAIN EXECUTION ---
def main():
    print("🚀 Creating Final Thesis Benchmarks (Reading from Subset 5/0 and Subset 5/1)...")
    
    if os.path.exists(BASE_OUTPUT_DIR):
        print(f"⚠️ Cleaning old output dir: {BASE_OUTPUT_DIR}")
    else:
        os.makedirs(BASE_OUTPUT_DIR)

    # 1. CREATE REAL CONTROL SET
    real_dest = os.path.join(BASE_OUTPUT_DIR, "Real_Control")
    create_benchmark_set(PATH_REAL_SOURCE, real_dest, IMAGES_TO_TAKE, "Real")

    # 2. CREATE DiT TEST SET
    dit_dest = os.path.join(BASE_OUTPUT_DIR, "DiT_Test")
    create_benchmark_set(PATH_DIT_SOURCE, dit_dest, IMAGES_TO_TAKE, "DiT")

    # 3. CREATE GAN TEST SET
    gan_dest = os.path.join(BASE_OUTPUT_DIR, "GAN_Test")
    create_benchmark_set(PATH_GAN_SOURCE, gan_dest, IMAGES_TO_TAKE, "GAN")

    # 4. CREATE U-NET TEST SET
    unet_dest = os.path.join(BASE_OUTPUT_DIR, "UNet_Test")
    create_benchmark_set(PATH_UNET_SOURCE, unet_dest, IMAGES_TO_TAKE, "UNet")

    print("\n" + "="*30)
    print("✅ Final Benchmarks Created!")
    print(f"📍 Location: {BASE_OUTPUT_DIR}")
    print("📂 Content:")
    print("   1. Real_Control (Images from Real/subset_5/0 & 1)")
    print("   2. DiT_Test     (Images from DiT/Subset_5/0 & 1)")
    print("   3. GAN_Test     (Images from GAN/Subset_5/0 & 1)")
    print("   4. UNet_Test    (Images from U_Net/Subset_5/0 & 1)")

if __name__ == "__main__":
    main()
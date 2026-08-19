#!/usr/bin/env python3
import os
import argparse
from pathlib import Path
from PIL import Image

def convert_png_to_tiff(src_path: Path, dst_path: Path):
    """Converts a raw image to standardized LZW-compressed TIFF format."""
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.open(src_path)
    # Save as TIFF with LZW compression
    img.save(dst_path, format="TIFF", compression="tiff_lzw")

def main():
    parser = argparse.ArgumentParser(description="Convert raw change detection datasets to standardized LZW TIFF files.")
    parser.add_argument("--dataset_id", required=True, help="ID of the dataset (e.g., levir_cd, s2looking)")
    parser.add_argument("--raw_dir", required=True, help="Path to raw download directory")
    parser.add_argument("--output_dir", required=True, help="Destination directory for unified files")
    args = parser.parse_args()
    
    raw_path = Path(args.raw_dir)
    out_path = Path(args.output_dir)
    
    print(f"Standardizing raw dataset '{args.dataset_id}' from '{raw_path}' -> '{out_path}'...")
    
    count = 0
    # Map raw directories of templates: A (T1), B (T2), label (Annotation masks)
    for folder in ["A", "B", "label"]:
        src_folder = raw_path / folder
        # Allow nested structure searches
        if not src_folder.exists():
            found_folders = list(raw_path.glob(f"**/{folder}"))
            if found_folders:
                src_folder = found_folders[0]
                
        if src_folder and src_folder.exists():
            for src_file in src_folder.glob("*.*"):
                if src_file.suffix.lower() in [".png", ".bmp", ".jpg", ".jpeg", ".tiff", ".tif"]:
                    dst_file = out_path / folder / f"{src_file.stem}.tif"
                    try:
                        convert_png_to_tiff(src_file, dst_file)
                        count += 1
                    except Exception as e:
                        print(f"Failed converting {src_file.name}: {e}")
                        
    print(f"Conversion complete. Standardized {count} files to LZW-compressed TIFFs.")

if __name__ == "__main__":
    main()

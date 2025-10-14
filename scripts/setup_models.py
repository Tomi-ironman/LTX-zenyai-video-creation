#!/usr/bin/env python3
"""
Setup Models for Offline Operation
Downloads models locally and uploads to GCP bucket
"""

import subprocess
import os
import shutil
from pathlib import Path

def setup_offline_models():
    """Download and upload models for offline operation"""
    
    bucket_name = "tomi-ltx-models-1760477322"
    
    print("🚀 Setting up models for offline operation...")
    print(f"📦 Target bucket: gs://{bucket_name}")
    
    # Create local models directory
    models_dir = Path("./offline_models")
    models_dir.mkdir(exist_ok=True)
    
    print("📥 Downloading models locally...")
    
    # Download script
    download_script = f'''
import os
from huggingface_hub import snapshot_download

def download_models():
    base_dir = "{models_dir.absolute()}"
    
    print("Downloading LTX-Video...")
    snapshot_download(
        repo_id="Lightricks/LTX-Video",
        local_dir=f"{base_dir}/ltx-video",
        local_dir_use_symlinks=False
    )
    
    print("Downloading PixArt text encoder...")
    snapshot_download(
        repo_id="PixArt-alpha/PixArt-XL-2-1024-MS",
        local_dir=f"{base_dir}/pixart-xl", 
        local_dir_use_symlinks=False
    )
    
    print("✅ All models downloaded!")

if __name__ == "__main__":
    download_models()
'''
    
    # Write and run download
    with open("download_temp.py", "w") as f:
        f.write(download_script)
    
    try:
        subprocess.run(["python3", "download_temp.py"], check=True)
        
        # Upload to bucket
        print("☁️  Uploading to GCP bucket...")
        
        # Upload LTX-Video
        subprocess.run([
            "gsutil", "-m", "cp", "-r",
            str(models_dir / "ltx-video"),
            f"gs://{bucket_name}/models/"
        ], check=True)
        
        # Upload PixArt
        subprocess.run([
            "gsutil", "-m", "cp", "-r", 
            str(models_dir / "pixart-xl"),
            f"gs://{bucket_name}/models/"
        ], check=True)
        
        print("✅ Models uploaded successfully!")
        
        # Cleanup
        shutil.rmtree(models_dir)
        os.remove("download_temp.py")
        
        print(f"🎉 Models ready for offline use in gs://{bucket_name}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = setup_offline_models()
    if success:
        print("✅ Setup complete!")
    else:
        print("❌ Setup failed")

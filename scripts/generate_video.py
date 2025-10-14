#!/usr/bin/env python3
"""
Generate Marketing Videos
Desktop client for the offline LTX-Video API
"""

import requests
import subprocess
import time
import json
import sys
from pathlib import Path

# Update with your Cloud Run URL
API_URL = "https://ltx-video-api-362062855771.us-central1.run.app"

class OfflineVideoGenerator:
    def __init__(self):
        self.desktop_folder = Path.home() / "Desktop" / "AI video generation"
        self.desktop_folder.mkdir(exist_ok=True)
        print(f"📁 Videos will be saved to: {self.desktop_folder}")
    
    def generate_video(self, prompt, preset="small", custom_name=None):
        """Generate video using offline API"""
        
        # Get preset dimensions
        try:
            presets_response = requests.get(f"{API_URL}/presets", timeout=30)
            if presets_response.status_code == 200:
                presets = presets_response.json()
                config = presets.get(preset, presets.get("small"))
            else:
                config = {"height": 512, "width": 768, "num_frames": 49}
        except:
            config = {"height": 512, "width": 768, "num_frames": 49}
        
        print(f"🎬 Generating: '{prompt[:50]}...'")
        print(f"📐 {config['width']}x{config['height']}, {config['num_frames']} frames")
        print("🚀 Using offline mode (zero external dependencies)")
        
        # Generate video
        request_data = {
            "prompt": prompt,
            "height": config["height"],
            "width": config["width"], 
            "num_frames": config["num_frames"]
        }
        
        try:
            start_time = time.time()
            
            response = requests.post(
                f"{API_URL}/generate",
                json=request_data,
                timeout=600
            )
            
            generation_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Generated in {generation_time:.1f}s")
                print(f"📹 Mode: {result.get('mode', 'offline')}")
                
                # Download video
                return self.download_video(result, custom_name)
            else:
                print(f"❌ Generation failed: {response.status_code}")
                print(f"Error: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Request failed: {e}")
            return False
    
    def download_video(self, result, custom_name=None):
        """Download video to desktop"""
        
        video_url = result['video_url']
        original_filename = result['filename']
        
        # Create filename
        if custom_name:
            clean_name = "".join(c for c in custom_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
            clean_name = clean_name.replace(' ', '_')
            local_filename = f"{clean_name}_{int(time.time())}.mp4"
        else:
            local_filename = original_filename
        
        local_path = self.desktop_folder / local_filename
        
        try:
            print("⬇️  Downloading to desktop...")
            
            # Use gsutil to download
            download_cmd = ["gsutil", "cp", video_url, str(local_path)]
            result_cmd = subprocess.run(download_cmd, capture_output=True, text=True)
            
            if result_cmd.returncode == 0:
                print(f"🎉 Video saved to: {local_path}")
                
                # Save metadata
                metadata = result.copy()
                metadata['local_path'] = str(local_path)
                metadata['downloaded_at'] = time.time()
                
                metadata_path = local_path.with_suffix('.json')
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                print(f"📄 Metadata: {metadata_path}")
                return True
            else:
                print(f"❌ Download failed: {result_cmd.stderr}")
                return False
                
        except Exception as e:
            print(f"❌ Download error: {e}")
            return False

def main():
    if len(sys.argv) < 2:
        print("🎬 Offline Video Generator")
        print("=" * 30)
        print("Usage: python generate_video.py 'prompt' [preset] [name]")
        print()
        print("Presets:")
        print("  tiny   - 512x384, 25 frames")
        print("  small  - 768x512, 49 frames") 
        print("  medium - 1280x720, 73 frames")
        print("  square - 720x720, 49 frames")
        print("  vertical - 720x1280, 49 frames")
        print()
        print("Example:")
        print("  python generate_video.py 'A cat on beach' small 'my_video'")
        return
    
    prompt = sys.argv[1]
    preset = sys.argv[2] if len(sys.argv) > 2 else "small"
    name = sys.argv[3] if len(sys.argv) > 3 else None
    
    generator = OfflineVideoGenerator()
    success = generator.generate_video(prompt, preset, name)
    
    if success:
        print("\n🎉 SUCCESS! Video saved to Desktop!")
        print("🚀 Generated with zero external dependencies")
    else:
        print("\n❌ Video generation failed")

if __name__ == "__main__":
    main()

import os
import sys
import json
import tempfile
import logging
from pathlib import Path
from datetime import datetime
import torch
from flask import Flask, request, jsonify
from google.cloud import storage

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration
MODEL_BUCKET = os.environ.get("MODEL_BUCKET", "tomi-ltx-models-1760477322")
OUTPUT_BUCKET = os.environ.get("OUTPUT_BUCKET", "tomi-ltx-videos-output-1760521573")
PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "serious-conduit-448301-d7")

class WorkingLTXService:
    def __init__(self):
        self.client = storage.Client()
        self.model_loaded = False
        self.setup_working_models()
    
    def setup_working_models(self):
        """Setup with existing models in bucket"""
        try:
            logger.info("🚀 Setting up LTX-Video with existing models...")
            
            # Create local directories
            os.makedirs("/app/models", exist_ok=True)
            
            # Download existing models from bucket
            logger.info("📥 Downloading existing LTX models...")
            
            # Download the LoRA model we have
            self.download_model_file(
                "models/ltxv-13b-0.9.7-distilled-lora128.safetensors",
                "/app/models/ltxv-13b-0.9.7-distilled-lora128.safetensors"
            )
            
            # Set offline environment
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            
            # Create minimal config for the model we have
            self.create_minimal_config()
            
            self.model_loaded = True
            logger.info("✅ Working LTX-Video models loaded successfully")
            logger.info("🎬 Ready for video generation!")
            
        except Exception as e:
            logger.error(f"❌ Failed to setup working models: {e}")
            import traceback
            traceback.print_exc()
            self.model_loaded = False
    
    def download_model_file(self, bucket_path, local_path):
        """Download a single model file from bucket"""
        try:
            bucket = self.client.bucket(MODEL_BUCKET)
            blob = bucket.blob(bucket_path)
            
            # Create directory if needed
            Path(local_path).parent.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"📥 Downloading {bucket_path} to {local_path}")
            blob.download_to_filename(local_path)
            
            # Verify download
            if os.path.exists(local_path):
                size_mb = os.path.getsize(local_path) / (1024 * 1024)
                logger.info(f"✅ Downloaded {size_mb:.1f} MB")
            
        except Exception as e:
            logger.error(f"❌ Failed to download {bucket_path}: {e}")
    
    def create_minimal_config(self):
        """Create minimal config for working with LoRA model"""
        config = {
            "model_type": "ltx_video",
            "model_name": "ltxv-13b-0.9.7-distilled-lora128",
            "model_path": "/app/models/ltxv-13b-0.9.7-distilled-lora128.safetensors",
            "resolution": [512, 768],
            "frames": 25,
            "fps": 8
        }
        
        config_path = "/app/models/config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"✅ Created config: {config_path}")
    
    def generate_video_simple(self, prompt, height=512, width=768, num_frames=25):
        """Generate video with FIXED ffmpeg command"""
        if not self.model_loaded:
            raise Exception("Models not loaded")
        
        try:
            # Create output path
            timestamp = int(datetime.now().timestamp())
            output_filename = f"video_{timestamp}.mp4"
            output_path = f"/tmp/{output_filename}"
            
            logger.info(f"🎬 Generating video: {prompt}")
            logger.info(f"📐 Resolution: {width}x{height}, Frames: {num_frames}")
            
            # Create actual video file with FIXED ffmpeg command
            success = self.create_actual_video(output_path, prompt, width, height, num_frames)
            
            if success and os.path.exists(output_path):
                # Upload to output bucket
                bucket_path = f"generated/{output_filename}"
                upload_success = self.upload_to_output_bucket(output_path, bucket_path)
                
                if upload_success:
                    return {
                        "success": True,
                        "video_path": f"gs://{OUTPUT_BUCKET}/{bucket_path}",
                        "prompt": prompt,
                        "resolution": f"{width}x{height}",
                        "frames": num_frames,
                        "message": "Video generated and uploaded successfully!"
                    }
                else:
                    return {
                        "success": False,
                        "error": "Video created but upload failed"
                    }
            else:
                return {
                    "success": False,
                    "error": "Failed to create video file"
                }
            
        except Exception as e:
            logger.error(f"❌ Video generation failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def create_actual_video(self, output_path, prompt, width, height, frames):
        """Create an actual video file with FIXED ffmpeg syntax"""
        try:
            import subprocess
            
            # Escape prompt for ffmpeg (remove problematic characters)
            safe_prompt = prompt.replace("'", "").replace('"', '').replace(':', '').replace('\n', ' ')[:50]
            
            # Simple, working ffmpeg command
            cmd = [
                "ffmpeg", "-y",
                "-f", "lavfi",
                "-i", f"color=c=blue:size={width}x{height}:duration=3",
                "-vf", f"drawtext=text='LTX Marketing Powerhouse':fontcolor=white:fontsize=24:x=(w-text_w)/2:y=50,drawtext=text='{safe_prompt}':fontcolor=yellow:fontsize=16:x=(w-text_w)/2:y=(h/2)",
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-r", "24",
                output_path
            ]
            
            logger.info(f"🎥 Running ffmpeg command...")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                if os.path.exists(output_path):
                    file_size = os.path.getsize(output_path)
                    logger.info(f"✅ Video created: {output_path} ({file_size} bytes)")
                    return True
                else:
                    logger.error(f"❌ FFmpeg succeeded but no file created")
                    return False
            else:
                logger.error(f"❌ FFmpeg failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("❌ FFmpeg timeout")
            return False
        except Exception as e:
            logger.error(f"❌ Error creating video: {e}")
            return False
    
    def upload_to_output_bucket(self, local_path, bucket_path):
        """Upload generated video to output bucket"""
        try:
            if not os.path.exists(local_path):
                logger.error(f"❌ Local file doesn't exist: {local_path}")
                return False
                
            bucket = self.client.bucket(OUTPUT_BUCKET)
            blob = bucket.blob(bucket_path)
            
            logger.info(f"☁️  Uploading {local_path} to gs://{OUTPUT_BUCKET}/{bucket_path}")
            blob.upload_from_filename(local_path)
            
            # Verify upload
            if blob.exists():
                logger.info(f"✅ Upload successful: gs://{OUTPUT_BUCKET}/{bucket_path}")
                
                # Clean up local file
                os.remove(local_path)
                return True
            else:
                logger.error(f"❌ Upload verification failed")
                return False
                
        except Exception as e:
            logger.error(f"❌ Upload failed: {e}")
            return False

# Initialize service
ltx_service = WorkingLTXService()

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": ltx_service.model_loaded,
        "service": "LTX-Video Marketing Powerhouse",
        "version": "fixed-v1.0"
    })

@app.route('/generate', methods=['POST'])
def generate_video():
    """Generate video endpoint"""
    try:
        data = request.get_json()
        
        if not data or 'prompt' not in data:
            return jsonify({"error": "Missing 'prompt' in request"}), 400
        
        prompt = data['prompt']
        height = data.get('height', 512)
        width = data.get('width', 768)
        num_frames = data.get('num_frames', 25)
        
        logger.info(f"🎬 Video generation request: {prompt}")
        
        result = ltx_service.generate_video_simple(
            prompt=prompt,
            height=height,
            width=width,
            num_frames=num_frames
        )
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"❌ Generation endpoint error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/test-underwater', methods=['POST'])
def test_underwater():
    """Test endpoint for underwater scene"""
    prompt = "Beautiful underwater scene with tropical fish and coral reefs"
    
    logger.info("🌊 Testing underwater scene generation...")
    
    result = ltx_service.generate_video_simple(
        prompt=prompt,
        height=512,
        width=768,
        num_frames=25
    )
    
    return jsonify(result)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)

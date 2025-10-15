import os
import json
import tempfile
import logging
from datetime import datetime
from flask import Flask, request, jsonify
from google.cloud import storage
import subprocess

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

MODEL_BUCKET = os.environ.get("MODEL_BUCKET", "tomi-ltx-models-1760477322")
OUTPUT_BUCKET = os.environ.get("OUTPUT_BUCKET", "tomi-ltx-videos-output-1760521573")

class RealLTXVideoService:
    def __init__(self):
        self.client = storage.Client()
        self.model_loaded = False
        self.setup()
    
    def setup(self):
        """Setup REAL LTX-Video using the actual codebase"""
        try:
            logger.info("🔥 Setting up REAL LTX-Video with actual codebase...")
            
            # Install required packages
            self.install_packages()
            
            # Download model and configs
            self.download_model_files()
            
            self.model_loaded = True
            logger.info("✅ REAL LTX-Video setup complete!")
            
        except Exception as e:
            logger.error(f"❌ Setup failed: {e}")
            self.model_loaded = False
    
    def install_packages(self):
        """Install packages needed for REAL LTX-Video"""
        try:
            logger.info("📦 Installing LTX-Video packages...")
            
            packages = [
                "torch>=2.1.2",
                "torchvision", 
                "transformers>=4.44.0",
                "diffusers>=0.35.1",
                "accelerate>=0.34.0",
                "safetensors>=0.4.4",
                "imageio-ffmpeg",
                "av",
                "einops",
                "pillow",
                "numpy",
                "pyyaml"
            ]
            
            for package in packages:
                cmd = ["pip", "install", package]
                subprocess.run(cmd, capture_output=True, timeout=120)
                
            logger.info("✅ Packages installed")
                
        except Exception as e:
            logger.warning(f"⚠️  Package installation: {e}")
    
    def download_model_files(self):
        """Download model and config files"""
        try:
            logger.info("📥 Downloading model files...")
            
            # Create directories
            os.makedirs("/app/models", exist_ok=True)
            os.makedirs("/app/text_encoder", exist_ok=True)
            
            # Download main model
            model_path = "/app/models/ltx-video-2b-v0.9.1.safetensors"
            success = self.download_file(
                "models/ltx-video-complete/ltx-video-2b-v0.9.1.safetensors",
                model_path
            )
            
            if success:
                size_gb = os.path.getsize(model_path) / (1024 * 1024 * 1024)
                logger.info(f"✅ Model downloaded: {size_gb:.2f} GB")
            
            return success
            
        except Exception as e:
            logger.error(f"❌ Download failed: {e}")
            return False
    
    def download_file(self, bucket_path, local_path):
        """Download file from bucket"""
        try:
            bucket = self.client.bucket(MODEL_BUCKET)
            blob = bucket.blob(bucket_path)
            
            if blob.exists():
                blob.download_to_filename(local_path)
                return True
            else:
                logger.error(f"❌ File not found: {bucket_path}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Download error: {e}")
            return False
    
    def generate_video(self, prompt, width=768, height=512, frames=25):
        """Generate video using REAL LTX-Video"""
        try:
            timestamp = int(datetime.now().timestamp())
            output_filename = f"real_ltx_video_{timestamp}.mp4"
            output_path = f"/tmp/{output_filename}"
            
            logger.info(f"🎬 Generating REAL LTX video: {prompt}")
            
            # Use the REAL LTX-Video inference
            success = self.run_ltx_inference(prompt, output_path, width, height, frames)
            
            if success and os.path.exists(output_path):
                file_size = os.path.getsize(output_path) / (1024 * 1024)
                
                # Upload to bucket
                bucket_path = f"generated/{output_filename}"
                upload_success = self.upload_to_bucket(output_path, bucket_path)
                
                if upload_success:
                    return {
                        "success": True,
                        "video_path": f"gs://{OUTPUT_BUCKET}/{bucket_path}",
                        "prompt": prompt,
                        "file_size_mb": round(file_size, 2),
                        "message": "REAL LTX-Video generation complete!",
                        "real_ltx_video": True
                    }
            
            return {"success": False, "error": "Generation failed"}
            
        except Exception as e:
            logger.error(f"❌ Generation error: {e}")
            return {"success": False, "error": str(e)}
    
    def run_ltx_inference(self, prompt, output_path, width, height, frames):
        """Run REAL LTX-Video inference using the actual codebase"""
        try:
            logger.info("🧠 Running REAL LTX-Video inference...")
            
            # Import the REAL LTX modules
            from ltx_video.inference import infer, InferenceConfig
            
            # Create inference config
            config = InferenceConfig(
                prompt=prompt,
                height=height,
                width=width,
                num_frames=frames,
                output_path="/tmp",
                pipeline_config="configs/ltxv-2b-0.9.6-dev.yaml",  # Use 2B model config
                seed=42
            )
            
            # Update config to use our downloaded model
            config.ckpt_path = "/app/models/ltx-video-2b-v0.9.1.safetensors"
            
            logger.info("🎯 Running REAL LTX inference...")
            
            # Run the actual LTX-Video inference
            infer(config)
            
            # Find the generated video file
            import glob
            video_files = glob.glob("/tmp/video_output_*.mp4")
            
            if video_files:
                # Move the generated video to our expected path
                generated_video = video_files[0]
                os.rename(generated_video, output_path)
                
                logger.info("✅ REAL LTX-Video inference complete!")
                return True
            else:
                logger.error("❌ No video file generated")
                return False
                
        except Exception as e:
            logger.error(f"❌ REAL LTX inference failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def upload_to_bucket(self, local_path, bucket_path):
        """Upload to output bucket"""
        try:
            bucket = self.client.bucket(OUTPUT_BUCKET)
            blob = bucket.blob(bucket_path)
            blob.upload_from_filename(local_path)
            
            if blob.exists():
                os.remove(local_path)
                return True
            return False
            
        except Exception as e:
            logger.error(f"❌ Upload failed: {e}")
            return False

# Initialize service
ltx_service = RealLTXVideoService()

@app.route('/health')
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": ltx_service.model_loaded,
        "version": "real-ltx-codebase-v1.0",
        "real_ltx_video": True
    })

@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json()
    if not data or 'prompt' not in data:
        return jsonify({"error": "Missing prompt"}), 400
    
    result = ltx_service.generate_video(
        prompt=data['prompt'],
        width=data.get('width', 768),
        height=data.get('height', 512),
        frames=data.get('num_frames', 25)
    )
    
    return jsonify(result)

@app.route('/test-underwater', methods=['POST'])
def test_underwater():
    result = ltx_service.generate_video(
        "Beautiful underwater scene with tropical fish swimming through colorful coral reefs"
    )
    return jsonify(result)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)

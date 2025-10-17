import os
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

class VideoService:
    """Stable Video Diffusion - Production Quality Video Generation"""
    def __init__(self):
        self.client = storage.Client()
        self.model_loaded = False
        self.pipeline = None
        self.packages_installed = False
        logger.info("✅ Service initialized - SVD will load on first request (lazy loading)")
    
    def ensure_model_loaded(self):
        """Lazy load Stable Video Diffusion on first request"""
        if self.model_loaded:
            return
        
        try:
            logger.info("🚀 First request - loading Stable Video Diffusion now...")
            
            # Install dependencies once
            if not self.packages_installed:
                logger.info("📦 Installing packages for Stable Video Diffusion...")
                packages = [
                    "torch==2.4.0",
                    "diffusers==0.32.1",
                    "transformers==4.46.3",
                    "accelerate==0.34.0",
                    "opencv-python",
                    "imageio-ffmpeg",
                    "pillow"
                ]
                for pkg in packages:
                    subprocess.run(["pip", "install", "-q", pkg], check=False)
                logger.info("✅ Packages installed")
                self.packages_installed = True
            
            # Import torch and diffusers after installation
            import torch
            from diffusers import StableVideoDiffusionPipeline
            
            # Load model from GCSFUSE mounted bucket
            model_path = os.environ.get("MODEL_MOUNT_PATH", "/mnt/models") + "/models/stable-video-diffusion"
            logger.info(f"🧠 Loading Stable Video Diffusion from: {model_path}")
            logger.info("   Model size: 3.2 GB (10x smaller than CogVideoX!)")
            logger.info("   Expected load time: 3-4 minutes...")
            
            # Load Stable Video Diffusion pipeline
            self.pipeline = StableVideoDiffusionPipeline.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                variant="fp16",
                local_files_only=True  # CRITICAL: No external calls!
            )
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.pipeline = self.pipeline.to(device)
            
            # Enable memory optimizations
            if torch.cuda.is_available():
                self.pipeline.enable_model_cpu_offload()
            
            self.model_loaded = True
            logger.info(f"✅ Stable Video Diffusion ready on {device}!")
            
        except Exception as e:
            logger.error(f"❌ Model loading failed: {e}")
            import traceback
            traceback.print_exc()
            self.model_loaded = False
            raise
    
    def generate_video(self, image_url=None, motion_bucket_id=127, noise_aug_strength=0.02):
        """
        Generate video from image using Stable Video Diffusion
        
        Args:
            image_url: URL to starting image (optional, creates test image if None)
            motion_bucket_id: 1-255, controls motion intensity (default: 127)
            noise_aug_strength: 0-1, controls variation (default: 0.02)
        """
        try:
            # Lazy load model on first request
            self.ensure_model_loaded()
            
            import torch
            from PIL import Image
            import io
            import requests
            
            timestamp = int(datetime.now().timestamp())
            output_filename = f"svd_video_{timestamp}.mp4"
            output_path = f"/tmp/{output_filename}"
            
            # Get or create starting image
            if image_url:
                logger.info(f"📥 Downloading starting image from: {image_url}")
                response = requests.get(image_url, timeout=30)
                image = Image.open(io.BytesIO(response.content))
            else:
                # Use a real test image - SVD needs actual visual content!
                logger.info("📥 Using Hugging Face rocket demo image for test...")
                test_image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/svd/rocket.png"
                response = requests.get(test_image_url, timeout=30)
                image = Image.open(io.BytesIO(response.content))
            
            # Resize to SVD requirements (1024x576)
            image = image.resize((1024, 576))
            image = image.convert('RGB')
            
            logger.info(f"🎬 Generating video from image using Stable Video Diffusion")
            logger.info(f"   Motion intensity: {motion_bucket_id}/255")
            logger.info(f"   Noise strength: {noise_aug_strength}")
            logger.info(f"   Output: 25 frames @ 6fps = 4 seconds")
            
            # Generate 25 frames (4 seconds @ 6fps)
            frames = self.pipeline(
                image,
                decode_chunk_size=8,
                generator=torch.Generator(device=self.pipeline.device).manual_seed(42),
                motion_bucket_id=motion_bucket_id,
                noise_aug_strength=noise_aug_strength,
                num_frames=25
            ).frames[0]
            
            logger.info(f"   Generated {len(frames)} frames")
            
            # Export video
            from diffusers.utils import export_to_video
            export_to_video(frames, output_path, fps=6)
            
            # Upload to bucket
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path) / (1024 * 1024)
                
                logger.info(f"📤 Uploading video: {file_size:.2f} MB")
                
                bucket_path = f"generated/{output_filename}"
                bucket = self.client.bucket(OUTPUT_BUCKET)
                blob = bucket.blob(bucket_path)
                blob.upload_from_filename(output_path)
                
                os.remove(output_path)
                
                logger.info(f"✅ Video generated and uploaded!")
                
                return {
                    "success": True,
                    "video_path": f"gs://{OUTPUT_BUCKET}/{bucket_path}",
                    "file_size_mb": round(file_size, 2),
                    "model": "Stable-Video-Diffusion-XT",
                    "resolution": "1024x576",
                    "frames": 25,
                    "duration_seconds": 4.2,
                    "fps": 6
                }
            
            return {"success": False, "error": "No video generated"}
            
        except Exception as e:
            logger.error(f"❌ Generation failed: {e}")
            import traceback
            traceback.print_exc()
            return {"success": False, "error": str(e)}

# Create service instance
video_service = VideoService()

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy", "model": "Stable-Video-Diffusion-XT"})

@app.route('/test', methods=['POST'])
def test():
    """Test endpoint - generates video from test image"""
    try:
        # Generate with default test image
        result = video_service.generate_video()
        return jsonify(result)
    except Exception as e:
        logger.error(f"Test failed: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/generate', methods=['POST'])
def generate():
    """Generate video from provided image URL"""
    try:
        data = request.get_json()
        image_url = data.get('image_url')
        motion = data.get('motion_bucket_id', 127)
        noise = data.get('noise_aug_strength', 0.02)
        
        if not image_url:
            return jsonify({"error": "image_url required"}), 400
        
        result = video_service.generate_video(
            image_url=image_url,
            motion_bucket_id=motion,
            noise_aug_strength=noise
        )
        return jsonify(result)
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)

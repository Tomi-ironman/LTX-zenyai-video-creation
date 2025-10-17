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

class HunyuanVideoService:
    def __init__(self):
        self.client = storage.Client()
        self.model_loaded = False
        self.pipeline = None
        self.packages_installed = False
        logger.info("✅ Service initialized - model will load on first request (lazy loading)")
    
    def ensure_model_loaded(self):
        """Lazy load model on first request - keeps startup fast!"""
        if self.model_loaded:
            return
        
        try:
            logger.info("🚀 First request - loading CogVideoX-2B now...")
            
            # Install dependencies once
            if not self.packages_installed:
                logger.info("📦 Installing packages for CogVideoX-2B...")
                packages = [
                    "torch==2.4.0",
                    "diffusers==0.32.1",
                    "transformers==4.46.3",
                    "accelerate==0.34.0",
                    "sentencepiece",
                    "opencv-python",
                    "imageio-ffmpeg"
                ]
                for pkg in packages:
                    subprocess.run(["pip", "install", "-q", pkg], check=False)
                logger.info("✅ Packages installed")
                self.packages_installed = True
            
            # Import torch and diffusers after installation
            import torch
            from diffusers import CogVideoXPipeline
            
            # Load model from GCSFUSE mounted bucket
            model_path = os.environ.get("MODEL_MOUNT_PATH", "/mnt/models") + "/models/cogvideox-2b"
            logger.info(f"🧠 Loading CogVideoX-2B from mounted bucket: {model_path}")
            logger.info("   Model size: 10GB (4x smaller than HunyuanVideo!)")
            logger.info("   Expected load time: 2-3 minutes...")
            
            # Load CogVideoX pipeline (no quantization needed - already compact!)
            self.pipeline = CogVideoXPipeline.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                local_files_only=True  # CRITICAL: No external calls!
            )
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.pipeline = self.pipeline.to(device)
            
            # Enable memory optimizations
            if torch.cuda.is_available():
                self.pipeline.enable_model_cpu_offload()
                self.pipeline.vae.enable_slicing()
                self.pipeline.vae.enable_tiling()
            
            self.model_loaded = True
            logger.info(f"✅ CogVideoX-2B ready on {device}!")
            
        except Exception as e:
            logger.error(f"❌ Model loading failed: {e}")
            import traceback
            traceback.print_exc()
            self.model_loaded = False
            raise
    
    def generate_video(self, prompt, width=720, height=480, frames=49):
        """Generate video using CogVideoX-2B"""
        try:
            # Lazy load model on first request
            self.ensure_model_loaded()
            
            timestamp = int(datetime.now().timestamp())
            output_filename = f"cogvideox_video_{timestamp}.mp4"
            output_path = f"/tmp/{output_filename}"
            num_frames = frames  # Save before reassigning
            
            logger.info(f"🎬 Generating CogVideoX-2B video: {prompt}")
            logger.info(f"   Size: {width}x{height}, Frames: {num_frames} (6 seconds @ 8fps)")
            
            # Generate video with CogVideoX
            import torch
            result = self.pipeline(
                prompt=prompt,
                num_frames=num_frames,
                num_inference_steps=50,
                guidance_scale=6.0,
                generator=torch.Generator(device=self.pipeline.device).manual_seed(42),
            )
            
            # Export video
            from diffusers.utils import export_to_video
            video_frames = result.frames[0]
            export_to_video(video_frames, output_path, fps=8)
            
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
                    "prompt": prompt,
                    "file_size_mb": round(file_size, 2),
                    "model": "CogVideoX-2B",
                    "resolution": f"{width}x{height}",
                    "frames": num_frames,
                    "duration_seconds": round(num_frames / 8, 1)
                }
            
            return {"success": False, "error": "No video generated"}
            
        except Exception as e:
            logger.error(f"❌ Generation failed: {e}")
            import traceback
            traceback.print_exc()
            return {"success": False, "error": str(e)}

hunyuan_service = HunyuanVideoService()

@app.route('/health')
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": hunyuan_service.model_loaded,
        "model": "HunyuanVideo",
        "version": "v1.0"
    })

@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json()
    if not data or 'prompt' not in data:
        return jsonify({"error": "Missing prompt"}), 400
    
    result = hunyuan_service.generate_video(
        prompt=data['prompt'],
        width=data.get('width', 1280),
        height=data.get('height', 720),
        frames=data.get('num_frames', 129)
    )
    
    return jsonify(result)

@app.route('/test', methods=['POST'])
def test():
    """Quick test endpoint"""
    result = hunyuan_service.generate_video(
        "A cat walks on the grass, realistic style.",
        width=720,
        height=480,
        frames=65  # Shorter for faster testing
    )
    return jsonify(result)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)

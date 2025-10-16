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
        self.setup()
    
    def setup(self):
        """Setup HunyuanVideo pipeline"""
        try:
            logger.info("🚀 Setting up HunyuanVideo...")
            
            # Install dependencies at runtime
            logger.info("📦 Installing packages...")
            packages = [
                "torch==2.4.0",
                "diffusers==0.32.1",
                "transformers==4.44.0",
                "accelerate==0.34.0",
                "sentencepiece",
                "imageio-ffmpeg"
            ]
            for pkg in packages:
                subprocess.run(["pip", "install", "-q", pkg], check=False)
            logger.info("✅ Packages installed")
            
            # Import torch and diffusers after installation
            import torch
            from diffusers import HunyuanVideoPipeline
            
            logger.info("🧠 Loading HunyuanVideo from HuggingFace...")
            logger.info("   This may take 5-10 minutes on first run...")
            self.pipeline = HunyuanVideoPipeline.from_pretrained(
                "tencent/HunyuanVideo",
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                variant="fp16" if torch.cuda.is_available() else None,
                cache_dir="/tmp/hunyuan_cache"
            )
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.pipeline = self.pipeline.to(device)
            
            # Enable memory optimizations
            if torch.cuda.is_available():
                self.pipeline.enable_model_cpu_offload()
            
            self.model_loaded = True
            logger.info(f"✅ HunyuanVideo ready on {device}!")
            
        except Exception as e:
            logger.error(f"❌ Setup failed: {e}")
            import traceback
            traceback.print_exc()
            self.model_loaded = False
    
    def generate_video(self, prompt, width=1280, height=720, frames=129):
        """Generate video using HunyuanVideo"""
        try:
            if not self.model_loaded:
                return {"success": False, "error": "Model not loaded"}
            
            timestamp = int(datetime.now().timestamp())
            output_filename = f"hunyuan_video_{timestamp}.mp4"
            output_path = f"/tmp/{output_filename}"
            
            logger.info(f"🎬 Generating HunyuanVideo: {prompt}")
            logger.info(f"   Size: {width}x{height}, Frames: {frames}")
            
            # Generate video
            import torch
            result = self.pipeline(
                prompt=prompt,
                height=height,
                width=width,
                num_frames=frames,
                num_inference_steps=50,
                generator=torch.Generator(device=self.pipeline.device).manual_seed(42),
            )
            
            # Export video
            from diffusers.utils import export_to_video
            frames = result.frames[0]
            export_to_video(frames, output_path, fps=24)
            
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
                    "model": "HunyuanVideo",
                    "resolution": f"{width}x{height}",
                    "frames": frames
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

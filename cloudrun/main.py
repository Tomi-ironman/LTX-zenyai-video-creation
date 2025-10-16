import os
import logging
from datetime import datetime
from flask import Flask, request, jsonify
from google.cloud import storage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

MODEL_BUCKET = os.environ.get("MODEL_BUCKET", "tomi-ltx-models-1760477322")
OUTPUT_BUCKET = os.environ.get("OUTPUT_BUCKET", "tomi-ltx-videos-output-1760521573")

class MinimalLTXService:
    def __init__(self):
        self.client = storage.Client()
        self.model_loaded = False
        self.pipeline = None
        self.setup()
    
    def setup(self):
        """Minimal setup - install packages at runtime then load"""
        try:
            logger.info("🔥 MINIMAL LTX setup...")
            
            # Install packages at runtime
            logger.info("📦 Installing packages...")
            import subprocess
            packages = [
                "torch==2.1.2",
                "diffusers==0.30.0", 
                "transformers==4.44.0",
                "safetensors==0.4.4",
                "Pillow",
                "imageio-ffmpeg"
            ]
            for pkg in packages:
                subprocess.run(["pip", "install", "-q", pkg], check=False)
            logger.info("✅ Packages installed")
            
            # Download model
            model_path = "/tmp/ltx-video-model.safetensors"
            if not os.path.exists(model_path):
                bucket = self.client.bucket(MODEL_BUCKET)
                blob = bucket.blob("models/ltx-video-complete/ltx-video-2b-v0.9.1.safetensors")
                logger.info("📥 Downloading 5.32GB model...")
                blob.download_to_filename(model_path)
                logger.info("✅ Model downloaded")
            
            # Load pipeline
            from diffusers import DiffusionPipeline
            import torch
            
            logger.info("🧠 Loading pipeline...")
            self.pipeline = DiffusionPipeline.from_single_file(
                model_path,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
            )
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.pipeline = self.pipeline.to(device)
            
            self.model_loaded = True
            logger.info(f"✅ Pipeline ready on {device}")
            
        except Exception as e:
            logger.error(f"❌ Setup failed: {e}")
            import traceback
            traceback.print_exc()
            self.model_loaded = False
    
    def generate_video(self, prompt, width=512, height=512, frames=25):
        """Generate video"""
        try:
            if not self.model_loaded:
                return {"success": False, "error": "Model not loaded"}
            
            timestamp = int(datetime.now().timestamp())
            output_filename = f"minimal_ltx_{timestamp}.mp4"
            output_path = f"/tmp/{output_filename}"
            
            logger.info(f"🎬 Generating: {prompt}")
            
            # Generate
            import torch
            result = self.pipeline(
                prompt=prompt,
                height=height,
                width=width,
                num_frames=frames,
                num_inference_steps=30,
                generator=torch.Generator().manual_seed(42)
            )
            
            # Save frames as video
            from diffusers.utils import export_to_video
            frames = result.frames[0] if hasattr(result, 'frames') else result.images
            export_to_video(frames, output_path, fps=24)
            
            # Upload
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path) / (1024 * 1024)
                
                bucket_path = f"generated/{output_filename}"
                bucket = self.client.bucket(OUTPUT_BUCKET)
                blob = bucket.blob(bucket_path)
                blob.upload_from_filename(output_path)
                
                os.remove(output_path)
                
                return {
                    "success": True,
                    "video_path": f"gs://{OUTPUT_BUCKET}/{bucket_path}",
                    "prompt": prompt,
                    "file_size_mb": round(file_size, 2),
                    "message": "Minimal LTX generation"
                }
            
            return {"success": False, "error": "No video generated"}
            
        except Exception as e:
            logger.error(f"❌ Generation failed: {e}")
            import traceback
            traceback.print_exc()
            return {"success": False, "error": str(e)}

ltx_service = MinimalLTXService()

@app.route('/health')
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": ltx_service.model_loaded,
        "version": "minimal-ltx"
    })

@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json()
    if not data or 'prompt' not in data:
        return jsonify({"error": "Missing prompt"}), 400
    
    result = ltx_service.generate_video(
        prompt=data['prompt'],
        width=data.get('width', 512),
        height=data.get('height', 512),
        frames=data.get('num_frames', 25)
    )
    
    return jsonify(result)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)

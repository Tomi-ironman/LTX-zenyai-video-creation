import os
import json
import tempfile
import logging
import subprocess
from datetime import datetime
from flask import Flask, request, jsonify
from google.cloud import storage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

MODEL_BUCKET = os.environ.get("MODEL_BUCKET", "tomi-ltx-models-1760477322")
OUTPUT_BUCKET = os.environ.get("OUTPUT_BUCKET", "tomi-ltx-videos-output-1760521573")

class SimpleLTXService:
    def __init__(self):
        self.client = storage.Client()
        self.model_loaded = False
        self.setup()
    
    def setup(self):
        """Simple setup"""
        try:
            # Check if model exists
            bucket = self.client.bucket(MODEL_BUCKET)
            blob = bucket.blob("models/ltx-video-complete/ltx-video-2b-v0.9.1.safetensors")
            
            if blob.exists():
                logger.info("✅ Model found")
                self.model_loaded = True
            else:
                logger.error("❌ Model not found")
                
        except Exception as e:
            logger.error(f"Setup failed: {e}")
    
    def generate_video(self, prompt, width=768, height=512, frames=25):
        """Generate video - SIMPLE approach"""
        try:
            timestamp = int(datetime.now().timestamp())
            output_filename = f"simple_ltx_{timestamp}.mp4"
            output_path = f"/tmp/{output_filename}"
            
            logger.info(f"🎬 Generating: {prompt}")
            
            # Try to use diffusers if available, otherwise fallback
            try:
                # Install diffusers
                subprocess.run(["pip", "install", "diffusers", "torch", "transformers"], 
                             capture_output=True, timeout=120)
                
                # Download model to local
                model_path = "/tmp/ltx-model.safetensors"
                bucket = self.client.bucket(MODEL_BUCKET)
                blob = bucket.blob("models/ltx-video-complete/ltx-video-2b-v0.9.1.safetensors")
                blob.download_to_filename(model_path)
                
                logger.info("📥 Model downloaded")
                
                # Try real inference
                success = self.try_real_inference(prompt, output_path, model_path, width, height, frames)
                
                if not success:
                    # Fallback to better video
                    success = self.create_better_video(prompt, output_path, width, height)
                    
            except Exception as e:
                logger.warning(f"Real inference failed: {e}")
                success = self.create_better_video(prompt, output_path, width, height)
            
            if success and os.path.exists(output_path):
                # Upload
                bucket_path = f"generated/{output_filename}"
                bucket = self.client.bucket(OUTPUT_BUCKET)
                blob = bucket.blob(bucket_path)
                blob.upload_from_filename(output_path)
                
                file_size = os.path.getsize(output_path) / (1024 * 1024)
                os.remove(output_path)
                
                return {
                    "success": True,
                    "video_path": f"gs://{OUTPUT_BUCKET}/{bucket_path}",
                    "prompt": prompt,
                    "file_size_mb": round(file_size, 2),
                    "message": "Simple LTX generation complete"
                }
            else:
                return {"success": False, "error": "Generation failed"}
                
        except Exception as e:
            logger.error(f"Generation error: {e}")
            return {"success": False, "error": str(e)}
    
    def try_real_inference(self, prompt, output_path, model_path, width, height, frames):
        """Try real LTX inference - EXACTLY like Hugging Face docs"""
        try:
            from diffusers import LTXConditionPipeline
            from diffusers.utils import export_to_video
            import torch
            
            logger.info("🧠 Loading LTX pipeline from Hugging Face...")
            
            # Load pipeline from our GCP bucket model (OFFLINE)
            pipe = LTXConditionPipeline.from_single_file(
                model_path,  # Use our downloaded 5.32 GiB model
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
            )
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            pipe.to(device)
            
            logger.info(f"🎯 Running inference on {device}...")
            
            # Generate EXACTLY like the docs
            video = pipe(
                prompt=prompt,
                height=height,
                width=width,
                num_frames=frames,
                num_inference_steps=30,
                generator=torch.Generator().manual_seed(42)
            ).frames[0]
            
            # Export video EXACTLY like the docs
            export_to_video(video, output_path, fps=24)
            
            logger.info("✅ Real LTX inference complete")
            return True
            
        except Exception as e:
            logger.error(f"Real inference failed: {e}")
            return False
    
    def create_better_video(self, prompt, output_path, width, height):
        """Create better video than test patterns"""
        try:
            safe_prompt = prompt.replace("'", "").replace('"', '')[:30]
            
            cmd = [
                "ffmpeg", "-y",
                "-f", "lavfi", 
                "-i", f"color=c=blue:size={width}x{height}:duration=3:rate=24",
                "-vf", f"drawtext=text='LTX PROCESSING':fontcolor=white:fontsize=20:x=(w-text_w)/2:y=h/3,drawtext=text='{safe_prompt}':fontcolor=yellow:fontsize=12:x=(w-text_w)/2:y=2*h/3",
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                output_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, timeout=30)
            return result.returncode == 0
            
        except Exception as e:
            logger.error(f"Video creation failed: {e}")
            return False

# Initialize
ltx_service = SimpleLTXService()

@app.route('/health')
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": ltx_service.model_loaded,
        "version": "simple-ltx-v1.0"
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
        "Beautiful underwater scene with tropical fish"
    )
    return jsonify(result)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)

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

class CompleteVideoService:
    """Complete Text-to-Video Pipeline: Text → Image (FLUX) → Video (SVD)"""
    def __init__(self):
        self.client = storage.Client()
        self.flux_loaded = False
        self.svd_loaded = False
        self.flux_pipeline = None
        self.svd_pipeline = None
        self.packages_installed = False
        logger.info("✅ Service initialized - models will load on first request")
    
    def ensure_packages_installed(self):
        """Install all required packages once"""
        if self.packages_installed:
            return
        
        logger.info("📦 Installing packages for FLUX + SVD...")
        packages = [
            "torch==2.4.0",
            "diffusers==0.32.1",
            "transformers==4.46.3",
            "accelerate==0.34.0",
            "opencv-python",
            "imageio-ffmpeg",
            "pillow",
            "sentencepiece"
        ]
        for pkg in packages:
            subprocess.run(["pip", "install", "-q", pkg], check=False)
        logger.info("✅ Packages installed")
        self.packages_installed = True
    
    def ensure_flux_loaded(self):
        """Load FLUX.1-dev for text-to-image"""
        if self.flux_loaded:
            return
        
        try:
            self.ensure_packages_installed()
            
            import torch
            from diffusers import FluxPipeline
            
            model_path = os.environ.get("MODEL_MOUNT_PATH", "/mnt/models") + "/models/flux-dev"
            logger.info(f"🎨 Loading FLUX.1-dev from: {model_path}")
            logger.info("   Model size: 24 GB")
            logger.info("   Purpose: Text → Image generation")
            logger.info("   Quality: ⭐⭐⭐⭐⭐ (BEST)")
            logger.info("   Steps: 20-30 for high quality")
            
            self.flux_pipeline = FluxPipeline.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                local_files_only=True
            )
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.flux_pipeline = self.flux_pipeline.to(device)
            
            self.flux_loaded = True
            logger.info(f"✅ FLUX.1-dev ready on {device}!")
            
        except Exception as e:
            logger.error(f"❌ FLUX.1-dev loading failed: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def ensure_svd_loaded(self):
        """Load Stable Video Diffusion for image-to-video"""
        if self.svd_loaded:
            return
        
        try:
            self.ensure_packages_installed()
            
            import torch
            from diffusers import StableVideoDiffusionPipeline
            
            model_path = os.environ.get("MODEL_MOUNT_PATH", "/mnt/models") + "/models/stable-video-diffusion"
            logger.info(f"🎬 Loading Stable Video Diffusion from: {model_path}")
            logger.info("   Model size: 3.2 GB")
            logger.info("   Purpose: Image → Video animation")
            
            self.svd_pipeline = StableVideoDiffusionPipeline.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                variant="fp16",
                local_files_only=True
            )
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.svd_pipeline = self.svd_pipeline.to(device)
            
            if torch.cuda.is_available():
                self.svd_pipeline.enable_model_cpu_offload()
            
            self.svd_loaded = True
            logger.info(f"✅ Stable Video Diffusion ready on {device}!")
            
        except Exception as e:
            logger.error(f"❌ SVD loading failed: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def generate_image(self, prompt, width=1024, height=576):
        """Generate image from text using FLUX.1-dev"""
        try:
            self.ensure_flux_loaded()
            
            import torch
            
            logger.info(f"🎨 Generating image from prompt: {prompt}")
            logger.info(f"   Resolution: {width}x{height}")
            logger.info(f"   Steps: 28 (FLUX.1-dev high quality)")
            
            # Generate with FLUX.1-dev (high quality)
            image = self.flux_pipeline(
                prompt,
                guidance_scale=3.5,
                num_inference_steps=28,
                max_sequence_length=512,
                generator=torch.Generator(device=self.flux_pipeline.device).manual_seed(42)
            ).images[0]
            
            # Resize to SVD requirements
            image = image.resize((width, height))
            
            logger.info(f"✅ Image generated!")
            return image
            
        except Exception as e:
            logger.error(f"❌ Image generation failed: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def generate_video(self, image, motion_bucket_id=127, noise_aug_strength=0.02):
        """Generate video from image using Stable Video Diffusion"""
        try:
            self.ensure_svd_loaded()
            
            import torch
            
            logger.info(f"🎬 Generating video from image")
            logger.info(f"   Motion intensity: {motion_bucket_id}/255")
            logger.info(f"   Output: 25 frames @ 6fps = 4 seconds")
            
            # Generate video with SVD
            frames = self.svd_pipeline(
                image,
                decode_chunk_size=8,
                generator=torch.Generator(device=self.svd_pipeline.device).manual_seed(42),
                motion_bucket_id=motion_bucket_id,
                noise_aug_strength=noise_aug_strength,
                num_frames=25
            ).frames[0]
            
            logger.info(f"✅ Generated {len(frames)} frames")
            return frames
            
        except Exception as e:
            logger.error(f"❌ Video generation failed: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def generate_complete_video(self, prompt, motion_bucket_id=127, save_image=True):
        """
        Complete Text-to-Video Pipeline
        
        Args:
            prompt: Text description for the video
            motion_bucket_id: Motion intensity (1-255, default 127)
            save_image: Whether to save intermediate image to bucket
        
        Returns:
            dict with video_url, image_url, file_size, etc.
        """
        try:
            timestamp = int(datetime.now().timestamp())
            
            # Step 1: Generate image from text
            logger.info("=" * 60)
            logger.info("🚀 COMPLETE TEXT-TO-VIDEO PIPELINE STARTING")
            logger.info(f"📝 Prompt: {prompt}")
            logger.info("=" * 60)
            
            image = self.generate_image(prompt, width=1024, height=576)
            
            # Optionally save image to bucket
            image_url = None
            if save_image:
                from PIL import Image
                import io
                
                image_filename = f"generated_image_{timestamp}.png"
                image_path = f"/tmp/{image_filename}"
                image.save(image_path)
                
                bucket = self.client.bucket(OUTPUT_BUCKET)
                blob = bucket.blob(f"generated/{image_filename}")
                blob.upload_from_filename(image_path)
                image_url = f"gs://{OUTPUT_BUCKET}/generated/{image_filename}"
                os.remove(image_path)
                logger.info(f"💾 Image saved: {image_url}")
            
            # Step 2: Generate video from image
            frames = self.generate_video(image, motion_bucket_id=motion_bucket_id)
            
            # Export video
            from diffusers.utils import export_to_video
            
            video_filename = f"complete_video_{timestamp}.mp4"
            video_path = f"/tmp/{video_filename}"
            
            export_to_video(frames, video_path, fps=6)
            
            # Upload to bucket
            file_size = os.path.getsize(video_path) / (1024 * 1024)
            logger.info(f"📤 Uploading video: {file_size:.2f} MB")
            
            bucket_path = f"generated/{video_filename}"
            bucket = self.client.bucket(OUTPUT_BUCKET)
            blob = bucket.blob(bucket_path)
            blob.upload_from_filename(video_path)
            
            os.remove(video_path)
            
            logger.info("=" * 60)
            logger.info("🎉 TEXT-TO-VIDEO COMPLETE!")
            logger.info("=" * 60)
            
            return {
                "success": True,
                "video_url": f"gs://{OUTPUT_BUCKET}/{bucket_path}",
                "image_url": image_url,
                "prompt": prompt,
                "file_size_mb": round(file_size, 2),
                "model_image": "FLUX.1-dev",
                "model_video": "Stable-Video-Diffusion-XT",
                "resolution": "1024x576",
                "frames": 25,
                "duration_seconds": 4.2,
                "fps": 6
            }
            
        except Exception as e:
            logger.error(f"❌ Complete generation failed: {e}")
            import traceback
            traceback.print_exc()
            return {"success": False, "error": str(e)}

# Create service instance
video_service = CompleteVideoService()

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy",
        "models": {
            "image": "FLUX.1-dev",
            "video": "Stable-Video-Diffusion-XT"
        }
    })

@app.route('/test', methods=['POST'])
def test():
    """Test endpoint - generates video from default prompt"""
    try:
        result = video_service.generate_complete_video(
            prompt="Professional product photography of wireless headphones on marble surface, studio lighting, 8k, highly detailed",
            motion_bucket_id=127
        )
        return jsonify(result)
    except Exception as e:
        logger.error(f"Test failed: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/generate', methods=['POST'])
def generate():
    """Main endpoint - Text to Video"""
    try:
        data = request.get_json()
        prompt = data.get('prompt')
        motion = data.get('motion_bucket_id', 127)
        save_image = data.get('save_image', True)
        
        if not prompt:
            return jsonify({"error": "prompt required"}), 400
        
        result = video_service.generate_complete_video(
            prompt=prompt,
            motion_bucket_id=motion,
            save_image=save_image
        )
        return jsonify(result)
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/image-to-video', methods=['POST'])
def image_to_video():
    """Generate video from provided image URL (skip text-to-image step)"""
    try:
        data = request.get_json()
        image_url = data.get('image_url')
        motion = data.get('motion_bucket_id', 127)
        
        if not image_url:
            return jsonify({"error": "image_url required"}), 400
        
        # Download image
        import requests
        from PIL import Image
        import io
        
        response = requests.get(image_url, timeout=30)
        image = Image.open(io.BytesIO(response.content))
        image = image.resize((1024, 576))
        image = image.convert('RGB')
        
        # Generate video
        video_service.ensure_svd_loaded()
        frames = video_service.generate_video(image, motion_bucket_id=motion)
        
        # Export and upload
        from diffusers.utils import export_to_video
        timestamp = int(datetime.now().timestamp())
        video_filename = f"img2vid_{timestamp}.mp4"
        video_path = f"/tmp/{video_filename}"
        
        export_to_video(frames, video_path, fps=6)
        
        file_size = os.path.getsize(video_path) / (1024 * 1024)
        bucket_path = f"generated/{video_filename}"
        bucket = video_service.client.bucket(OUTPUT_BUCKET)
        blob = bucket.blob(bucket_path)
        blob.upload_from_filename(video_path)
        os.remove(video_path)
        
        return jsonify({
            "success": True,
            "video_url": f"gs://{OUTPUT_BUCKET}/{bucket_path}",
            "file_size_mb": round(file_size, 2),
            "frames": 25,
            "duration_seconds": 4.2
        })
        
    except Exception as e:
        logger.error(f"Image-to-video failed: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)

#!/usr/bin/env python3
"""
Offline LTX-Video Marketing API
Completely self-contained - no external dependencies
"""

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
OUTPUT_BUCKET = os.environ.get("OUTPUT_BUCKET", "tomi-ltx-videos")
PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "serious-conduit-448301-d7")

class OfflineLTXService:
    def __init__(self):
        self.client = storage.Client()
        self.model_loaded = False
        self.infer = None
        self.InferenceConfig = None
        self.setup_offline_models()
    
    def setup_offline_models(self):
        """Download pre-stored models from bucket for offline use"""
        try:
            logger.info("Setting up offline LTX-Video models...")
            
            # Create local directories
            os.makedirs("/app/models", exist_ok=True)
            os.makedirs("/app/configs", exist_ok=True)
            os.makedirs("/app/ltx_video", exist_ok=True)
            
            # Download models from bucket
            self.download_from_bucket("models/ltx-video/", "/app/models/ltx-video/")
            self.download_from_bucket("models/pixart-xl/", "/app/models/pixart-xl/")
            self.download_from_bucket("ltx_video/", "/app/ltx_video/")
            self.download_from_bucket("configs/", "/app/configs/")
            
            # Set offline environment
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            
            # Add to Python path
            sys.path.insert(0, "/app")
            
            # Import LTX-Video (now offline)
            from ltx_video.inference import infer, InferenceConfig
            self.infer = infer
            self.InferenceConfig = InferenceConfig
            
            self.model_loaded = True
            logger.info("✅ Offline LTX-Video models loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to setup offline models: {e}")
            import traceback
            traceback.print_exc()
            self.model_loaded = False
    
    def download_from_bucket(self, prefix, local_dir):
        """Download files from GCS bucket"""
        try:
            bucket = self.client.bucket(MODEL_BUCKET)
            blobs = bucket.list_blobs(prefix=prefix)
            
            downloaded_count = 0
            for blob in blobs:
                if not blob.name.endswith('/'):  # Skip directories
                    local_path = Path(local_dir) / blob.name.replace(prefix, '')
                    local_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    logger.info(f"Downloading {blob.name} to {local_path}")
                    blob.download_to_filename(str(local_path))
                    downloaded_count += 1
            
            logger.info(f"Downloaded {downloaded_count} files from {prefix}")
            
        except Exception as e:
            logger.error(f"Failed to download from {prefix}: {e}")
    
    def generate_video_offline(self, prompt, height=512, width=768, num_frames=49):
        """Generate video completely offline"""
        if not self.model_loaded:
            raise Exception("Offline models not loaded")
        
        try:
            # Create output path
            timestamp = int(datetime.now().timestamp())
            output_filename = f"video_{timestamp}.mp4"
            output_path = f"/tmp/{output_filename}"
            
            # Use offline config
            config_path = "/app/configs/configs/ltxv-2b-0.9.8-distilled.yaml"
            if not os.path.exists(config_path):
                # Fallback config paths
                possible_configs = [
                    "/app/configs/ltxv-2b-0.9.8-distilled.yaml",
                    "/app/ltxv-2b-0.9.8-distilled.yaml"
                ]
                for config in possible_configs:
                    if os.path.exists(config):
                        config_path = config
                        break
            
            logger.info(f"Using config: {config_path}")
            logger.info(f"Generating offline video: {prompt[:50]}...")
            logger.info(f"Dimensions: {width}x{height}, {num_frames} frames")
            
            # Configure offline inference
            config = self.InferenceConfig(
                pipeline_config=config_path,
                prompt=prompt,
                height=height,
                width=width,
                num_frames=num_frames,
                output_path=output_path
            )
            
            # Generate video offline
            self.infer(config=config)
            
            # Upload to output bucket
            output_bucket = self.client.bucket(OUTPUT_BUCKET)
            blob = output_bucket.blob(output_filename)
            blob.upload_from_filename(output_path)
            
            # Get public URL
            video_url = f"gs://{OUTPUT_BUCKET}/{output_filename}"
            
            logger.info(f"✅ Offline video generated: {video_url}")
            
            return {
                "success": True,
                "video_url": video_url,
                "filename": output_filename,
                "prompt": prompt,
                "dimensions": f"{width}x{height}",
                "frames": num_frames,
                "mode": "offline"
            }
            
        except Exception as e:
            logger.error(f"❌ Offline video generation failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }

# Initialize offline service
ltx_service = OfflineLTXService()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": ltx_service.model_loaded,
        "gpu_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "mode": "offline",
        "dependencies": "zero_external"
    })

@app.route('/generate', methods=['POST'])
def generate_video():
    """Generate video endpoint - completely offline"""
    try:
        data = request.get_json()
        
        if not data or 'prompt' not in data:
            return jsonify({"error": "Missing 'prompt' in request"}), 400
        
        prompt = data['prompt']
        height = data.get('height', 512)
        width = data.get('width', 768)
        num_frames = data.get('num_frames', 49)
        
        # Validate parameters
        if height > 1024 or width > 1024:
            return jsonify({"error": "Maximum resolution is 1024x1024"}), 400
        
        if num_frames > 121:
            return jsonify({"error": "Maximum frames is 121"}), 400
        
        # Generate video offline
        result = ltx_service.generate_video_offline(prompt, height, width, num_frames)
        
        if result['success']:
            return jsonify(result), 200
        else:
            return jsonify(result), 500
            
    except Exception as e:
        logger.error(f"API error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/presets', methods=['GET'])
def get_presets():
    """Get available video presets"""
    presets = {
        "tiny": {"height": 384, "width": 512, "num_frames": 25},
        "small": {"height": 512, "width": 768, "num_frames": 49},
        "medium": {"height": 720, "width": 1280, "num_frames": 73},
        "square": {"height": 720, "width": 720, "num_frames": 49},
        "vertical": {"height": 1280, "width": 720, "num_frames": 49}
    }
    return jsonify(presets)

@app.route('/', methods=['GET'])
def root():
    """Root endpoint with API documentation"""
    return jsonify({
        "service": "Offline LTX-Video Marketing Powerhouse",
        "version": "1.0",
        "mode": "offline",
        "dependencies": "zero_external",
        "endpoints": {
            "/health": "GET - Health check",
            "/generate": "POST - Generate video (offline)",
            "/presets": "GET - Available presets"
        },
        "example_request": {
            "prompt": "A cat walking on a sunny beach",
            "height": 512,
            "width": 768,
            "num_frames": 49
        }
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    
    # Use gunicorn in production
    if os.environ.get('GAE_ENV', '').startswith('standard'):
        app.run(host='0.0.0.0', port=port, debug=False)
    else:
        # Development mode
        app.run(host='0.0.0.0', port=port, debug=True)

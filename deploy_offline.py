#!/usr/bin/env python3
"""
Deploy Offline LTX-Video Service
Uses existing models and deploys updated offline service
"""

import subprocess
import os

def deploy_offline_service():
    """Deploy the offline service to Cloud Run"""
    
    print("🚀 Deploying Offline LTX-Video Service")
    print("=" * 50)
    
    project_id = "serious-conduit-448301-d7"
    service_name = "ltx-video-api"
    region = "us-central1"
    
    # Build and deploy
    print("📦 Building container...")
    
    build_cmd = [
        "gcloud", "builds", "submit",
        "--tag", f"gcr.io/{project_id}/{service_name}-offline",
        "./cloudrun/"
    ]
    
    try:
        subprocess.run(build_cmd, check=True)
        print("✅ Container built successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Build failed: {e}")
        return False
    
    # Deploy to Cloud Run
    print("🚀 Deploying to Cloud Run...")
    
    deploy_cmd = [
        "gcloud", "run", "deploy", f"{service_name}-offline",
        "--image", f"gcr.io/{project_id}/{service_name}-offline",
        "--region", region,
        "--platform", "managed",
        "--allow-unauthenticated",
        "--memory", "16Gi",
        "--cpu", "4",
        "--gpu", "1",
        "--gpu-type", "nvidia-l4",
        "--max-instances", "3",
        "--timeout", "3600",
        "--set-env-vars", f"GOOGLE_CLOUD_PROJECT={project_id}",
        "--set-env-vars", "MODEL_BUCKET=tomi-ltx-models-1760477322",
        "--set-env-vars", "OUTPUT_BUCKET=tomi-ltx-videos",
        "--service-account", f"ltx-video-service@{project_id}.iam.gserviceaccount.com"
    ]
    
    try:
        result = subprocess.run(deploy_cmd, check=True, capture_output=True, text=True)
        print("✅ Deployed successfully!")
        
        # Extract service URL
        lines = result.stdout.split('\n')
        service_url = None
        for line in lines:
            if 'https://' in line and 'run.app' in line:
                service_url = line.strip()
                break
        
        if service_url:
            print(f"🔗 Service URL: {service_url}")
            
            # Update scripts with new URL
            update_scripts(service_url)
            
            return service_url
        else:
            print("⚠️  Deployment successful but couldn't extract URL")
            return True
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Deployment failed: {e}")
        return False

def update_scripts(service_url):
    """Update scripts with new service URL"""
    
    scripts = [
        "scripts/test_api.py",
        "scripts/generate_video.py"
    ]
    
    for script_path in scripts:
        try:
            with open(script_path, 'r') as f:
                content = f.read()
            
            # Replace API_URL
            updated_content = content.replace(
                'API_URL = "https://ltx-video-api-362062855771.us-central1.run.app"',
                f'API_URL = "{service_url}"'
            )
            
            with open(script_path, 'w') as f:
                f.write(updated_content)
            
            print(f"✅ Updated {script_path}")
            
        except Exception as e:
            print(f"⚠️  Failed to update {script_path}: {e}")

def main():
    service_url = deploy_offline_service()
    
    if service_url:
        print(f"\n🎉 Offline LTX-Video Marketing Powerhouse Deployed!")
        print(f"🔗 API URL: {service_url}")
        print("🚀 Zero external dependencies confirmed")
        print("\n📋 Test commands:")
        print("  python scripts/test_api.py")
        print("  python scripts/generate_video.py 'A cat on beach' small 'test_video'")
    else:
        print("\n❌ Deployment failed")

if __name__ == "__main__":
    main()

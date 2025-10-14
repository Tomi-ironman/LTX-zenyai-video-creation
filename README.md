<<<<<<< HEAD
# LTX-Video Marketing Powerhouse

A completely self-contained Google Cloud Platform solution for high-volume marketing video generation using LTX-Video. **Zero external dependencies** - all models are pre-downloaded and stored in GCP.

## 🏗️ Architecture

```
Pre-downloaded Models (GCP Bucket) → Cloud Run + GPU → API Endpoint → Desktop Videos
```

## ✨ Features

- **🚀 Zero External Dependencies**: No Hugging Face API calls, completely offline
- **⚡ GPU Acceleration**: NVIDIA L4 GPU on Cloud Run
- **📈 Auto-scaling**: Scales to zero when not in use
- **💰 Cost Effective**: Pay only when generating videos
- **🖥️ Desktop Integration**: Videos automatically save to desktop
- **🔄 CI/CD Ready**: GitHub Actions for automatic deployments

## 🚀 Quick Start

### 1. Deploy to Cloud Run

```bash
# Clone repository
git clone https://github.com/yourusername/ltx-video-marketing.git
cd ltx-video-marketing

# Deploy to Cloud Run
gcloud run deploy ltx-video-api \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 16Gi \
  --cpu 4 \
  --gpu 1 \
  --gpu-type nvidia-l4 \
  --max-instances 3
```

### 2. Generate Videos

```bash
# Test the API
python3 test_api.py

# Generate marketing video
python3 generate_video.py "A sleek product commercial" medium "my_video"
```

## 📁 Project Structure

```
ltx-video-marketing/
├── cloudrun/
│   ├── Dockerfile              # Container definition
│   ├── main.py                 # Flask API server
│   ├── requirements.txt        # Python dependencies
│   └── offline_inference.py    # Offline video generation
├── scripts/
│   ├── setup_models.py         # Download models to bucket
│   ├── test_api.py            # API testing
│   └── generate_video.py      # Video generation client
├── .github/
│   └── workflows/
│       └── deploy.yml         # CI/CD pipeline
├── cloudbuild.yaml            # Cloud Build configuration
└── README.md                  # This file
```

## 🔧 Configuration

### Environment Variables

- `GOOGLE_CLOUD_PROJECT`: Your GCP project ID
- `MODEL_BUCKET`: GCS bucket containing pre-downloaded models
- `OUTPUT_BUCKET`: GCS bucket for generated videos

### Model Storage

Models are pre-downloaded and stored in GCS:
- `gs://your-bucket/models/ltx-video/` - Main LTX-Video model
- `gs://your-bucket/models/pixart-xl/` - Text encoder model

## 🎬 API Endpoints

### Health Check
```bash
GET /health
```

### Generate Video
```bash
POST /generate
{
  "prompt": "Your video description",
  "height": 512,
  "width": 768,
  "num_frames": 49
}
```

### Get Presets
```bash
GET /presets
```

## 💰 Cost Estimation

- **Cloud Run**: ~$0.50-1.00/hour (only when active)
- **Per Video**: ~$0.05-0.20 depending on size
- **Storage**: ~$0.02/GB/month
- **100 videos/month**: ~$5-20 total

## 🔄 CI/CD Pipeline

Automatic deployment on push to main branch:

1. **Build**: Container image built with Cloud Build
2. **Test**: API health checks
3. **Deploy**: Updated service deployed to Cloud Run
4. **Verify**: Deployment verification

## 🛠️ Development

### Local Development

```bash
# Install dependencies
pip install -r cloudrun/requirements.txt

# Run locally (CPU mode)
cd cloudrun
python main.py
```

### Update Models

```bash
# Download new models
python scripts/setup_models.py

# Redeploy service
git push origin main
```

## 📊 Monitoring

- **Cloud Run Metrics**: CPU, Memory, Request latency
- **Custom Metrics**: Video generation time, success rate
- **Logging**: Structured logs in Cloud Logging

## 🔒 Security

- **IAM**: Least privilege service accounts
- **VPC**: Optional private networking
- **Authentication**: API key support (optional)

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Make changes
4. Test locally
5. Submit pull request

## 📄 License

MIT License - see LICENSE file for details.
=======
# LTX-zenyai-video-creation
>>>>>>> cefbc4f582924d6ee98f4fb5ad7b1da0fbc09f55

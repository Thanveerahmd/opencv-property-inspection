#!/bin/bash
# EC2 user-data: install the app from the S3 bundle and run it on port 80. deploy.sh fills in the placeholders.
set -euxo pipefail
BUCKET="__BUCKET__"
REGION="__REGION__"
MODEL="__MODEL__"

apt-get update -y
apt-get install -y python3-venv python3-pip unzip curl
# AWS CLI v2 for arm64
curl -sSL "https://awscli.amazonaws.com/awscli-exe-linux-aarch64.zip" -o /tmp/awscli.zip
unzip -q /tmp/awscli.zip -d /tmp && /tmp/aws/install

mkdir -p /opt/app && cd /opt/app
aws s3 cp "s3://$BUCKET/deploy/app.tgz" /tmp/app.tgz --region "$REGION"
tar xzf /tmp/app.tgz
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -e .
# model files straight from their published sources, checked against pinned SHA-256 hashes
SPATIAL_INSPECTION_MODELS_DIR=/opt/app/models .venv/bin/python -m spatial_inspection.models download
mkdir -p results uploads
aws s3 sync "s3://$BUCKET/deploy/results" results --region "$REGION" || true

cat > /etc/systemd/system/inspection.service <<EOF
[Unit]
Description=Spatially aware property inspection
After=network-online.target

[Service]
WorkingDirectory=/opt/app
Environment=AWS_REGION=$REGION
Environment=S3_BUCKET=$BUCKET
Environment=BEDROCK_MODEL_ID=$MODEL
Environment=SPATIAL_INSPECTION_MODELS_DIR=/opt/app/models
Environment=RESULTS_DIR=/opt/app/results
Environment=UPLOADS_DIR=/opt/app/uploads
Environment=SI_THREADS=8
Environment=SI_PRIORITY=normal
ExecStart=/opt/app/.venv/bin/uvicorn spatial_inspection.server:app --host 0.0.0.0 --port 80
Restart=always

[Install]
WantedBy=multi-user.target
EOF
systemctl daemon-reload
systemctl enable --now inspection.service

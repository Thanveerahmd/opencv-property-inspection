#!/usr/bin/env bash
# Deploy the inspection endpoint to AWS: an S3 bucket for uploads and results, an IAM role that may use S3 and
# Amazon Bedrock, and one EC2 instance (Graviton, arm64) that runs the FastAPI app on port 80.
#
#   aws login                      # once, in your own terminal
#   deploy/aws/deploy.sh           # from the repository root; prints the endpoint URL
#
# Settings (environment): AWS_REGION (default: the CLI's region), INSTANCE_TYPE (default c7g.2xlarge),
# BEDROCK_MODEL_ID (default: picked from the region's inference profiles), STACK (default spatial-inspection).
set -euo pipefail

STACK="${STACK:-spatial-inspection}"
REGION="${AWS_REGION:-$(aws configure get region)}"
INSTANCE_TYPE="${INSTANCE_TYPE:-c7g.2xlarge}"
ACCOUNT="$(aws sts get-caller-identity --query Account --output text)"
BUCKET="${BUCKET:-$STACK-$ACCOUNT-$REGION}"
ROLE="$STACK-ec2"
export AWS_REGION="$REGION"
echo "account $ACCOUNT, region $REGION, bucket $BUCKET"

# 1. bucket for uploads, results and the code bundle
if ! aws s3api head-bucket --bucket "$BUCKET" 2>/dev/null; then
  if [ "$REGION" = "us-east-1" ]; then aws s3api create-bucket --bucket "$BUCKET" >/dev/null
  else aws s3api create-bucket --bucket "$BUCKET" --create-bucket-configuration LocationConstraint="$REGION" >/dev/null; fi
  aws s3api put-public-access-block --bucket "$BUCKET" --public-access-block-configuration \
    BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true
fi

# 2. a Bedrock model the agent can call: newest Claude inference profile in this region unless one is given
if [ -z "${BEDROCK_MODEL_ID:-}" ]; then
  BEDROCK_MODEL_ID="$(aws bedrock list-inference-profiles --query \
    "inferenceProfileSummaries[?contains(inferenceProfileId,'anthropic.claude') && (contains(inferenceProfileId,'sonnet') || contains(inferenceProfileId,'haiku'))].inferenceProfileId" \
    --output text | tr '\t' '\n' | sort | tail -1 || true)"
fi
echo "bedrock model: ${BEDROCK_MODEL_ID:-none found (the agent will use its rule-based policy)}"

# 3. role: read/write the bucket, call Bedrock, be reachable through Systems Manager
if ! aws iam get-role --role-name "$ROLE" >/dev/null 2>&1; then
  aws iam create-role --role-name "$ROLE" --assume-role-policy-document \
    '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"ec2.amazonaws.com"},"Action":"sts:AssumeRole"}]}' >/dev/null
  aws iam attach-role-policy --role-name "$ROLE" --policy-arn arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore
  aws iam create-instance-profile --instance-profile-name "$ROLE" >/dev/null
  aws iam add-role-to-instance-profile --instance-profile-name "$ROLE" --role-name "$ROLE"
fi
aws iam put-role-policy --role-name "$ROLE" --policy-name app --policy-document "{\"Version\":\"2012-10-17\",\"Statement\":[
  {\"Effect\":\"Allow\",\"Action\":[\"s3:GetObject\",\"s3:PutObject\",\"s3:ListBucket\"],\"Resource\":[\"arn:aws:s3:::$BUCKET\",\"arn:aws:s3:::$BUCKET/*\"]},
  {\"Effect\":\"Allow\",\"Action\":[\"bedrock:InvokeModel\",\"bedrock:InvokeModelWithResponseStream\",\"bedrock:Converse\"],\"Resource\":\"*\"}]}"

# 4. code bundle and the precomputed sample results
BUNDLE="$(mktemp -d)/app.tgz"
git archive --format=tar.gz -o "$BUNDLE" HEAD
aws s3 cp "$BUNDLE" "s3://$BUCKET/deploy/app.tgz" >/dev/null
if [ -d results ]; then aws s3 sync results "s3://$BUCKET/deploy/results" --only-show-errors; fi

# 5. network: allow HTTP in
VPC="$(aws ec2 describe-vpcs --filters Name=isDefault,Values=true --query 'Vpcs[0].VpcId' --output text)"
SG="$(aws ec2 describe-security-groups --filters Name=group-name,Values="$STACK" Name=vpc-id,Values="$VPC" --query 'SecurityGroups[0].GroupId' --output text)"
if [ "$SG" = "None" ]; then
  SG="$(aws ec2 create-security-group --group-name "$STACK" --description "inspection endpoint" --vpc-id "$VPC" --query GroupId --output text)"
  aws ec2 authorize-security-group-ingress --group-id "$SG" --protocol tcp --port 80 --cidr 0.0.0.0/0 >/dev/null
fi

# 6. instance: Ubuntu 24.04 arm64, set up by user-data (deploy/aws/bootstrap.sh)
AMI="$(aws ssm get-parameter --name /aws/service/canonical/ubuntu/server/24.04/stable/current/arm64/hvm/ebs-gp3/ami-id --query Parameter.Value --output text)"
USERDATA="$(mktemp)"
sed -e "s|__BUCKET__|$BUCKET|g" -e "s|__REGION__|$REGION|g" -e "s|__MODEL__|${BEDROCK_MODEL_ID:-}|g" deploy/aws/bootstrap.sh > "$USERDATA"
sleep 10   # a new instance profile takes a few seconds to become usable
ID="$(aws ec2 run-instances --image-id "$AMI" --instance-type "$INSTANCE_TYPE" --security-group-ids "$SG" \
  --iam-instance-profile Name="$ROLE" --user-data "file://$USERDATA" \
  --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=40,VolumeType=gp3}' \
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$STACK}]" --query 'Instances[0].InstanceId' --output text)"
aws ec2 wait instance-running --instance-ids "$ID"
IP="$(aws ec2 describe-instances --instance-ids "$ID" --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)"
echo "instance $ID is starting; the app is ready in about 5 minutes at: http://$IP/"
echo "follow setup with: aws ssm start-session --target $ID   (then: sudo tail -f /var/log/cloud-init-output.log)"

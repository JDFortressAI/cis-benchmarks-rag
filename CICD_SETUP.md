# CI/CD Setup Guide for CIS App

This guide will help you set up complete CI/CD for your CIS RAG application using GitHub Actions.

## Overview

The CI/CD pipeline will:
1. Trigger on every push to `main` or `master` branch
2. Build your Docker image
3. Push it to your existing ECR repository
4. Update your ECS service with the new image
5. Wait for deployment to complete

## Prerequisites

- Your AWS infrastructure is already set up (✅ Done)
- Your code is in a GitHub repository
- You have admin access to the GitHub repository

## Step 1: Create IAM User for GitHub Actions

Create a dedicated IAM user for GitHub Actions with minimal required permissions:

```bash
# Create IAM user
aws iam create-user --user-name github-actions-cis-app --region eu-central-1

# Create access key (save the output securely!)
aws iam create-access-key --user-name github-actions-cis-app --region eu-central-1
```

## Step 2: Attach IAM Policy

```bash
# Create the policy
aws iam create-policy \
    --policy-name GitHubActionsCISAppPolicy \
    --policy-document file://github-actions-policy.json \
    --region eu-central-1

# Attach policy to user (replace ACCOUNT-ID with your account: 530256939177)
aws iam attach-user-policy \
    --user-name github-actions-cis-app \
    --policy-arn arn:aws:iam::530256939177:policy/GitHubActionsCISAppPolicy \
    --region eu-central-1
```

## Step 3: Configure GitHub Secrets

In your GitHub repository, go to **Settings > Secrets and variables > Actions** and add these secrets:

- `AWS_ACCESS_KEY_ID`: The access key ID from Step 1
- `AWS_SECRET_ACCESS_KEY`: The secret access key from Step 1

**Note**: Your application environment variables (OPENAI_API_KEY, DEMO_USERNAME, DEMO_PASSWORD, AWS credentials for the app) are already configured in AWS Secrets Manager and will be automatically injected into your ECS tasks. The GitHub secrets above are only for the CI/CD pipeline itself.

## Step 4: Test the Pipeline

1. Commit and push any change to your `main` or `master` branch
2. Go to **Actions** tab in your GitHub repository
3. Watch the deployment process
4. Once complete, your app at https://cis.jdfortress.com should reflect the changes

## Pipeline Features

### Automatic Deployment
- Deploys on every push to main/master
- Uses commit SHA as image tag for traceability
- Also maintains a `latest` tag

### Zero Downtime Deployment
- ECS rolling deployment ensures no downtime
- Health checks ensure new version is working before old one is terminated
- Pipeline waits for service stability before completing

### Security
- Uses dedicated IAM user with minimal permissions
- Secrets stored securely in GitHub
- No hardcoded credentials in code

## Monitoring Deployments

### GitHub Actions
- View deployment logs in the Actions tab
- See build status and deployment progress
- Get notified of failures

### AWS Console
- Monitor ECS service deployments
- Check CloudWatch logs for application logs
- View ALB target group health

### Application
- Your app is available at: https://cis.jdfortress.com
- Check functionality after each deployment

## Troubleshooting

### Common Issues

1. **ECR Push Fails**
   - Check AWS credentials in GitHub secrets
   - Verify IAM permissions

2. **ECS Deployment Fails**
   - Check ECS service logs in CloudWatch
   - Verify task definition is valid
   - Check if new image starts successfully

3. **Health Check Failures**
   - Ensure your app responds to health checks on `/`
   - Check application logs for startup errors

### Rollback Process

If a deployment fails, you can quickly rollback:

```bash
# Get previous task definition revision
aws ecs describe-services --cluster cis-cluster --services cis-service --region eu-central-1

# Update service to previous revision
aws ecs update-service \
    --cluster cis-cluster \
    --service cis-service \
    --task-definition cis-task:PREVIOUS_REVISION \
    --region eu-central-1
```

## Next Steps

### Optional Enhancements

1. **Add staging environment**
   - Create separate ECS service for staging
   - Deploy to staging first, then production

2. **Add tests to pipeline**
   - Run unit tests before deployment
   - Add integration tests

3. **Add notifications**
   - Slack/email notifications for deployments
   - Alert on failures

4. **Environment-specific configurations**
   - Different configs for staging/production
   - Feature flags

## Files Created

- `.github/workflows/deploy.yml` - Main CI/CD pipeline
- `github-actions-policy.json` - IAM policy for GitHub Actions
- `CICD_SETUP.md` - This setup guide

Your CI/CD pipeline is now ready! 🚀

## 🛠️ Quick Setup (4 steps):

1. **Run the initial setup script**:
   ```bash
   ./setup-cicd.sh
   ```

2. **Run the complete setup script**:
   ```bash
   ./complete-setup.sh
   ```

3. **Add GitHub Secrets**:
   - Go to your GitHub repo → Settings → Secrets and variables → Actions
   - Add `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` from step 1 output

4. **Push to trigger deployment**:
   ```bash
   git add .
   git commit -m "Add CI/CD pipeline"
   git push origin main
   ```

## Environment Variables Handled

Your application requires these environment variables, which are now properly configured:

### In AWS Secrets Manager (`cis-secrets-complete`):
- `AWS_ACCESS_KEY_ID` - For S3 and S3 Vectors access
- `AWS_SECRET_ACCESS_KEY` - For S3 and S3 Vectors access  
- `OPENAI_API_KEY` - For GPT API calls
- `DEMO_USERNAME` - For app authentication
- `DEMO_PASSWORD` - For app authentication

### In ECS Task Environment:
- `AWS_REGION` - Set to `eu-central-1`

### From config.yaml (loaded by app):
- S3 bucket configurations
- Vector database settings
- Model configurations
- Regional settings

The CI/CD pipeline automatically handles all these configurations and your app will have access to all required services: AWS S3, AWS S3 Vectors, and OpenAI API.
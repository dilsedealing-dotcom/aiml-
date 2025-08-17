from google.cloud import run_v2
from google.cloud import storage
import os
import zipfile

class CloudDeployer:
    def __init__(self, project_id, region='us-central1'):
        self.project_id = project_id
        self.region = region
        
    def create_dockerfile(self):
        dockerfile_content = """
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["python", "flask_server.py"]
"""
        with open('Dockerfile', 'w') as f:
            f.write(dockerfile_content)
        
        print("Dockerfile created")
    
    def create_cloud_run_config(self):
        config = """
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: mt5-trading-system
  annotations:
    run.googleapis.com/ingress: all
spec:
  template:
    metadata:
      annotations:
        autoscaling.knative.dev/maxScale: "10"
        run.googleapis.com/cpu-throttling: "false"
        run.googleapis.com/memory: "2Gi"
        run.googleapis.com/cpu: "1000m"
    spec:
      containers:
      - image: gcr.io/{project_id}/mt5-trading-system
        ports:
        - containerPort: 5000
        env:
        - name: PORT
          value: "5000"
        resources:
          limits:
            memory: "2Gi"
            cpu: "1000m"
"""
        with open('cloud-run-service.yaml', 'w') as f:
            f.write(config.format(project_id=self.project_id))
        
        print("Cloud Run config created")
    
    def deploy_to_cloud_run(self):
        """Deploy the trading system to Google Cloud Run"""
        
        # Create necessary files
        self.create_dockerfile()
        self.create_cloud_run_config()
        
        # Build and deploy commands
        commands = [
            f"gcloud config set project {self.project_id}",
            "gcloud builds submit --tag gcr.io/{}/mt5-trading-system".format(self.project_id),
            f"gcloud run deploy mt5-trading-system --image gcr.io/{self.project_id}/mt5-trading-system --platform managed --region {self.region} --allow-unauthenticated --port 5000"
        ]
        
        print("Deployment commands:")
        for cmd in commands:
            print(f"  {cmd}")
        
        print("\nTo deploy, run these commands in your terminal after installing Google Cloud SDK")
        
        return commands

# Usage example
if __name__ == "__main__":
    project_id = input("Enter your Google Cloud Project ID: ")
    
    deployer = CloudDeployer(project_id)
    commands = deployer.deploy_to_cloud_run()
    
    print("\nDeployment preparation complete!")
    print("Make sure you have:")
    print("1. Google Cloud SDK installed")
    print("2. Authenticated with 'gcloud auth login'")
    print("3. Enabled Cloud Run and Cloud Build APIs")
# heart-disease-mlops

## Deployed API — Access Instructions (Local Testing)

This project does not expose a public API URL. It is deployed and tested locally
using Kubernetes running on Docker Desktop. Follow the steps below to reproduce
the deployment and test the API on your own machine.

### Prerequisites

- Docker Desktop installed, with Kubernetes enabled
  (Docker Desktop → Settings → Kubernetes → Enable Kubernetes)
- `kubectl` CLI (installed automatically with Docker Desktop's Kubernetes)

### 1. Build the Docker image

```bash
docker build -t heart-disease-api:v2 ./app
```

### 2. Deploy to Kubernetes

```bash
kubectl apply -f deployment/k8s-manifest.yaml
```

This creates a Deployment with 2 replicas and a LoadBalancer Service.

### 3. Verify the deployment

```bash
kubectl get pods
kubectl get services
```

Both pods should show `1/1 Running`, and `heart-disease-api-service` should be
listed as `LoadBalancer` on port 80.

### 4. Test the API

Health check:

```bash
curl http://localhost/health
```

Expected response:

```json
{"status": "healthy"}
```

Prediction:

```bash
curl -X POST http://localhost/predict \
  -H "Content-Type: application/json" \
  -d '{"age": 63, "sex": 1, "cp": 1, "trestbps": 145, "chol": 233, "fbs": 1, "restecg": 2, "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 3, "ca": 0, "thal": 6}'
```

Expected response:

```json
{"confidence": 0.828, "prediction": 0, "prediction_label": "No Disease"}
```

Metrics (basic request/prediction counters):

```bash
curl http://localhost/metrics
```

### 5. Alternative: run without Kubernetes (plain Docker)

If you only want to test the container directly, without deploying to
Kubernetes:

```bash
docker run -d -p 5001:5001 heart-disease-api:v2
curl http://localhost:5001/health
curl -X POST http://localhost:5001/predict \
  -H "Content-Type: application/json" \
  -d '{"age": 63, "sex": 1, "cp": 1, "trestbps": 145, "chol": 233, "fbs": 1, "restecg": 2, "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 3, "ca": 0, "thal": 6}'
```

### 6. Tear down

```bash
kubectl delete -f deployment/k8s-manifest.yaml
```

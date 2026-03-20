# Kubernetes (minimal)

1. Build and load images into your cluster (examples):

   ```bash
   docker build --build-arg SORAKAI_SERVICE=gateway -t sorakai-gateway:latest .
   docker build --build-arg SORAKAI_SERVICE=ingest -t sorakai-ingest:latest .
   docker build --build-arg SORAKAI_SERVICE=rag -t sorakai-rag:latest .
   kind load docker-image sorakai-gateway:latest sorakai-ingest:latest sorakai-rag:latest
   ```

2. Apply manifests:

   ```bash
   kubectl apply -f k8s/namespace.yaml
   kubectl apply -f k8s/redis.yaml
   kubectl apply -f k8s/ingest.yaml
   kubectl apply -f k8s/rag.yaml
   kubectl apply -f k8s/gateway.yaml
   ```

3. Port-forward the gateway:

   ```bash
   kubectl -n sorakai port-forward svc/sorakai-gateway 8000:8000
   ```

4. **Redis** is required so ingest and RAG share the same knowledge base across pods.

5. Optional **MLflow**: deploy an MLflow tracking server and set `MLFLOW_TRACKING_URI` on ingest/rag Deployments (see commented env in `ingest.yaml`).

6. **OpenAPI**: images include `/app/openapi/*.openapi.{json,yaml}`. From the cluster, `kubectl port-forward` the gateway and open `http://127.0.0.1:8000/openapi.bundled.json` or use live `http://127.0.0.1:8000/openapi.json`. To publish specs without hitting pods, create a ConfigMap from `openapi/` (see `openapi/README.md`).

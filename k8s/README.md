# Kubernetes (minimal)

1. Build and load images into your cluster (examples):

   ```bash
   docker build --build-arg SORAKAI_SERVICE=gateway -t sorakai-gateway:latest .
   docker build --build-arg SORAKAI_SERVICE=ingest -t sorakai-ingest:latest .
   docker build --build-arg SORAKAI_SERVICE=rag -t sorakai-rag:latest .
   docker build -f Dockerfile.mlflow -t sorakai-mlflow:latest .
   kind load docker-image sorakai-gateway:latest sorakai-ingest:latest sorakai-rag:latest sorakai-mlflow:latest
   ```

2. Apply manifests:

   ```bash
   kubectl apply -f k8s/namespace.yaml
   kubectl apply -f k8s/redis.yaml
   kubectl apply -f k8s/qdrant.yaml         # Wave 5: optional, only needed when VECTOR_STORE=qdrant
   kubectl apply -f k8s/mlflow.yaml
   kubectl apply -f k8s/ollama.yaml
   kubectl apply -f k8s/ingest.yaml
   kubectl apply -f k8s/rag.yaml
   kubectl apply -f k8s/gateway.yaml
   ```

   Pull Ollama models (ingest/RAG use **semantic embeddings** + chat):

   ```bash
   kubectl -n sorakai rollout status deploy/ollama
   kubectl -n sorakai exec deploy/ollama -- ollama pull llama3.2:1b
   kubectl -n sorakai exec deploy/ollama -- ollama pull nomic-embed-text
   ```

3. Port-forward the gateway:

   ```bash
   kubectl -n sorakai port-forward svc/sorakai-gateway 8000:8000
   ```

4. **MLflow UI** (tracking server runs in-cluster; SQLite + `emptyDir` — data is lost if the pod is deleted):

   ```bash
   kubectl -n sorakai port-forward svc/mlflow 5000:5000
   ```

   Open `http://127.0.0.1:5000`. Ingest/RAG use `MLFLOW_TRACKING_URI=http://mlflow.sorakai.svc.cluster.local:5000`.

5. **Redis** is required so ingest and RAG share the same knowledge base across pods.

6. **OpenAPI**: images include `/app/openapi/*.openapi.{json,yaml}`. From the cluster, `kubectl port-forward` the gateway and open `http://127.0.0.1:8000/openapi.bundled.json` or use live `http://127.0.0.1:8000/openapi.json`. To publish specs without hitting pods, create a ConfigMap from `openapi/` (see `openapi/README.md`).

7. **Production MLflow**: replace `emptyDir` in `mlflow.yaml` with a PVC and use Postgres/MySQL for `--backend-store-uri` (custom image or `args` override).

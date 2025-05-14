<div align="center">
    <h1 align="center">Self-host Models with Triton and BentoML</h1>
</div>

This project shows how to self-host models using:

- [Triton Inference Server](https://github.com/triton-inference-server/server) for fast, high-performance inference
- [BentoML](https://github.com/bentoml/BentoML) for packaging, API serving, deployment and orchestration

💡 You can use this project as a foundation for advanced customization, such as custom models and inference logic.

See [the full list](https://docs.bentoml.com/en/latest/examples/overview.html) of BentoML example projects.

## Prerequisites

Before you begin, make sure you have:

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) for environment management (or `pip`, optional)
- Docker with GPU support (`nvidia-container-toolkit`)
- An NVIDIA GPU (e.g., A100), if your model requires GPU acceleration

## Setup

1. Clone the repository.

   ```bash
   git clone https://github.com/bentoml/BentoTriton.git && cd BentoTriton
   ```

2. Create a virtual environment and install dependencies.

   ```bash
   uv venv
   uv pip install --editable .
   ```

3. Add your model to the [model_repository](./model_repository/) directory. Note that [Triton requires models in a specific format](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_repository.html).

## Build and package

A [Bento](https://docs.bentoml.com/en/latest/get-started/packaging-for-deployment.html) is packaged with all the source code, Python dependencies, model references, and environment setup, making it easy to deploy consistently across different environments.

```bash
# Build the Bento and extract its tag
BENTO_TAG=$(bentoml build -o tag | sed 's/__tag__://')

# Create a Docker image from the Bento
bentoml containerize "$BENTO_TAG" -t "$BENTO_TAG"
```

## Run locally

Run the container on your local GPU with:

```bash
docker run --gpus=all --rm --net=host ${BENTO_TAG} serve --debug
```

Triton will launch inside the container. BentoML will expose your service at `http://localhost:3000`. Once it's running, you can interact with your service using routes like:

- `GET /health/live`, `/health/ready`: Basic readiness checks
- `GET /v2/models/{model_name}/ready`: Check if a specific model is ready
- `GET /v2/models/{model_name}`: Get model metadata
- `PUT /v2/models/{model_name}/infer`: Perform inference

See [service.py](./service.py) for more information.

## Deploy to BentoCloud

After your service is ready, you can deploy it to BentoCloud for better management and scalability. [Sign up](https://www.bentoml.com/) if you haven't got a BentoCloud account.

Make sure you have [logged in to BentoCloud](https://docs.bentoml.com/en/latest/scale-with-bentocloud/manage-api-tokens.html).

```bash
bentoml cloud login
```

Create it from the existing directory:

```bash
bentoml deploy .
```

**Note**: You can also deploy the pre-built Docker image to your own infrastructure.
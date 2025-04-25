<div align="center">
    <h1 align="center">Self-host with Triton Inference Server and BentoML</h1>
</div>

To setup, make sure you have docker installed and a machine with GPU to run the models if your models require GPUs.

```bash
git clone https://github.com/bentoml/BentoTriton.git && cd BentoTriton

uv venv && uv pip install -r pyproject.toml
```

To run locally:

```bash
BENTO_TAG = $(bentoml build -o tag | sed 's/__tag__://')
bentoml containerize ${BENTO_TAG}  -t ${BENTO_TAG}

docker run --gpus=all --rm --net=host ${BENTO_TAG} serve --debug
```

To deploy:

```bash
bentoml deploy
```

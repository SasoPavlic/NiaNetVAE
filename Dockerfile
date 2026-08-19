FROM pytorch/pytorch:2.9.1-cuda12.6-cudnn9-runtime

LABEL org.opencontainers.image.title="NiaNetVAE MetroPT study"
LABEL org.opencontainers.image.description="Controlled shared-core MetroPT workflow comparison"

WORKDIR /app

COPY pyproject.toml README.md /app/
COPY src /app/src
COPY configs /app/configs
COPY main.py /app/main.py

RUN python -m pip install --no-cache-dir . \
    && python -c "import nianetvae, torch; print(nianetvae.__version__, torch.__version__)"

RUN mkdir -p /app/data /app/artifacts

ENTRYPOINT ["python", "-m", "nianetvae.cli"]

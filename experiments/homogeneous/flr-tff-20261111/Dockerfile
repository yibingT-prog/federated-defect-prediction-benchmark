FROM python:3.8-slim-bookworm@sha256:1d52838af602b4b5a831beb13a0e4d073280665ea7be7f69ce2382f29c5a613f
ENV PYTHONUNBUFFERED=1 PIP_DISABLE_PIP_VERSION_CHECK=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 TF_CPP_MIN_LOG_LEVEL=2 CUDA_VISIBLE_DEVICES=-1
RUN apt-get update && apt-get install -y --no-install-recommends build-essential libgomp1 ca-certificates && rm -rf /var/lib/apt/lists/*
RUN python -m pip install --no-cache-dir pip==24.3.1 setuptools==59.6.0 wheel==0.37.1
COPY requirements-tff.txt /tmp/requirements-tff.txt
COPY wheelhouse /opt/wheels
RUN --mount=type=cache,target=/root/.cache/pip python -m pip install --timeout 120 --retries 10 -r /tmp/requirements-tff.txt && python -m pip check
WORKDIR /work
CMD ["python", "smoke_tff.py"]

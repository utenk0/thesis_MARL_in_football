# syntax=docker/dockerfile:1
FROM python:3.11-slim-bookworm

ARG DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    SDL_VIDEODRIVER=dummy \
    SDL_AUDIODRIVER=dummy \
    MPLCONFIGDIR=/tmp/thesis-matplotlib \
    PROJECT_DIR=/workspace \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

RUN sed -i 's|http://deb.debian.org|https://deb.debian.org|g' \
      /etc/apt/sources.list.d/debian.sources \
    && apt-get -o Acquire::Retries=5 \
         -o Acquire::https::Timeout=30 update \
    && apt-get -o Acquire::Retries=5 \
         -o Acquire::https::Timeout=30 install -y --no-install-recommends \
      cmake build-essential pkg-config \
      libgl1-mesa-dev libsdl2-dev libsdl2-image-dev libsdl2-ttf-dev \
      libsdl2-gfx-dev libboost-python-dev libboost-filesystem-dev \
      libboost-thread-dev ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /tmp/thesis-matplotlib

WORKDIR /workspace

COPY requirements-project.txt requirements-methods.txt requirements-grf.txt ./
COPY football/requirements.txt ./football/requirements.txt

RUN python3 -m venv .venv-methods \
    && .venv-methods/bin/python -m pip install --upgrade pip setuptools wheel \
    && .venv-methods/bin/python -m pip install \
         -r requirements-project.txt -r requirements-methods.txt

RUN python3 -m venv .venv-grf \
    && .venv-grf/bin/python -m pip install \
         pip==24.0 setuptools==65.5.0 wheel==0.38.4 \
    && .venv-grf/bin/python -m pip install \
         -r requirements-grf.txt -r football/requirements.txt -r requirements-project.txt

COPY football ./football
RUN GFOOTBALL_PYTHON=/workspace/.venv-grf/bin/python \
    .venv-grf/bin/python -m pip install -e football --no-deps

COPY thesis_experiments ./thesis_experiments
COPY vm ./vm
COPY main.py README.md GRF_SETUP.md ./

# Training data and artifacts are deliberately not part of the image.  They
# are mounted at runtime so a dataset change does not create another multi-GB
# image layer and so checkpoints survive container replacement.
RUN mkdir -p /workspace/data/processed /workspace/artifacts/vm_runs

RUN .venv-grf/bin/python -m thesis_experiments.scripts.check_environment --target grf \
    && .venv-methods/bin/python -m thesis_experiments.scripts.check_environment --target methods \
    && .venv-grf/bin/python main.py --steps 2 \
    && .venv-methods/bin/python -m pytest thesis_experiments/tests -q

CMD ["bash", "vm/run_training.sh"]

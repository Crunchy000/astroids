# A competition entry: serves one agent over WebSocket on port 8080.
#
#   docker build -t my-entry .                                    # serves entry/my_agent.py
#   docker build -t m10 --build-arg AGENT=competition/models/m10.npz .
#   docker run --rm -p 8080:8080 my-entry
#
# AGENT is any agent spec (python -m astroids_gym.competition --help): a
# module:Class, a .npz model, or bots[:style].
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_CACHE_DIR=/tmp/numba
WORKDIR /app

COPY pyproject.toml README.md ./
COPY astroids_gym ./astroids_gym
RUN pip install --no-cache-dir ".[competition,fast]"

COPY competition/models ./competition/models
COPY entry ./entry

ARG AGENT=entry.my_agent:MyAgent
ENV AGENT=${AGENT}
# Compile the numba observation encoder now, not during the first match.
RUN python -c "from astroids_gym import Simulation, ObsConfig; from astroids_gym.obs import encode_battle; encode_battle(Simulation(), range(10), ObsConfig(extended=True))"

EXPOSE 8080
CMD ["sh", "-c", "exec python -m astroids_gym.competition serve \"$AGENT\" --port 8080"]

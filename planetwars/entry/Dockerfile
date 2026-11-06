FROM python:3.11-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# compile the planner's numba code now, into numba's cache in the image, rather than when the first game starts
RUN python -c "import planner; planner._warm_fast(); assert planner.fastplan is not None"
EXPOSE 8080
CMD ["python", "main.py"]

FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

VOLUME /app/checkpoints

CMD ["python3", "main.py", "--mode", "teacher", "--teacher-loop", \
     "--teacher-learning-passes", "3", "--teacher-replay-steps", "24"]

FROM python:3.10.11

WORKDIR /app

COPY . .

RUN pip3 install -r requirements.txt

CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:5000", "main:fetch_app()"]
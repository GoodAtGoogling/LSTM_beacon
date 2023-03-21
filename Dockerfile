FROM python:latest
WORKDIR /app

COPY requirements.txt /app
RUN pip install -r ./requirements.txt

COPY app.py /app
COPY model.h5 /app
COPY scaler_data /app
COPY templates /app/templates

#ENV FLASK_DEBUG=1

CMD ["python", "app.py"]
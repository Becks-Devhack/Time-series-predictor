FROM python:3.8
COPY requirements.txt /tmp
RUN pip install -r /tmp/requirements.txt
COPY /src /app
WORKDIR /app
EXPOSE 80
CMD ["python", "script.py"]
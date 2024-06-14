FROM python:3.6-slim-buster

COPY . /app
WORKDIR /app
RUN pip install -U sentence-transformers
RUN pip install --trusted-host pypi.python.org -r requirements.txt --verbose

EXPOSE 80
CMD ["/bin/bash"]
ENTRYPOINT ["python","main.py"]


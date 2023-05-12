FROM python:3.8.11-slim-buster

# create virtual enviaronment
ENV VIRTUAL_ENV=/opt/venv
RUN python -m venv ${VIRTUAL_ENV}
ENV PATH = "${VIRTUAL_ENV}/bin:$PATH"

ENV TZ=America/Bogota

# update pip
RUN python -m pip install --upgrade pip

# install requeriments
COPY requeriments.txt .
RUN pip install -r requeriments.txt --no-cache-dir

# add user
RUN useradd --create-home natalia
USER natalia

# copy apis
WORKDIR /Users/natalia/app
ENV PYTHONPATH=$PYTHONPATH:/Users/natalia/
COPY app/. .


CMD gunicorn app.main:app --bind=0.0.0.0:5000\
    --workers=2 \
    --worker-class=uvicorn.workers.UvicornWorker \
    --timeout 120 \
    --log-config app/logging.conf

    


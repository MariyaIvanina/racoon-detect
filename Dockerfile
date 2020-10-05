FROM python:3.7.4

RUN apt-get update && apt-get install -y build-essential libxml2 curl apt-utils
RUN apt-get install -y protobuf-compiler python-pil python-lxml python-tk

ADD requirements.txt /requirements.txt

RUN python -m venv /venv \
    && /venv/bin/pip install -U pip \
    && LIBRARY_PATH=/lib:/usr/lib /bin/sh -c "/venv/bin/pip install --no-cache-dir -r /requirements.txt"

RUN mkdir /backend/
WORKDIR /backend/

RUN git clone --depth 1 https://github.com/tensorflow/models.git

# Run protoc on the object detection repo
RUN cd models/research && \
    protoc object_detection/protos/*.proto --python_out=.
RUN cp models/research/object_detection/packages/tf2/setup.py .
RUN LIBRARY_PATH=/lib:/usr/lib /bin/sh -c "/venv/bin/pip install ."


ADD . /backend/

# uWSGI will listen on this port
EXPOSE 5000

# uWSGI configuration (customize as needed):
ENV FLASK_APP=main.py UWSGI_WSGI_FILE=main.py UWSGI_HTTP=:5000 UWSGI_VIRTUALENV=/venv UWSGI_MASTER=1 UWSGI_WORKERS=1 UWSGI_THREADS=8 UWSGI_LAZY_APPS=1 UWSGI_WSGI_ENV_BEHAVIOR=holy PYTHONDONTWRITEBYTECODE=1
ENV PATH="/venv/bin:${PATH}"
ENV PYTHONPATH="$PYTHONPATH:/backend:/backend/models/research/object_detection"
ENV PYTHONPATH="$PYTHONPATH:/backend/models/research/slim"
ENV PYTHONPATH="$PYTHONPATH:/backend/models/research"

# Start uWSGI
CMD ["/venv/bin/uwsgi", "--http-auto-chunked", "--http-keepalive"]
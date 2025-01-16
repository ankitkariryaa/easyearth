FROM python:3.8.0b1-slim-stretch

RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app

# Upgrade pip
RUN pip install --upgrade pip

COPY requirements.txt /usr/src/app/

RUN pip3 install --no-cache-dir -r requirements.txt --upgrade

COPY . /usr/src/app

EXPOSE 3781

ENTRYPOINT [ "gunicorn" ]
CMD ["-w", "2", "-b", "0.0.0.0:3781", "easyearth.wsgi"]

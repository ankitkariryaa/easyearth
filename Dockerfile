FROM python:3.10-slim

RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app

# Upgrade pip
RUN pip install --upgrade pip

COPY requirements.txt /usr/src/app/

RUN pip3 install --no-cache-dir -r requirements.txt --upgrade

COPY . /usr/src/app

EXPOSE 3781

ENTRYPOINT [ "gunicorn" ]
CMD ["-w", "2", "-b", "0.0.0.0:3781", "easyearth.app:app"]
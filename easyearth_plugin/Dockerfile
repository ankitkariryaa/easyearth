FROM python:3.10

# Install system dependencies
RUN apt-get update && apt-get install -y libexpat1 && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /usr/src/app
RUN mkdir -p /usr/src/app/user
WORKDIR /usr/src/app

# Copy only requirements first to leverage Docker cache
COPY requirements.txt /usr/src/app/

# upgrade pip
RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt --upgrade

# Copy the rest of the application
COPY . /usr/src/app

EXPOSE 3781

CMD ["python", "-m", "easyearth.app", "--host", "0.0.0.0", "--port", "3781"]

# TODO: fix gunicorn, so far it's not working
# ENTRYPOINT ["gunicorn"]
# CMD ["-w", "2", "-b", "0.0.0.0:3781", "easyearth.wsgi:app"]
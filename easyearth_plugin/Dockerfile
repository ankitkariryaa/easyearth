FROM python:3.10

# Install system dependencies
RUN apt-get update && apt-get install -y libexpat1 libgdal-dev gdal-bin --no-install-recommends && rm -rf /var/lib/apt/lists/*

# Define environment variables for paths
ENV APP_DIR=/usr/src/app
ENV EASYEARTH_DATA_DIR=/usr/src/app/data
ENV EASYEARTH_TEMP_DIR=/usr/src/app/tmp
ENV EASYEARTH_LOG_DIR=/usr/src/app/logs

# Create required directories
RUN mkdir -p $APP_DIR $EASYEARTH_DATA_DIR $EASYEARTH_TEMP_DIR $EASYEARTH_LOG_DIR
WORKDIR $APP_DIR

# Copy only requirements first to leverage Docker cache
COPY requirements.txt $APP_DIR/

# upgrade pip
RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt --upgrade

# Copy the application code
COPY . $APP_DIR/

EXPOSE 3781

CMD ["python", "-m", "easyearth.app", "--host", "0.0.0.0", "--port", "3781"]

# TODO: fix gunicorn, so far it's not working
# ENTRYPOINT ["gunicorn"]
# CMD ["-w", "2", "-b", "0.0.0.0:3781", "easyearth.wsgi:app"]
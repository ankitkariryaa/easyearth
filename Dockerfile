FROM python:3.10-slim

RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app

# Upgrade pip
RUN pip install --upgrade pip

# Copy necessary files to the container
COPY requirements.txt /usr/src/app/
# COPY download_models.py /usr/src/app/

# Create a virtual environment in the container
RUN python3 -m venv .venv

# Activate the virtual environment
ENV PATH="/usr/src/app/.venv/bin:$PATH"

# Install Python dependencies from the requirements file
RUN pip3 install --no-cache-dir -r requirements.txt --upgrade && \
    # Get the models from Hugging Face to bake into the container
    #     python3 download_models.py

COPY . /usr/src/app

EXPOSE 3781

ENTRYPOINT [ "gunicorn" ]
CMD ["-w", "2", "-b", "0.0.0.0:3781", "easyearth.app:app"]
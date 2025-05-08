#!/bin/bash

# Exit on any error
set -e

# Set the script's directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Change the permissions of the script directory
chmod -R 755 "$SCRIPT_DIR"

# Function to ensure Docker Compose is installed
check_docker_installation() {
  if ! command -v docker-compose &>/dev/null; then
    echo "Installing docker-compose..."
    sudo apt-get update && sudo apt-get install -y docker-compose
  else
    echo "docker-compose is already installed."
  fi
}

# Check if the docker image easyearth_plugin_easyearth-server exists, if exists return 0 else return 1
check_docker_image() {
  if sudo docker images | grep -q "easyearth_plugin_easyearth-server"; then
    echo "Docker image easyearth_plugin_easyearth-server already exists."
    return 0
  else
    echo "Docker image easyearth_plugin_easyearth-server does not exist."
    return 1
  fi
}

# Function to configure directories
configure_directory() {
  local dir_name="$1"
  local default_dir="$2"
  local result_dir

  read -p "Specify folder for $dir_name (default: $default_dir): " result_dir
  result_dir="${result_dir:-$default_dir}"

  [ ! -d "$result_dir" ] && mkdir -p "$result_dir" && echo "Created $dir_name at $result_dir"
  chmod -R 755 "$result_dir"
  echo "$result_dir"
}

# Build Docker image
build_docker_image() {
  echo "Building Docker image..."
  sudo docker-compose build --no-cache -t easyearth_plugin_easyearth-server:latest
}

# Start Docker container
start_docker_container() {
  echo "Starting Docker container..."
  sudo docker-compose up -d
}

# Test if the server is running
test_server() {
  echo "Testing if the server is running..."
  sleep 5
  if curl -s http://localhost:3781/v1/easyearth/ping | grep -q "Server is alive"; then
    echo "Server is running!"
  else
    echo "Server is not running. Check the logs."
    exit 1
  fi
}

# Main execution
main() {
  echo "Starting setup..."
  check_docker_installation

  if check_docker_image; then
    echo "Skipping Docker image build."
  else
    build_docker_image
  fi

  DATA_DIR=$(configure_directory "data directory" "$SCRIPT_DIR/data")
  TEMP_DIR=$(configure_directory "temp directory" "$SCRIPT_DIR/tmp")
  MODEL_DIR=$(configure_directory "cache directory" "$SCRIPT_DIR/.cache/models")
  LOG_DIR=$(configure_directory "logs directory" "$SCRIPT_DIR/logs")

  start_docker_container
  test_server

  echo "Setup completed successfully!"
}

# Run the script
main

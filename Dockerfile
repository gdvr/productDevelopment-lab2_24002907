# Use a Python base image (or any image compatible with your code)
FROM python:3.12-slim

# Set the working directory
WORKDIR /app

# Install system dependencies if necessary (for example, git and other utilities)
RUN apt-get update && \
    apt-get install -y git && \
    apt-get clean

# Install DVC and any additional dependencies
#RUN pip install --no-cache-dir dvc[all]  # Use [all] to include support for all DVC remotes

# Copy the rest of your application code
COPY . /app

# Install Python dependencies from your requirements file
RUN pip install --no-cache-dir -r requirements.txt


# Entry point for running DVC pipeline
CMD ["dvc", "repro"]

# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR ..

# Copy the current directory contents into the container at /app
COPY .. /app

# Install any needed dependencies
RUN pip install --no-cache-dir -r /app/requirements.txt


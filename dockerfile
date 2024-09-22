# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY credit_fraud_detection_with_dagster_v2.py .
COPY dataset .
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 3000 available to the world outside this container
EXPOSE 3001/tcp
EXPOSE 3001/udp

# Run dagit when the container launches
CMD ["dagster", "dev", "-f", "credit_fraud_detection_with_dagster_v2.py", "-p", "3001"]

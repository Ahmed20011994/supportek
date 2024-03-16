# Use the official Python image as the base image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements.txt file into the container
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Expose the port that the application runs on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000"]
# Start with an official Python 3.10 slim base image for a smaller footprint
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all your project files (Python scripts, etc.) into the container
COPY . .

# Expose the port that the Gradio interface will run on
EXPOSE 7860

# The command to run your application when the container starts
# This will launch the Gradio web server
CMD ["python", "main.py"]

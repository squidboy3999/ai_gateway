# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /usr/src/app
RUN mkdir -p /var/log/doc_handler/
# Copy the client script into the container at /usr/src/app
COPY . .
RUN --mount=type=cache,target=/root/.cache CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install --timeout 100 -r requirements.txt

# use port 5006 or set port to different number
# Run client.py when the container launches
#CMD ["python", "-m", "flask", "run", "--host=0.0.0.0"]

# Copyright and license information
#__copyright__   = "Copyright 2024, VISA Lab"
#__license__     = "MIT"

# Define arguments for function directory and Python runtime version
ARG FUNCTION_DIR="/home/app/"
ARG RUNTIME_VERSION="3.12"

# Base image: Python slim version
FROM python:${RUNTIME_VERSION}-slim AS python-base
# Upgrade pip in the base image
RUN pip install --upgrade pip

# Build image: Install awslambdaric
FROM python:${RUNTIME_VERSION}-slim AS build-image
ARG FUNCTION_DIR
# Create function directory
RUN mkdir -p ${FUNCTION_DIR}
# Install awslambdaric in the function directory
RUN pip install awslambdaric --target ${FUNCTION_DIR}

# Final image: Set up the Lambda function
FROM python:${RUNTIME_VERSION}-slim
ARG FUNCTION_DIR
WORKDIR ${FUNCTION_DIR}
# Copy awslambdaric from build-image
COPY --from=build-image ${FUNCTION_DIR} ${FUNCTION_DIR}
# Set environment variable for PyTorch model directory
ENV TORCH_HOME=${FUNCTION_DIR}

# Copy requirements, data files, and model weights
COPY requirements.txt requirements_extra.txt data.pt ${FUNCTION_DIR}

# Install main requirements
RUN pip install --no-cache-dir -r requirements.txt --target ${FUNCTION_DIR}

# Install facenet-pytorch without dependencies
RUN pip install --no-cache-dir --no-deps facenet-pytorch==2.5.3 --target ${FUNCTION_DIR}

# Copy entry script and make it executable
COPY entry.sh /
RUN chmod +x /entry.sh

# Copy the main Python script
COPY face-recognition-code.py ${FUNCTION_DIR}
COPY data.pt ${FUNCTION_DIR}

# Set the entry point and default command
ENTRYPOINT [ "/entry.sh" ]
CMD [ "face-recognition-code.lambda_handler" ]

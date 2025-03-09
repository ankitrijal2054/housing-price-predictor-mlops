FROM python:3.9

# Set the working directory
WORKDIR /app

# Copy the necessary file
COPY fastapi-app/ src/
COPY models/xgboost_model.json models/

# Install the dependencies
RUN pip install --no-cache-dir -r src/requirements.txt

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
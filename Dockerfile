FROM python:3.9-alpine

# Copy only the requirements file first to leverage Docker caching
COPY requirements.txt .

# Upgrade PIP
RUN pip install --upgrade pip

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire application code
COPY . .

# Expose the port your application will run on
EXPOSE 80

# Specify the command to run on container start
CMD ["python", "main.py"]
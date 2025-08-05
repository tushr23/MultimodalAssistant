
# GenAI Chatbot API

A production-ready Generative AI Chatbot API built with Python, FastAPI, Hugging Face Transformers, and SQLite. Accepts user questions via REST API, generates smart answers using GPT-2, and logs every conversation. Easily deployable on AWS EC2 for real-world applications.

![CI](https://github.com/tushr23/genai-chatbot-api/actions/workflows/ci.yml/badge.svg)
![Coverage](https://img.shields.io/badge/coverage-100%25-brightgreen)

## Features
- Accepts user questions via API
- Generates answers using GPT-2 (Hugging Face)
- Logs conversations to a database
- Endpoint to view all logs
- Easy cloud deployment (AWS EC2)
- Well-documented code and API

## Tech Stack
- Python 3.10+
- FastAPI
- Hugging Face Transformers
- SQLite
- Uvicorn
- AWS EC2

## Setup Instructions

1. **Clone the repo:**
   ```bash
   git clone https://github.com/tushr23/genai-chatbot-api.git
   cd genai-chatbot-api
   ```

2. Install requirements:
   ```
   pip install -r requirements.txt
   ```

3. **Run the app locally:**
   ```bash
   python -m uvicorn main:app --reload
   ```


## Running Automated Tests

To run all automated tests and check coverage:

```bash
pytest --cov=.
```

This will show you which lines of code are covered by tests. The CI workflow also runs these tests automatically on every push or pull request.

## Run with Docker (Recommended for Everyone)

Docker makes it easy to run this app anywhere, with no setup headaches. Just follow these steps:

1. **Install Docker Desktop:**
   - Download from https://www.docker.com/products/docker-desktop/
   - Install and start Docker on your computer.

2. **Build the Docker image:**
   ```bash
   docker build -t genai-chatbot-api .
   ```
   - This command creates a portable image of your app with all dependencies.

3. **Run the app in Docker:**
   ```bash
   docker run -p 8000:8000 genai-chatbot-api
   ```
   - This starts your app in a container. Access it at http://localhost:8000

4. **Stop the app:**
   - Press `Ctrl+C` in the terminal, or run `docker ps` and `docker stop <container_id>` in another terminal.

**Why use Docker?**
- No need to install Python or libraries on your computer.
- Works the same on any machine or cloud.
- Makes your project look professional and production-ready.

## API Usage

- **POST /chat**
  - Request: `{ "question": "Your question here" }`
  - Response: `{ "answer": "Chatbot's answer" }`

- **GET /logs**
  - Returns all saved conversations.

- **Interactive Docs:** Visit `/docs` for Swagger UI.


## How to Deploy on AWS EC2 (with Docker)

Deploying with Docker is the easiest and most reliable way to run your app in the cloud. Here’s how to do it step by step:

1. **Launch an EC2 instance (Ubuntu):**
   - Go to the AWS Console → EC2 → Launch Instance.
   - Choose Ubuntu Server 22.04 LTS (free tier eligible).
   - Select t2.micro (free tier).
   - Add a security group rule to allow TCP port 80 (for web traffic) and SSH (22) from your IP.

2. **Connect to your EC2 instance:**
   ```bash
   ssh -i path/to/your-key.pem ubuntu@your-ec2-public-dns
   ```
   - Replace with your key and EC2 DNS.

3. **Install Docker on EC2:**
   ```bash
   sudo apt update
   sudo apt install -y docker.io
   sudo systemctl start docker
   sudo systemctl enable docker
   sudo usermod -aG docker ubuntu
   ```
   - Log out and log back in to enable Docker for your user.

4. **Get your project onto EC2:**
   - **Option 1:** Copy from your computer:
     ```bash
     scp -i path/to/your-key.pem -r path/to/genai-chatbot-api ubuntu@your-ec2-public-dns:~
     ```
   - **Option 2:** Clone from GitHub:
     ```bash
     git clone https://github.com/tushr23/genai-chatbot-api.git
     cd genai-chatbot-api
     ```

5. **Build and run your app with Docker:**
   ```bash
   docker build -t genai-chatbot-api .
   docker run -d -p 80:8000 genai-chatbot-api
   ```
   - `-d` runs the app in the background.
   - `-p 80:8000` maps public port 80 to your app’s port 8000.

6. **Open your app in the browser:**
   - Go to `http://your-ec2-public-dns/` in your browser.
   - You should see your FastAPI docs at `/docs`.

**Tip:** If you update your code, just repeat steps 5 to rebuild and restart your app.

## Example Request

```bash
curl -X POST "http://your-ec2-public-dns/chat" -H "Content-Type: application/json" -d "{\"question\": \"What is AI?\"}"
```

## Test Coverage

This project uses `pytest` and `pytest-cov` for automated testing and coverage reporting. All endpoints and error cases are covered by tests in `test_api.py`.

CI/CD is set up with GitHub Actions. Every code change is automatically tested and checked for coverage.

## Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

## License

MIT

## Credits

Built by Tushr Verma

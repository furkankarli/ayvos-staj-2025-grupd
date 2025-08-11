# AI Fashion Search

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![Framework](https://img.shields.io/badge/Framework-FastAPI-green.svg)](https://fastapi.tiangolo.com/)
[![Powered by](https://img.shields.io/badge/Powered%20by-OpenCLIP%20%26%20FAISS-orange.svg)](https://github.com/mlfoundations/open_clip)

A visual-based fashion product search engine. This project allows you to find visually similar products in the Fashion-MNIST dataset by uploading a product image. Developed using FastAPI, OpenCLIP, and FAISS.

---

### [Live Demo](http://your-live-demo-url.com) &middot; [API Docs](/docs) &middot; [Report Bug](https://github.com/furkankarli/ayvos-staj-2025-grupd/issues)

![Fashion Search Screenshot](https://via.placeholder.com/800x400.png?text=App+Screenshot+Here)
*A placeholder for the application's user interface.*

## Key Features

-   **Visual Search**: Find similar products by uploading an image.
-   **High Speed**: Get results in seconds thanks to the highly optimized FAISS index.
-   **Accurate Results**: High-accuracy visual recognition with the state-of-the-art OpenCLIP model.
-   **User-Friendly Interface**: A clean and intuitive web interface built with Tailwind CSS.
-   **RESTful API**: Provides a well-documented API for integration with external applications.

## Technology Stack

This project leverages a modern stack to deliver a high-performance AI application.

-   **Backend**:
    -   **FastAPI**: A modern, high-performance web framework for building APIs with Python. Chosen for its speed, automatic documentation, and ease of use.
    -   **Uvicorn**: A lightning-fast ASGI server, used to run the FastAPI application.
-   **Artificial Intelligence**:
    -   **PyTorch**: The core machine learning framework used for tensor computations and model management.
    -   **OpenCLIP**: A high-performance implementation of OpenAI's CLIP model, used for generating robust image embeddings.
-   **Vector Search**:
    -   **FAISS**: (Facebook AI Similarity Search) A library for efficient similarity search and clustering of dense vectors. It's essential for achieving sub-second search times on large datasets.
-   **Frontend**:
    -   **HTML5 & JavaScript**: Standard web technologies for the user interface.
    -   **Tailwind CSS**: A utility-first CSS framework for rapidly building custom designs.
-   **Utilities**:
    -   **Loguru**: A library for making logging in Python simple and enjoyable.
    -   **python-dotenv**: For managing environment variables.

## Project Structure

```
fashion_search/
├── app/
│   ├── __init__.py
│   ├── main.py           # FastAPI app, API endpoints
│   ├── models.py         # AI models, search engine logic
│   └── utils.py          # Configuration and utilities
├── data/
│   ├── embeddings/       # Stored FAISS index and embeddings
│   └── fashion/          # Fashion-MNIST dataset
├── logs/
│   └── app.log           # Application logs
├── static/
│   ├── css/
│   └── js/
├── templates/
│   ├── base.html
│   ├── index.html
│   └── results.html
├── .env.example          # Example environment variables
├── requirements.txt
└── README.md
```

## Installation

Follow the steps below to run the project on your local machine.

### Prerequisites

-   Python 3.8 or higher
-   `pip` package manager

### Steps

1.  **Clone the project:**
    ```bash
    git clone https://github.com/furkankarli/ayvos-staj-2025-grupd.git
    cd ayvos-staj-2025-grupd/03-TeamProjects/fashion_search
    ```
2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```
3.  **Install the required libraries:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Configure your environment:**
    -   Rename `.env.example` to `.env`.
    -   Modify the variables in `.env` if needed.
5.  **Run the application:**
    ```bash
    python app/main.py
    ```
    The application will start at `http://127.0.0.1:8000`. The first run may take a few minutes to download the dataset and compute the embeddings.

## API Endpoints

-   `GET /`: Displays the home page.
-   `POST /api/search`: Searches for similar products by uploading an image.
-   `GET /api/health`: Checks the health status of the system.
-   `GET /docs`: Opens the Swagger UI interface for API documentation.


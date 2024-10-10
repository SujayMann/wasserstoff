# PDF Metadata Extractor

This project is a PDF metadata extractor that downloads PDFs from specified URLs, extracts metadata, summarizes the content, and extracts keywords. It leverages MongoDB for storage and employs a multithreaded approach for efficient processing.

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Installation](#installation)
3. [Usage](#usage)
4. [How It Works](#how-it-works)
5. [Deployment](#deployment)

## System Requirements

- **Python**: 3.7 or higher
- **MongoDB**: MongoDB Atlas account (or a local MongoDB instance)
- **Pip**: Python package manager
- **Git**: For cloning the repository
- **NLTK**: Natural Language Toolkit for text processing (automatically downloaded)

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/SujayMann/wasserstoff
   cd wasserstoff
   ```
Install required packages:

It is recommended to create a virtual environment:

```bash
  python -m venv venv
  source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```
Install the dependencies:

```
pip install -r requirements.txt
```
Set up MongoDB:

Create a MongoDB Atlas account and set up a cluster.

Add your IP address to the IP whitelist to allow access.

Create a database named pdf_database and a collection named pdfs.

Obtain your connection string from the MongoDB Atlas dashboard and store it as an environment variable:

Download NLTK Resources:

The application requires certain NLTK resources, which are automatically downloaded when the script runs for the first time.

## Usage
Prepare your Dataset:

Create a Dataset.json file in the root directory. The format should be as follows:

```json
{
    "pdf1": "http://example.com/file1.pdf",
    "pdf2": "http://example.com/file2.pdf",
}
```
Run the Application:

Execute the main script:

```bash
python main.py
```
The script performs the following:

Downloads PDFs from the specified URLs.

Stores metadata (name, filepath, size) in the MongoDB database.

Summarizes the content of each PDF.

Extracts keywords from the PDF content.

Updates the MongoDB database with the summaries and keywords.

## How It Works
The application consists of the following main components:

PDF Downloading: Uses multithreading to download multiple PDFs concurrently.

Metadata Storage: Stores essential metadata in a MongoDB database for future reference.

Content Summarization: Utilizes the T5 transformer model from Hugging Face to generate concise summaries of the PDF content based on its length.

Keyword Extraction: Implements TF-IDF to extract relevant keywords from the text, helping to identify key topics within the document.

Error Handling: Logs errors during processing and ensures that resources are properly managed.

## Deployment
To deploy this application using Render:

Create a Render Account: Sign up at Render.

Create a New Web Service: Connect your GitHub repository containing this project.

Configure the Web Service:

Set the build command to:
```
pip install -r requirements.txt
```
Set the start command to:
```
python main.py
```
Add Environment Variables:

Add your MongoDB URI as an environment variable (MONGODB_URI).

Deploy the Service: Click on "Create Web Service" to start the deployment.

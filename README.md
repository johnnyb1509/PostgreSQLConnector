# 🚀 PostgresConnector (Ultimate Edition)

## 📖 Introduction
Welcome to **PostgresConnector**, the ultimate database connection package built for our team's data engineering and AI workflows. 

This package simplifies interactions with PostgreSQL databases by automating tedious tasks like schema evolution, data type mapping, and bulk upserts. It goes beyond standard SQL by providing native, out-of-the-box support for **TimescaleDB** (for time-series data) and **pgvector** (for AI embeddings).

### ✨ Key Features
* **Smart Upsert (`ON CONFLICT DO UPDATE`):** Blazing fast data ingestion with conflict resolution strategies (`last`, `sum`, `skip`).
* **Auto Schema Evolution:** Automatically adds missing columns to your database tables based on your Pandas DataFrames.
* **Native JSONB Support:** Automatically detects nested Python dictionaries/lists and maps them to PostgreSQL `JSONB` format.
* **TimescaleDB Integration:** Easily convert standard tables into hypertables for optimized time-series data storage.
* **pgvector for AI:** Automatically detects lists of floats (embeddings) and creates Vector columns with HNSW/IVFFlat indexing for fast similarity searches.

---

## 📂 Directory Structure

This project is managed using [Poetry](https://python-poetry.org/). The standard structure looks like this:

```text
PostgreSQLConnector/
│
├── pyproject.toml           # Poetry configuration, metadata, and dependencies
├── README.md                # This documentation file
├── src/       # The actual Python module
│   ├── __init__.py
│   └── postgres_connector.py
└── notebooks/               # (Optional) Tutorials and examples
    └── Tutorial.ipynb

```

## 💻 Installation
This package is published on PyPI. You can easily install it into your project using your preferred package manager.

Using Poetry (Recommended):

```bash
poetry add PostgreSQLConnector
```

Using pip:

```bash
pip install PostgreSQLConnector
```

## 🛠️ Dependencies
This package relies on several powerful Python libraries to function properly.

```pandas``` - For data manipulation and structures.

```SQLAlchemy``` - For database connection and ORM capabilities.

```psycopg2-binary``` - The most popular PostgreSQL adapter for Python.

```pgvector``` - For handling vector data types and AI embeddings in SQLAlchemy.

```loguru``` - For beautiful, easy-to-read logging.


## 🚀 Quick Start
Here is a quick example of how to connect and upsert data using the connector:

```python
import pandas as pd
from postgres_connector import PostgresConnector

# 1. Initialize the connection
pg = PostgresConnector(
    host='localhost', 
    database='my_database', 
    username='my_user', 
    password='my_password'
)

# 2. Prepare your data
data = {
    'id': [1, 2],
    'name': ['Alice', 'Bob'],
    'role': ['Admin', 'User']
}
df = pd.DataFrame(data)

# 3. Upsert into the database (Creates table if it doesn't exist!)
pg.upsert_data(
    df=df, 
    target_table='team_members', 
    primary_key='id'
)

# 4. Close the connection
pg.dispose()
```
For more advanced use cases, including **TimescaleDB** and **pgvector** for AI embeddings, please refer to the Tutorial.ipynb file included in this repository.

## 👨‍💻 Creator
Created by: Nguyen Minh Son, CQF (MinhSonCQF)

Contact / Support: nguyen.minhson1511@gmail.com

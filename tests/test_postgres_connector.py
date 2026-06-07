import pytest
import pandas as pd
from sqlalchemy.dialects.postgresql import JSONB, DOUBLE_PRECISION, TEXT, BIGINT
from pgvector.sqlalchemy import Vector
from src.postgres_connector import PostgresConnector


# Fix: no trailing commas — those would create tuples, not strings/ints
PG_HOST = "localhost"
PG_DB = "test_db"
PG_USER = "postgres"       # Change if your DB user is different
PG_PASSWORD = "password"   # Change if your DB password is different
PG_PORT = 5432

# ==========================================
# 1. UNIT TESTS (no real database required)
# ==========================================

def test_generate_dtype_mapping():
    """Verify automatic dtype detection maps DataFrame columns to correct SA types."""
    # PostgresConnector can be instantiated with dummy credentials because
    # SQLAlchemy creates the engine lazily (no real connection until first use).
    pg = PostgresConnector(
        host="dummy", database="dummy", username="dummy", password="dummy"
    )

    df = pd.DataFrame({
        "id": [1, 2],
        "name": ["Alice", "Bob"],
        "score": [9.5, 8.0],
        "metadata": [{"role": "admin"}, {"role": "user"}],   # dict  → JSONB
        "embedding": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],    # list  → VECTOR(3)
    })

    mapping = pg._generate_dtype_mapping(df)

    assert isinstance(mapping["id"], BIGINT)
    assert isinstance(mapping["name"], TEXT)
    assert isinstance(mapping["score"], DOUBLE_PRECISION)
    assert isinstance(mapping["metadata"], JSONB)
    assert isinstance(mapping["embedding"], Vector)
    assert mapping["embedding"].dim == 3


def test_safe_chunk_size():
    """_safe_chunk_size must stay within PostgreSQL's 32 767 bind-param limit."""
    assert PostgresConnector._safe_chunk_size(0) == 500
    assert PostgresConnector._safe_chunk_size(1) == 500
    assert PostgresConnector._safe_chunk_size(100) == 320     # 32000 // 100
    assert PostgresConnector._safe_chunk_size(1000) == 32     # 32000 // 1000
    # Edge: more cols than params would allow → at least 1
    assert PostgresConnector._safe_chunk_size(40_000) == 1


def test_lower_df_columns():
    """Column names must be lowercased."""
    pg = PostgresConnector(
        host="dummy", database="dummy", username="dummy", password="dummy"
    )
    df = pd.DataFrame({"ID": [1], "Name": [2], "SCORE": [3]})
    out = pg._lower_df_columns(df)
    assert list(out.columns) == ["id", "name", "score"]


def test_clean_records_types():
    """_clean_records must convert numpy scalars and NaT to plain Python types."""
    import numpy as np

    pg = PostgresConnector(
        host="dummy", database="dummy", username="dummy", password="dummy"
    )
    df = pd.DataFrame({
        "Int_Col": pd.array([1], dtype="Int64"),
        "Float_Col": [float("nan")],
        "Ts_Col": [pd.Timestamp("2024-01-01")],
    })
    records = df.to_dict(orient="records")
    cleaned = pg._clean_records(records)

    assert cleaned[0]["float_col"] is None            # NaN → None
    assert isinstance(cleaned[0]["ts_col"], __import__("datetime").datetime)


# ==========================================
# 2. INTEGRATION TESTS (require a real DB)
# ==========================================

@pytest.fixture(scope="module")
def pg_connector():
    """
    Create a real connection to the test database.
    Runs once for the whole module.  Change credentials above as needed.
    """
    connector = PostgresConnector(
        host=PG_HOST,
        database=PG_DB,
        username=PG_USER,
        password=PG_PASSWORD,
        port=PG_PORT,
    )
    yield connector

    # Teardown
    connector.execute_query("DROP TABLE IF EXISTS test_users CASCADE;")
    connector.dispose()


def test_upsert_data_insert_new(pg_connector):
    """Create a new table and insert rows via upsert."""
    df = pd.DataFrame({
        "id": [1, 2],
        "name": ["Alice", "Bob"],
        "balance": [100.0, 200.0],
    })

    pg_connector.upsert_data(df, target_table="test_users", primary_key="id")

    result_df = pg_connector.get_data("SELECT * FROM test_users ORDER BY id;")
    assert len(result_df) == 2
    assert result_df.iloc[0]["name"] == "Alice"


def test_upsert_data_conflict_update_and_schema_evolve(pg_connector):
    """Upsert should update existing rows and add new columns automatically."""
    df_updated = pd.DataFrame({
        "id": [2, 3],
        "name": ["Bob Updated", "Charlie"],
        "balance": [250.0, 300.0],
        "age": [30, 25],   # new column — triggers schema evolution
    })

    pg_connector.upsert_data(
        df=df_updated,
        target_table="test_users",
        primary_key="id",
        conflict_strategy="last",
    )

    result_df = pg_connector.get_data("SELECT * FROM test_users ORDER BY id;")

    bob_row = result_df[result_df["id"] == 2].iloc[0]
    assert bob_row["name"] == "Bob Updated"
    assert bob_row["balance"] == 250.0

    assert "age" in result_df.columns
    assert result_df[result_df["id"] == 3].iloc[0]["age"] == 25


def test_get_data_stream(pg_connector):
    """get_data with stream=True must return the same data as without streaming."""
    normal = pg_connector.get_data("SELECT * FROM test_users ORDER BY id;")
    streamed = pg_connector.get_data(
        "SELECT * FROM test_users ORDER BY id;", stream=True
    )
    assert list(normal["id"]) == list(streamed["id"])

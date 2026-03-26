import pytest
import pandas as pd
from sqlalchemy.dialects.postgresql import JSONB, DOUBLE_PRECISION, TEXT, BIGINT
from pgvector.sqlalchemy import Vector
from src.postgres_connector import PostgresConnector


PG_HOST='localhost', 
PG_DB='test_db',
PG_USER='postgres', # Change here if your DB user is different
PG_PASSWORD='password', # Change here if your DB password is different
PG_PORT=5432

# ==========================================
# 1. UNIT TESTS (Không cần database thật)
# ==========================================

def test_generate_dtype_mapping():
    """Kiểm tra xem tính năng tự động nhận diện kiểu dữ liệu có chuẩn không."""
    # Khởi tạo connector với thông tin dummy (vì hàm này không gọi DB)
    pg = PostgresConnector(
        host='dummy', database='dummy', username='dummy', password='dummy'
    )
    
    # Tạo dữ liệu giả lập bao gồm cả JSON và Vector
    df = pd.DataFrame({
        'id': [1, 2],
        'name': ['Alice', 'Bob'],
        'score': [9.5, 8.0],
        'metadata': [{'role': 'admin'}, {'role': 'user'}],  # Dict -> JSONB
        'embedding': [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]     # List of floats -> VECTOR(3)
    })
    
    mapping = pg._generate_dtype_mapping(df)
    
    # Kiểm tra các mapping có đúng kiểu của SQLAlchemy/Postgres không
    assert isinstance(mapping['id'], BIGINT)
    assert isinstance(mapping['name'], TEXT)
    assert isinstance(mapping['score'], DOUBLE_PRECISION)
    assert isinstance(mapping['metadata'], JSONB)
    assert isinstance(mapping['embedding'], Vector)
    assert mapping['embedding'].dim == 3  # Đảm bảo nhận diện đúng số chiều vector

# ==========================================
# 2. INTEGRATION TESTS (Cần database thật)
# ==========================================

@pytest.fixture(scope="module")
def pg_connector():
    """
    Fixture này sẽ tạo một kết nối thật tới Test Database.
    Nó sẽ chạy 1 lần cho toàn bộ module test này.
    Vui lòng thay đổi thông tin dưới đây thành DB test của bạn.
    """
    connector = PostgresConnector(
        host=PG_HOST, 
        database=PG_DB,   # Hãy tạo sẵn 1 database tên là test_db trong máy bạn
        username=PG_USER, 
        password=PG_PASSWORD,
        port=PG_PORT
    )
    yield connector
    
    # Dọn dẹp sau khi test xong
    connector.execute_query('DROP TABLE IF EXISTS test_users CASCADE;')
    connector.dispose()

def test_upsert_data_insert_new(pg_connector):
    """Kiểm tra khả năng tạo bảng mới và Insert dữ liệu."""
    df = pd.DataFrame({
        'id': [1, 2],
        'name': ['Alice', 'Bob'],
        'balance': [100.0, 200.0]
    })
    
    # Test Upsert
    pg_connector.upsert_data(df, target_table='test_users', primary_key='id')
    
    # Đọc lại từ DB để xác minh
    result_df = pg_connector.get_data('SELECT * FROM test_users ORDER BY id;')
    
    assert len(result_df) == 2
    assert result_df.iloc[0]['name'] == 'Alice'

def test_upsert_data_conflict_update_and_schema_evolve(pg_connector):
    """Kiểm tra khả năng Cập nhật (Upsert) và thêm cột mới (Schema Evolution)."""
    # Bob đổi tên thành 'Bob Updated', có thêm dòng số 3, và xuất hiện cột 'age'
    df_updated = pd.DataFrame({
        'id': [2, 3],
        'name': ['Bob Updated', 'Charlie'],
        'balance': [250.0, 300.0],
        'age': [30, 25]  # Cột mới!
    })
    
    pg_connector.upsert_data(
        df=df_updated, 
        target_table='test_users', 
        primary_key='id',
        conflict_strategy='last' # Ghi đè dòng cũ
    )
    
    result_df = pg_connector.get_data('SELECT * FROM test_users ORDER BY id;')
    
    # Kiểm tra dòng số 2 (Bob) đã được update chưa
    bob_row = result_df[result_df['id'] == 2].iloc[0]
    assert bob_row['name'] == 'Bob Updated'
    assert bob_row['balance'] == 250.0
    
    # Kiểm tra cột mới 'age' đã được tự động thêm vào DB chưa
    assert 'age' in result_df.columns
    assert result_df[result_df['id'] == 3].iloc[0]['age'] == 25
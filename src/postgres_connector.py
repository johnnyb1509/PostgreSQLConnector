import os
import pandas as pd
import numpy as np
import uuid
from typing import List, Optional, Dict, Union, Literal
from loguru import logger
from sqlalchemy import create_engine, text, URL, inspect, MetaData, Table
from sqlalchemy.types import BIGINT, DATE
from sqlalchemy.dialects.postgresql import JSONB, DOUBLE_PRECISION, TEXT, TIMESTAMP, insert
from pgvector.sqlalchemy import Vector # Cần cài đặt thư viện pgvector

class PostgresConnector:
    """
    Trình kết nối PostgreSQL chuẩn hóa (Ultimate Version).
    
    Features:
    - Core: Fast Execute, Schema Evolution, Native PostgreSQL types (TEXT, JSONB).
    - Upsert: Sử dụng ON CONFLICT DO UPDATE nguyên bản của Postgres.
    - Extensions: Hỗ trợ TimescaleDB (Hypertable) cho Time-series và pgvector cho AI.
    - Strategies: 'last' (Update), 'skip' (Ignore), 'sum' (Aggregate numeric).
    """

    def __init__(self, host: str, database: str, 
                 username: str, password: str, port: int = 5432,
                 **kwargs):
        self.host = host
        self.database = database
        self.username = username
        self.password = password
        self.port = port
        
        self.connection_url = URL.create(
            "postgresql+psycopg2",
            username=self.username,
            password=self.password,
            host=self.host,
            port=self.port,
            database=self.database
        )
        
        # Postgres driver tự động xử lý executemany tối ưu bằng execute_values nếu được cấu hình
        self.engine = create_engine(
                    self.connection_url,
                    pool_pre_ping=True,
                    insertmanyvalues_page_size=10000 # Tối ưu hóa bulk insert (cú pháp mới)
                )

    def execute_query(self, query: str, params: Optional[Dict] = None):
        """Thực thi lệnh không trả về dữ liệu (VD: CREATE EXTENSION, DROP TABLE)"""
        try:
            with self.engine.begin() as conn:
                conn.execute(text(query), params or {})
        except Exception as e:
            logger.error(f"Execute query error: {e}")
            raise e

    def get_data(self, query: str, params: Optional[Dict] = None) -> pd.DataFrame:
        try:
            with self.engine.connect() as conn:
                return pd.read_sql(text(query), conn, params=params)
        except Exception as e:
            logger.error(f"Get data error: {e}")
            raise e

    def _generate_dtype_mapping(self, df: pd.DataFrame) -> Dict:
        """Tự động map kiểu dữ liệu, tích hợp nhận diện JSONB và VECTOR"""
        dtype_map = {}
        for col in df.columns:
            sample_val = df[col].dropna().iloc[0] if not df[col].dropna().empty else None
            
            # 1. Nhận diện Vector (List các số thực)
            if isinstance(sample_val, list) and all(isinstance(x, (int, float)) for x in sample_val):
                dim = len(sample_val)
                dtype_map[col] = Vector(dim)
            # 2. Nhận diện JSONB (Dict hoặc List chứa Dict/String)
            elif isinstance(sample_val, (dict, list)):
                dtype_map[col] = JSONB()
            # 3. Các kiểu dữ liệu cơ bản
            elif pd.api.types.is_string_dtype(df[col]) or df[col].dtype == 'object':
                dtype_map[col] = TEXT()
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                dtype_map[col] = TIMESTAMP()
            elif pd.api.types.is_float_dtype(df[col]):
                dtype_map[col] = DOUBLE_PRECISION()
            elif pd.api.types.is_integer_dtype(df[col]):
                dtype_map[col] = BIGINT()
                
        return dtype_map

    def _get_table_columns(self, table_name: str, conn) -> List[str]:
        inspector = inspect(conn)
        return [col['name'] for col in inspector.get_columns(table_name)]

    def _add_missing_columns(self, table_name: str, missing_cols: List[str], dtype_map: Dict, conn):
        for col in missing_cols:
            col_type = dtype_map.get(col, TEXT())
            type_str = "TEXT"
            
            if isinstance(col_type, DOUBLE_PRECISION): type_str = "DOUBLE PRECISION"
            elif isinstance(col_type, BIGINT): type_str = "BIGINT"
            elif isinstance(col_type, TIMESTAMP): type_str = "TIMESTAMP"
            elif isinstance(col_type, JSONB): type_str = "JSONB"
            elif isinstance(col_type, Vector): type_str = f"VECTOR({col_type.dim})"
            
            conn.execute(text(f'ALTER TABLE "{table_name}" ADD COLUMN "{col}" {type_str}'))
            logger.info(f"Auto-evolve: Added column '{col}' to '{table_name}'")

    # ==========================================
    # DATABASE EXTENSIONS (Timescale & pgvector)
    # ==========================================

    def setup_extensions(self):
        """Kích hoạt các extension cần thiết cho Database"""
        self.execute_query("CREATE EXTENSION IF NOT EXISTS vector;")
        # Nếu Postgres đã cài sẵn TimescaleDB plugin, ta có thể tạo extension:
        self.execute_query("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;")
        logger.info("Checked and created 'vector' & 'timescaledb' extensions if applicable.")

    def enable_timescaledb(self, table_name: str, time_column: str, chunk_time_interval: str = '1 day'):
        """
        Chuyển đổi bảng thành Hypertable của TimescaleDB.
        Phù hợp cho dữ liệu crawl hàng ngày, logs, IoT, chứng khoán.
        """
        try:
            sql = f"""
            SELECT create_hypertable('{table_name}', '{time_column}', 
                   chunk_time_interval => INTERVAL '{chunk_time_interval}', 
                   if_not_exists => TRUE);
            """
            self.execute_query(sql)
            logger.success(f"Converted {table_name} to TimescaleDB Hypertable on column {time_column}.")
        except Exception as e:
            logger.error(f"Could not convert to Hypertable (is TimescaleDB installed?): {e}")

    def create_vector_index(self, table_name: str, vector_column: str, index_type: Literal['hnsw', 'ivfflat'] = 'hnsw'):
        """Tạo index cho cột Vector để tìm kiếm semantic search nhanh hơn"""
        index_name = f"idx_{table_name}_{vector_column}_{index_type}"
        try:
            if index_type == 'hnsw':
                sql = f'CREATE INDEX IF NOT EXISTS {index_name} ON "{table_name}" USING hnsw ("{vector_column}" vector_l2_ops);'
            else:
                sql = f'CREATE INDEX IF NOT EXISTS {index_name} ON "{table_name}" USING ivfflat ("{vector_column}" vector_l2_ops) WITH (lists = 100);'
            self.execute_query(sql)
            logger.success(f"Created {index_type} index on {table_name}.{vector_column}")
        except Exception as e:
            logger.error(f"Failed to create vector index: {e}")

    # ==========================================
    # CORE OPERATIONS (Upsert, Replace)
    # ==========================================

    def upsert_data(self, 
                    df: pd.DataFrame, 
                    target_table: str, 
                    primary_key: Union[str, List[str]] = None, 
                    auto_evolve_schema: bool = True,
                    conflict_strategy: Literal['sum', 'last', 'skip'] = 'last'):
        """
        Upsert sử dụng native 'ON CONFLICT DO UPDATE' của PostgreSQL.
        """
        if df.empty: return
        df = df.copy()

        join_keys = [primary_key] if isinstance(primary_key, str) else primary_key
        if not join_keys:
             logger.warning(f"No keys provided. Switch to APPEND mode.")
             df.to_sql(target_table, self.engine, if_exists='append', index=False)
             return

        # Ép kiểu datetime
        for col in df.select_dtypes(include=['object', 'str']):
            if df[col].astype(str).str.match(r'^\d{4}-\d{2}-\d{2}').any():
                df[col] = pd.to_datetime(df[col], errors='ignore')

        dtype_mapping = self._generate_dtype_mapping(df)

        try:
            with self.engine.begin() as conn:
                # 1. Tạo bảng nếu chưa có
                inspector = inspect(conn)
                if not inspector.has_table(target_table):
                    df.head(0).to_sql(target_table, conn, index=False, dtype=dtype_mapping)
                    pk_str = ", ".join([f'"{c}"' for c in join_keys])
                    conn.execute(text(f'ALTER TABLE "{target_table}" ADD PRIMARY KEY ({pk_str})'))
                    logger.info(f"Created new table {target_table} with PK {join_keys}")

                # 2. Schema Evolution
                db_cols = self._get_table_columns(target_table, conn)
                new_cols = [c for c in df.columns if c.lower() not in [dc.lower() for dc in db_cols]]
                if new_cols and auto_evolve_schema:
                    self._add_missing_columns(target_table, new_cols, dtype_mapping, conn)
                elif new_cols:
                    df = df.drop(columns=new_cols)

                # 3. Tạo câu lệnh Upsert (Sử dụng SQLAlchemy postgresql.insert)
                table_cols = df.columns.tolist()
                records = df.to_dict(orient='records')
                
                # Load cấu trúc bảng thực tế từ Database thành Table object
                metadata_obj = MetaData()
                target_table_obj = Table(target_table, metadata_obj, autoload_with=conn)
                
                # Truyền Table object vào hàm insert
                insert_stmt = insert(target_table_obj).values(records)
                
                if conflict_strategy == 'skip':
                    upsert_stmt = insert_stmt.on_conflict_do_nothing(index_elements=join_keys)
                else:
                    update_dict = {
                        col: insert_stmt.excluded[col] 
                        for col in table_cols if col not in join_keys
                    }
                    
                    if conflict_strategy == 'sum':
                        for col in update_dict:
                            if pd.api.types.is_numeric_dtype(df[col]):
                                update_dict[col] = text(f'"{target_table}"."{col}" + EXCLUDED."{col}"')

                    upsert_stmt = insert_stmt.on_conflict_do_update(
                        index_elements=join_keys,
                        set_=update_dict
                    )

                conn.execute(upsert_stmt)
                logger.success(f"Upserted {len(df)} rows to {target_table} (Strategy: {conflict_strategy})")

        except Exception as e:
            logger.error(f"Upsert failed for {target_table}: {e}")
            raise e

    def dispose(self):
        self.engine.dispose()
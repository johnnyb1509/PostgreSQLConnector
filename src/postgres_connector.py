import os
import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Union, Literal
from loguru import logger
from sqlalchemy import create_engine, text, URL, inspect, MetaData, Table
from sqlalchemy.types import BIGINT, DATE
from sqlalchemy.dialects.postgresql import JSONB, DOUBLE_PRECISION, TEXT, TIMESTAMP, insert
from pgvector.sqlalchemy import Vector

class PostgresConnector:
    """
    Trình kết nối PostgreSQL chuẩn hóa (Ultimate Version).
    """

    def __init__(self, host: str, database: str, 
                 username: str, password: str, port: int = 5432,
                 schema: str = "public", 
                 **kwargs):
        self.host = host
        self.database = database
        self.username = username
        self.password = password
        self.port = port
        self.schema = schema 
        
        self.connection_url = URL.create(
            "postgresql+psycopg2",
            username=self.username,
            password=self.password,
            host=self.host,
            port=self.port,
            database=self.database
        )
        
        self.engine = create_engine(
                    self.connection_url,
                    pool_pre_ping=True,
                    insertmanyvalues_page_size=10000,
                    connect_args={'options': f'-csearch_path={self.schema}'} 
                )

    def execute_query(self, query: str, params: Optional[Dict] = None):
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
        dtype_map = {}
        for col in df.columns:
            sample_val = df[col].dropna().iloc[0] if not df[col].dropna().empty else None
            
            if isinstance(sample_val, list) and all(isinstance(x, (int, float)) for x in sample_val):
                dim = len(sample_val)
                dtype_map[col] = Vector(dim)
            elif isinstance(sample_val, (dict, list)):
                dtype_map[col] = JSONB()
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
        return [col['name'] for col in inspector.get_columns(table_name, schema=self.schema)]

    def _add_missing_columns(self, table_name: str, missing_cols: List[str], dtype_map: Dict, conn):
        for col in missing_cols:
            col_type = dtype_map.get(col, TEXT())
            type_str = "TEXT"
            
            if isinstance(col_type, DOUBLE_PRECISION): type_str = "DOUBLE PRECISION"
            elif isinstance(col_type, BIGINT): type_str = "BIGINT"
            elif isinstance(col_type, TIMESTAMP): type_str = "TIMESTAMP"
            elif isinstance(col_type, JSONB): type_str = "JSONB"
            elif isinstance(col_type, Vector): type_str = f"VECTOR({col_type.dim})"
            
            conn.execute(text(f'ALTER TABLE "{self.schema}"."{table_name}" ADD COLUMN "{col}" {type_str}'))
            logger.info(f"Auto-evolve: Added column '{col}' to '{self.schema}.{table_name}'")

    def _clean_records(self, records: List[Dict]) -> List[Dict]:
        """
        THE MASTER SANITIZER.
        Converts ALL Pandas/Numpy types to pure Python primitives.
        """
        clean_records = []
        for rec in records:
            clean_rec = {}
            for k, v in rec.items():
                safe_k = str(k).lower()
                
                if pd.isna(v):
                    clean_rec[safe_k] = None
                elif isinstance(v, pd.Timestamp):
                    clean_rec[safe_k] = v.to_pydatetime()
                elif hasattr(v, 'item'): 
                    clean_rec[safe_k] = v.item()
                else:
                    clean_rec[safe_k] = v
            clean_records.append(clean_rec)
        return clean_records

    # ==========================================
    # CORE OPERATIONS
    # ==========================================
    def replace_table(self, df: pd.DataFrame, target_table: str, primary_key: Union[str, List[str]] = None):
        if df.empty: return
        dtype_mapping = self._generate_dtype_mapping(df)
        
        try:
            with self.engine.begin() as conn:
                logger.info(f"Replacing table '{target_table}'...")
                df.iloc[:0].to_sql(target_table, conn, schema=self.schema, if_exists='replace', index=False, dtype=dtype_mapping)
                
                if primary_key:
                    pk_cols = [primary_key] if isinstance(primary_key, str) else primary_key
                    pk_str = ", ".join([f'"{c}"' for c in pk_cols])
                    conn.execute(text(f'ALTER TABLE "{self.schema}"."{target_table}" ADD PRIMARY KEY ({pk_str})'))
                
                metadata_obj = MetaData()
                target_table_obj = Table(target_table, metadata_obj, schema=self.schema, autoload_with=conn)
                
                # SAFE CHUNK SIZE FOR POSTGRESQL (32,767 param limit)
                chunk_size = 500 
                for i in range(0, len(df), chunk_size):
                    raw_records = df.iloc[i : i + chunk_size].to_dict(orient='records')
                    clean_records = self._clean_records(raw_records)
                    
                    conn.execute(insert(target_table_obj).values(clean_records))
                    
                logger.success(f"Successfully replaced table '{target_table}' with {len(df)} rows.")
        except Exception as e:
            error_msg = str(e)
            if len(error_msg) > 1000:
                error_msg = error_msg[:1000] + "\n... [SQL LOG TRUNCATED TO PREVENT TERMINAL SPAM] ..."
            logger.error(f"Replace table failed: {error_msg}")
            raise RuntimeError(f"Database operation failed. Check logs.")

    def delete_and_insert(self, df: pd.DataFrame, target_table: str, delete_keys: Union[str, List[str]]):
        if df.empty: return
        keys = [delete_keys] if isinstance(delete_keys, str) else delete_keys
        dtype_mapping = self._generate_dtype_mapping(df)
        
        try:
            with self.engine.begin() as conn:
                inspector = inspect(conn)
                if not inspector.has_table(target_table, schema=self.schema):
                    df.iloc[:0].to_sql(target_table, conn, schema=self.schema, index=False, dtype=dtype_mapping)
                else:
                    for key in keys:
                        unique_vals = df[key].dropna().unique()
                        if len(unique_vals) > 0:
                            vals_str = ", ".join([f"'{v}'" if isinstance(v, str) else str(v) for v in unique_vals])
                            conn.execute(text(f'DELETE FROM "{self.schema}"."{target_table}" WHERE "{key}" IN ({vals_str})'))
                
                metadata_obj = MetaData()
                target_table_obj = Table(target_table, metadata_obj, schema=self.schema, autoload_with=conn)
                
                # SAFE CHUNK SIZE FOR POSTGRESQL (32,767 param limit)
                chunk_size = 500 
                for i in range(0, len(df), chunk_size):
                    raw_records = df.iloc[i : i + chunk_size].to_dict(orient='records')
                    clean_records = self._clean_records(raw_records)
                    
                    conn.execute(insert(target_table_obj).values(clean_records))
                    
                logger.success(f"Delete & Insert completed for {target_table}. Inserted {len(df)} rows.")
        except Exception as e:
            error_msg = str(e)
            if len(error_msg) > 1000:
                error_msg = error_msg[:1000] + "\n... [SQL LOG TRUNCATED TO PREVENT TERMINAL SPAM] ..."
            logger.error(f"Delete and Insert failed: {error_msg}")
            raise RuntimeError(f"Database operation failed. Check logs.")

    def upsert_data(self, 
                    df: pd.DataFrame, 
                    target_table: str, 
                    primary_key: Union[str, List[str]] = None, 
                    auto_evolve_schema: bool = True,
                    conflict_strategy: Literal['sum', 'last', 'skip'] = 'last'):
        if df.empty: return

        join_keys = [primary_key] if isinstance(primary_key, str) else primary_key
        if join_keys:
            join_keys = [k.lower() for k in join_keys]
            
        if not join_keys:
             logger.warning(f"No keys provided. Switch to APPEND mode.")
             df.to_sql(target_table, self.engine, schema=self.schema, if_exists='append', index=False)
             return

        for col in df.select_dtypes(include=['object', 'string']):
            if df[col].astype(str).str.match(r'^\d{4}-\d{2}-\d{2}').any():
                df[col] = pd.to_datetime(df[col], errors='ignore')

        dtype_mapping = self._generate_dtype_mapping(df)

        try:
            with self.engine.begin() as conn:
                inspector = inspect(conn)
                
                if not inspector.has_table(target_table, schema=self.schema):
                    df.head(0).to_sql(target_table, conn, schema=self.schema, index=False, dtype=dtype_mapping)
                    pk_str = ", ".join([f'"{c}"' for c in join_keys])
                    conn.execute(text(f'ALTER TABLE "{self.schema}"."{target_table}" ADD PRIMARY KEY ({pk_str})'))
                    logger.info(f"Created new table {target_table} with PK {join_keys}")

                db_cols = self._get_table_columns(target_table, conn)
                new_cols = [c for c in df.columns if c.lower() not in [dc.lower() for dc in db_cols]]
                if new_cols and auto_evolve_schema:
                    self._add_missing_columns(target_table, new_cols, dtype_mapping, conn)
                elif new_cols:
                    df = df.drop(columns=new_cols)

                table_cols = [c.lower() for c in df.columns]
                
                # SAFE CHUNK SIZE FOR POSTGRESQL (32,767 param limit)
                # LOWER CHUNK SIZE: 250 ensures we never hit the 32,767 limit even with 130 columns!
                chunk_size = 250 
                
                for i in range(0, len(df), chunk_size):
                    raw_records = df.iloc[i : i + chunk_size].to_dict(orient='records')
                    clean_records = self._clean_records(raw_records)
                    
                    metadata_obj = MetaData()
                    target_table_obj = Table(target_table, metadata_obj, schema=self.schema, autoload_with=conn)
                    
                    insert_stmt = insert(target_table_obj).values(clean_records)
                    
                    if conflict_strategy == 'skip':
                        upsert_stmt = insert_stmt.on_conflict_do_nothing(index_elements=join_keys)
                    else:
                        update_dict = {
                            col: insert_stmt.excluded[col] 
                            for col in table_cols if col not in join_keys
                        }
                        if conflict_strategy == 'sum':
                            for orig_col in df.columns:
                                lower_col = orig_col.lower()
                                if lower_col in update_dict and pd.api.types.is_numeric_dtype(df[orig_col]):
                                    update_dict[lower_col] = target_table_obj.c[lower_col] + insert_stmt.excluded[lower_col]
                                    
                        upsert_stmt = insert_stmt.on_conflict_do_update(
                            index_elements=join_keys,
                            set_=update_dict
                        )
                    conn.execute(upsert_stmt)
                logger.success(f"Upserted {len(df)} rows to {target_table} (Strategy: {conflict_strategy})")

        except Exception as e:
            # 1. Truncate the loguru output
            error_msg = str(e)
            if len(error_msg) > 800:
                error_msg = error_msg[:800] + "\n\n... [MASSIVE SQL PARAMETER LOG TRUNCATED] ..."
            
            logger.error(f"Upsert failed for {target_table}: {error_msg}")
            
            # 2. THE MAGIC FIX: 'from None' destroys the original massive error chain 
            # so the terminal doesn't spam thousands of lines.
            raise RuntimeError(f"DB Upsert Failed for {target_table}. See truncated log above.") from None

    def dispose(self):
        self.engine.dispose()
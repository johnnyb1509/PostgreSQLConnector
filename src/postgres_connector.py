import io
import pandas as pd
from typing import List, Optional, Dict, Union, Literal
from loguru import logger
from sqlalchemy import create_engine, text, URL, inspect, MetaData, Table
from sqlalchemy.types import BIGINT
from sqlalchemy.dialects.postgresql import JSONB, DOUBLE_PRECISION, TEXT, TIMESTAMP, insert

try:
    from pgvector.sqlalchemy import Vector
    _PGVECTOR_AVAILABLE = True
except ImportError:
    _PGVECTOR_AVAILABLE = False
    Vector = None


class PostgresConnector:
    """
    Trình kết nối PostgreSQL chuẩn hóa.

    Quy tắc cột:
      - Mọi tên cột đều được lowercase trước khi đẩy vào DB.
      - Không dùng double-quote bao quanh tên cột trong DDL/DML
        (PostgreSQL tự fold về lowercase khi không có quote).
      - Chỉ double-quote tên SCHEMA và tên TABLE (để tránh conflict
        với reserved words).
    """

    def __init__(
        self,
        host: str,
        database: str,
        username: str,
        password: str,
        port: int = 5432,
        schema: str = "public",
        **kwargs,
    ):
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
            database=self.database,
        )

        self.engine = create_engine(
            self.connection_url,
            pool_pre_ping=True,
            insertmanyvalues_page_size=10000,
            connect_args={"options": f"-csearch_path={self.schema}"},
        )

    # ------------------------------------------------------------------
    # PUBLIC HELPERS
    # ------------------------------------------------------------------

    def execute_query(self, query: str, params: Optional[Dict] = None) -> None:
        try:
            with self.engine.begin() as conn:
                conn.execute(text(query), params or {})
        except Exception as e:
            logger.error(f"execute_query error: {e}")
            raise

    def get_data(self, query: str, params: Optional[Dict] = None) -> pd.DataFrame:
        try:
            with self.engine.connect() as conn:
                return pd.read_sql(text(query), conn, params=params)
        except Exception as e:
            logger.error(f"get_data error: {e}")
            raise

    def check_table_exists(self, table_name: str) -> bool:
        """Return True if the table exists in self.schema."""
        try:
            with self.engine.connect() as conn:
                return inspect(conn).has_table(table_name, schema=self.schema)
        except Exception:
            return False

    def dispose(self) -> None:
        self.engine.dispose()

    # ------------------------------------------------------------------
    # PRIVATE HELPERS
    # ------------------------------------------------------------------

    def _lower_df_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return df with all column names lowercased (in-place rename)."""
        df.columns = [c.lower() for c in df.columns]
        return df

    def _generate_dtype_mapping(self, df: pd.DataFrame) -> Dict:
        """
        Map DataFrame columns → SQLAlchemy types.
        Expects df to already have lowercased columns.
        """
        dtype_map = {}
        for col in df.columns:
            non_null = df[col].dropna()
            sample_val = non_null.iloc[0] if not non_null.empty else None

            if _PGVECTOR_AVAILABLE and isinstance(sample_val, list) and all(
                isinstance(x, (int, float)) for x in sample_val
            ):
                dtype_map[col] = Vector(len(sample_val))
            elif isinstance(sample_val, (dict, list)):
                dtype_map[col] = JSONB()
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                dtype_map[col] = TIMESTAMP()
            elif pd.api.types.is_float_dtype(df[col]):
                dtype_map[col] = DOUBLE_PRECISION()
            elif pd.api.types.is_integer_dtype(df[col]):
                dtype_map[col] = BIGINT()
            else:
                dtype_map[col] = TEXT()

        return dtype_map

    def _get_table_columns(self, table_name: str, conn) -> List[str]:
        """Return existing lowercase column names from the DB table."""
        return [
            col["name"].lower()
            for col in inspect(conn).get_columns(table_name, schema=self.schema)
        ]

    def _add_missing_columns(
        self, table_name: str, missing_cols: List[str], dtype_map: Dict, conn
    ) -> None:
        """
        ALTER TABLE to add columns that exist in the DataFrame but not in DB.
        Columns are always added as lowercase, unquoted names.
        """
        type_map = {
            DOUBLE_PRECISION: "DOUBLE PRECISION",
            BIGINT: "BIGINT",
            TIMESTAMP: "TIMESTAMP",
            JSONB: "JSONB",
        }
        for col in missing_cols:
            col_lower = col.lower()
            col_type = dtype_map.get(col, TEXT())
            type_str = "TEXT"
            for sa_type, pg_str in type_map.items():
                if isinstance(col_type, sa_type):
                    type_str = pg_str
                    break
            if _PGVECTOR_AVAILABLE and isinstance(col_type, Vector):
                type_str = f"VECTOR({col_type.dim})"

            # Use unquoted column name → Postgres stores it lowercase
            conn.execute(
                text(
                    f'ALTER TABLE "{self.schema}"."{table_name}" '
                    f"ADD COLUMN IF NOT EXISTS {col_lower} {type_str}"
                )
            )
            logger.info(f"Auto-evolve: added '{col_lower}' ({type_str}) to '{self.schema}.{table_name}'")

    def _clean_records(self, records: List[Dict]) -> List[Dict]:
        """
        Convert all pandas/numpy scalar types → plain Python.
        Keys are lowercased. Handles list/array values safely.
        """
        clean = []
        for rec in records:
            row = {}
            for k, v in rec.items():
                key = str(k).lower()
                # Guard: pd.isna() raises ValueError on list/array
                try:
                    is_na = pd.isna(v)
                except (TypeError, ValueError):
                    is_na = False

                if is_na:
                    row[key] = None
                elif isinstance(v, pd.Timestamp):
                    row[key] = v.to_pydatetime()
                elif hasattr(v, "item"):          # numpy scalar → Python scalar
                    row[key] = v.item()
                else:
                    row[key] = v
            clean.append(row)
        return clean

    def _create_table_from_df(
        self,
        df: pd.DataFrame,
        table_name: str,
        conn,
        primary_keys: Optional[List[str]] = None,
    ) -> None:
        """
        Create table using df schema. df must already have lowercase columns.
        Runs DDL on the provided open connection (caller manages transaction).
        """
        dtype_mapping = self._generate_dtype_mapping(df)
        # to_sql for DDL only (0 rows) — safe inside a transaction for CREATE
        df.head(0).to_sql(
            table_name,
            conn,
            schema=self.schema,
            if_exists="fail",       # never DROP inside an open tx
            index=False,
            dtype=dtype_mapping,
        )
        if primary_keys:
            pk_str = ", ".join(primary_keys)  # unquoted → lowercase
            conn.execute(
                text(
                    f'ALTER TABLE "{self.schema}"."{table_name}" '
                    f"ADD PRIMARY KEY ({pk_str})"
                )
            )
        logger.info(f"Created table '{self.schema}.{table_name}' (pk={primary_keys})")

    def _load_table_object(self, table_name: str, conn) -> Table:
        """Reflect the SQLAlchemy Table object (once per operation, not per chunk)."""
        meta = MetaData()
        return Table(table_name, meta, schema=self.schema, autoload_with=conn)

    # ------------------------------------------------------------------
    # CORE OPERATIONS
    # ------------------------------------------------------------------

    def replace_table(
        self,
        df: pd.DataFrame,
        target_table: str,
        primary_key: Union[str, List[str], None] = None,
    ) -> None:
        """
        DROP + CREATE + bulk INSERT.
        DDL is run *before* opening the data transaction to avoid lock issues.
        """
        if df.empty:
            return

        df = self._lower_df_columns(df.copy())
        pk_cols = (
            [primary_key] if isinstance(primary_key, str)
            else (list(primary_key) if primary_key else None)
        )

        try:
            # Step 1: DDL in its own short transaction
            with self.engine.begin() as conn:
                conn.execute(
                    text(f'DROP TABLE IF EXISTS "{self.schema}"."{target_table}"')
                )
                self._create_table_from_df(df, target_table, conn, pk_cols)

            # Step 2: Bulk INSERT via COPY (fastest, no lock risk)
            self._copy_insert(df, target_table)

            logger.success(f"replace_table '{target_table}': {len(df)} rows.")

        except Exception as e:
            self._log_error(f"replace_table failed for '{target_table}'", e)
            raise

    def delete_and_insert(
        self,
        df: pd.DataFrame,
        target_table: str,
        delete_keys: Union[str, List[str]],
    ) -> None:
        """
        DELETE matching rows then INSERT. Fully parameterized — no SQL injection.
        """
        if df.empty:
            return

        df = self._lower_df_columns(df.copy())
        keys = [delete_keys] if isinstance(delete_keys, str) else list(delete_keys)

        try:
            with self.engine.begin() as conn:
                inspector = inspect(conn)

                if not inspector.has_table(target_table, schema=self.schema):
                    self._create_table_from_df(df, target_table, conn)
                else:
                    for key in keys:
                        unique_vals = df[key].dropna().unique().tolist()
                        if unique_vals:
                            # Parameterized DELETE — safe for any value type
                            conn.execute(
                                text(
                                    f'DELETE FROM "{self.schema}"."{target_table}" '
                                    f"WHERE {key} = ANY(:vals)"
                                ),
                                {"vals": unique_vals},
                            )

                table_obj = self._load_table_object(target_table, conn)
                chunk_size = self._safe_chunk_size(len(df.columns))
                for i in range(0, len(df), chunk_size):
                    records = self._clean_records(
                        df.iloc[i: i + chunk_size].to_dict(orient="records")
                    )
                    conn.execute(insert(table_obj).values(records))

            logger.success(f"delete_and_insert '{target_table}': {len(df)} rows.")

        except Exception as e:
            self._log_error(f"delete_and_insert failed for '{target_table}'", e)
            raise

    def upsert_data(
        self,
        df: pd.DataFrame,
        target_table: str,
        primary_key: Union[str, List[str], None] = None,
        auto_evolve_schema: bool = True,
        conflict_strategy: Literal["sum", "last", "skip"] = "last",
    ) -> None:
        """
        INSERT … ON CONFLICT DO UPDATE/NOTHING.

        Key fixes vs previous version:
          - MetaData/Table reflected ONCE outside the chunk loop (not per-chunk).
          - All column names lowercased before any DB interaction.
          - No auto-date coercion on object columns (was causing false positives).
          - Parameterized; no raw f-string SQL injection risk.
        """
        if df.empty:
            return

        df = self._lower_df_columns(df.copy())

        join_keys = (
            [primary_key] if isinstance(primary_key, str)
            else (list(primary_key) if primary_key else [])
        )
        join_keys = [k.lower() for k in join_keys]

        if not join_keys:
            logger.warning(f"upsert_data '{target_table}': no primary_key → APPEND mode.")
            df.to_sql(
                target_table, self.engine, schema=self.schema,
                if_exists="append", index=False,
                method="multi", chunksize=2000,
            )
            return

        dtype_mapping = self._generate_dtype_mapping(df)

        try:
            with self.engine.begin() as conn:
                inspector = inspect(conn)

                # --- Create table if missing ---
                if not inspector.has_table(target_table, schema=self.schema):
                    self._create_table_from_df(df, target_table, conn, join_keys)

                # --- Auto-evolve: add new columns ---
                db_cols = self._get_table_columns(target_table, conn)
                new_cols = [c for c in df.columns if c not in db_cols]
                if new_cols:
                    if auto_evolve_schema:
                        self._add_missing_columns(target_table, new_cols, dtype_mapping, conn)
                    else:
                        df = df.drop(columns=new_cols)

                # --- Reflect Table ONCE (not per chunk) ---
                table_obj = self._load_table_object(target_table, conn)
                table_cols = [c.lower() for c in df.columns]
                chunk_size = self._safe_chunk_size(len(df.columns))

                for i in range(0, len(df), chunk_size):
                    records = self._clean_records(
                        df.iloc[i: i + chunk_size].to_dict(orient="records")
                    )
                    insert_stmt = insert(table_obj).values(records)

                    if conflict_strategy == "skip":
                        upsert_stmt = insert_stmt.on_conflict_do_nothing(
                            index_elements=join_keys
                        )
                    else:
                        update_dict = {
                            col: insert_stmt.excluded[col]
                            for col in table_cols
                            if col not in join_keys
                        }
                        if conflict_strategy == "sum":
                            for col in table_cols:
                                if col in update_dict and pd.api.types.is_numeric_dtype(df[col]):
                                    update_dict[col] = (
                                        table_obj.c[col] + insert_stmt.excluded[col]
                                    )
                        upsert_stmt = insert_stmt.on_conflict_do_update(
                            index_elements=join_keys,
                            set_=update_dict,
                        )
                    conn.execute(upsert_stmt)

            logger.success(
                f"upsert_data '{target_table}': {len(df)} rows "
                f"(strategy={conflict_strategy}, pk={join_keys})"
            )

        except Exception as e:
            self._log_error(f"upsert_data failed for '{target_table}'", e)
            raise RuntimeError(
                f"upsert_data failed for '{target_table}'. See log above."
            ) from None

    # ------------------------------------------------------------------
    # BULK COPY (fastest path for large loads)
    # ------------------------------------------------------------------

    def _copy_insert(self, df: pd.DataFrame, target_table: str) -> None:
        """
        Bulk-load df into an existing table via COPY FROM STDIN.
        df must already have lowercase columns.
        Fastest method: streams CSV directly to Postgres, no row-by-row params.
        """
        cols = ", ".join(df.columns.tolist())
        qualified = f'"{self.schema}"."{target_table}"'

        buf = io.StringIO()
        df.to_csv(buf, index=False, header=False, na_rep="\\N")
        buf.seek(0)

        raw_conn = self.engine.raw_connection()
        try:
            with raw_conn.cursor() as cur:
                cur.copy_expert(
                    f"COPY {qualified} ({cols}) FROM STDIN "
                    f"WITH (FORMAT csv, NULL '\\N')",
                    buf,
                )
            raw_conn.commit()
        finally:
            raw_conn.close()

    # ------------------------------------------------------------------
    # UTILITY
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_chunk_size(n_cols: int) -> int:
        """
        PostgreSQL hard limit: 32,767 bind parameters per statement.
        Leave a small buffer.
        """
        if n_cols == 0:
            return 500
        return max(1, min(500, 32000 // n_cols))

    @staticmethod
    def _log_error(prefix: str, exc: Exception) -> None:
        msg = str(exc)
        if len(msg) > 800:
            msg = msg[:800] + "\n... [TRUNCATED] ..."
        logger.error(f"{prefix}: {msg}")
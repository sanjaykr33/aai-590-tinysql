import re

PROJECT_ID = os.environ.get("PROJECT_ID")
DATASET_ID = os.environ.get("DATASET_ID")
GCS_BUCKET = os.environ.get("GCS_BUCKET")


try:
    import sqlglot
    from sqlglot import exp
    SQLGLOT_AVAILABLE = True
except ImportError:
    SQLGLOT_AVAILABLE = False

def extract_and_fix_ddl(context_raw):
    """Transpiles MySQL DDL to BigQuery and strips constraints."""
    if not SQLGLOT_AVAILABLE:
        return "" # Skip or keep if not available

    try:
        statements = sqlglot.parse(context_raw, read="mysql")
    except Exception:
        return ""

    ddl_statements = []
    for expression in statements:
        if isinstance(expression, exp.Create) and expression.args.get("kind") == "TABLE":
            expression.set("exists", True)

            if isinstance(expression.this, exp.Schema):
                table_ident = expression.this.this
            else:
                table_ident = expression.this
            if isinstance(table_ident, exp.Table):
                table_ident.set("db", None)
                table_ident.set("catalog", None)

            schema = expression.this
            if isinstance(schema, exp.Schema):
                for column_def in schema.expressions:
                    if isinstance(column_def, exp.ColumnDef):
                        column_def.set("constraints", [
                            c for c in column_def.args.get("constraints", [])
                            if not (isinstance(c.kind, (exp.PrimaryKeyColumnConstraint, exp.Reference)) or "FOREIGN" in str(c.kind).upper())
                        ])
                schema.set("expressions", [
                    e for e in schema.expressions
                    if not (isinstance(e, exp.Constraint) and any(isinstance(k, (exp.PrimaryKey, exp.ForeignKey, exp.Reference)) for k in e.args.values()))
                ])
            ddl_statements.append(expression.sql(dialect="bigquery"))
    return ";\n".join(ddl_statements)

def transpile_to_bigquery(sql_query):
    """Fallback to transpile query from MySQL to BigQuery (handling syntax bias)."""
    if not SQLGLOT_AVAILABLE:
        return sql_query

    try:
        transpiled_list = sqlglot.transpile(sql_query, read="mysql", write="bigquery")
        if isinstance(transpiled_list, list) and transpiled_list and transpiled_list[0] != sql_query:
            return transpiled_list[0]
    except Exception:
        pass
    return sql_query


def prefix_table_names(sql_query, prefix_string):
    """Parses query and prefixes all real table names with prefix_string (e.g. {DATASET_ID})."""
    if not SQLGLOT_AVAILABLE:
        return sql_query
    try:
        expression = sqlglot.parse_one(sql_query, read="mysql")
        cte_names = {cte.alias for cte in expression.find_all(exp.CTE)} if hasattr(expression, "find_all") else set()

        for table in expression.find_all(exp.Table):
            if table.name in cte_names:
                continue # Skip CTEs
            if not table.args.get("db"):
                table.set("db", prefix_string)
        return expression.sql(dialect="bigquery")
    except Exception:
        return sql_query


def validate_sql_dry_run(sql_query):
    """Validates the generated SQL via BigQuery Dry Run."""
    if not sql_query or not any(
        k in sql_query.upper() for k in ["SELECT", "DELETE", "UPDATE", "INSERT", "WITH"]
    ):
        return False, "Malformed or Empty SQL"
    # Import bigquery here to keep it contained, or add it at the top of the file
    from google.cloud import bigquery
    # Initialize client and variables that were missing
    client = bigquery.Client()
    FULL_DATASET_PATH = f"{PROJECT_ID}.{DATASET_ID}"
    job_config = bigquery.QueryJobConfig(
        dry_run=True, default_dataset=FULL_DATASET_PATH
    )
    try:
        client.query(sql_query, job_config=job_config)
        return True, "Success"
    except Exception as e:
        error_msg = str(e)
        # Strip BigQuery URL
        error_msg = re.sub(
            r"400 POST https://bigquery\.googleapis\.com/bigquery/v2/projects/[^/]+/jobs\?prettyPrint=false:?",
            "",
            error_msg,
            flags=re.IGNORECASE,
        ).strip()
        # Strip Job ID
        error_msg = re.sub(
            r"Job ID:\s*[a-f0-9\-]+", "", error_msg, flags=re.IGNORECASE
        )
        # Strip Location
        error_msg = re.sub(
            r"Location:\s*[a-z0-9\-]+", "", error_msg, flags=re.IGNORECASE
        )
        # Clean double spaces and leading characters
        error_msg = re.sub(r"\s+", " ", error_msg).strip()
        error_msg = re.sub(r"^:\s*", "", error_msg)
        return False, error_msg

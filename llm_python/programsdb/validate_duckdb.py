#!/usr/bin/env python3
"""
DuckDB Validation Script
Validates DuckDB databases and checks their schemas and data integrity.
"""

import duckdb
import os
import sys
import argparse
from collections import defaultdict
from pathlib import Path

def get_db_schema(db_path):
    """Get the schema information for a DuckDB database"""
    try:
        conn = duckdb.connect(db_path, read_only=True)
        
        # Get all tables
        tables = conn.execute("SHOW TABLES").fetchall()
        
        schema_info = {}
        for table in tables:
            table_name = table[0]
            columns = conn.execute(f"DESCRIBE {table_name}").fetchall()
            schema_info[table_name] = [(col[0], col[1]) for col in columns]  # (name, type)
            
        conn.close()
        return schema_info
    except Exception as e:
        return f"Error: {e}"

def get_db_row_counts(db_path):
    """Get row counts for each table in the database"""
    try:
        conn = duckdb.connect(db_path, read_only=True)
        
        tables = conn.execute("SHOW TABLES").fetchall()
        
        counts = {}
        for table in tables:
            table_name = table[0]
            count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
            counts[table_name] = count
            
        conn.close()
        return counts
    except Exception as e:
        return f"Error: {e}"

def validate_single_db(db_path, verbose=False):
    """Validate a single database file"""
    print(f"\n📁 {db_path}")
    print("-" * 40)
    
    schema = get_db_schema(db_path)
    counts = get_db_row_counts(db_path)
    
    if isinstance(schema, str):
        print(f"❌ {schema}")
        return None, None
        
    if isinstance(counts, str):
        print(f"❌ Row count error: {counts}")
        return None, None
        
    for table_name, columns in schema.items():
        count = counts.get(table_name, 0)
        print(f"  📋 Table: {table_name} ({count:,} rows)")
        if verbose:
            for col_name, col_type in columns[:10]:  # Show first 10 columns
                print(f"    • {col_name}: {col_type}")
            if len(columns) > 10:
                print(f"    ... and {len(columns) - 10} more columns")
    
    return schema, counts

def validate_databases(db_paths, verbose=False):
    """Validate multiple database files"""
    print("DuckDB Database Validation Report")
    print("=" * 50)
    
    schemas = {}
    row_counts = {}
    valid_dbs = []
    
    for db_path in db_paths:
        schema, counts = validate_single_db(db_path, verbose)
        if schema:
            valid_dbs.append(db_path)
            schemas[db_path] = schema
            row_counts[db_path] = counts
    
    # Analyze schema compatibility
    if len(valid_dbs) > 1:
        print("\n\nSchema Compatibility Analysis")
        print("=" * 50)
        
        table_schemas = defaultdict(list)
        for db_file, schema in schemas.items():
            if isinstance(schema, dict):
                for table_name, columns in schema.items():
                    table_schemas[table_name].append((db_file, columns))
        
        for table_name, db_schemas in table_schemas.items():
            print(f"\n🔍 Table: {table_name}")
            print(f"   Found in {len(db_schemas)} database(s)")
            
            if len(db_schemas) > 1:
                # Check if schemas are compatible
                first_schema = db_schemas[0][1]
                compatible = True
                for db_file, schema in db_schemas[1:]:
                    if schema != first_schema:
                        compatible = False
                        break
                
                if compatible:
                    print("   ✅ All schemas are identical")
                    total_rows = sum(row_counts.get(db, {}).get(table_name, 0) for db, _ in db_schemas)
                    print(f"   📊 Total rows across all databases: {total_rows:,}")
                else:
                    print("   ⚠️  Schema differences detected:")
                    for db_file, schema in db_schemas[:3]:  # Show first 3 for brevity
                        print(f"     {os.path.basename(db_file)}: {len(schema)} columns")
                        if verbose:
                            print(f"       Columns: {[col[0] for col in schema[:5]]}")
    
    # Summary
    print(f"\n\n📊 Summary:")
    print(f"Total databases found: {len(db_paths)}")
    print(f"Valid DuckDB databases: {len(valid_dbs)}")
    print(f"Unique table names: {len(table_schemas) if 'table_schemas' in locals() else 0}")
    
    # Calculate total rows
    total_rows = 0
    for db_file in valid_dbs:
        for table, count in row_counts.get(db_file, {}).items():
            total_rows += count
    print(f"Total rows across all valid databases: {total_rows:,}")
    
    return valid_dbs, schemas, row_counts

def main():
    parser = argparse.ArgumentParser(description='Validate DuckDB database files')
    parser.add_argument('databases', nargs='*', help='Database files to validate (if none provided, validates all .db files in current directory)')
    parser.add_argument('-v', '--verbose', action='store_true', help='Show detailed schema information')
    parser.add_argument('-d', '--directory', default='.', help='Directory to search for .db files (default: current directory)')
    
    args = parser.parse_args()
    
    # Get database files
    if args.databases:
        db_files = args.databases
    else:
        # Find all .db files in the specified directory
        db_files = []
        for root, dirs, files in os.walk(args.directory):
            for file in files:
                if file.endswith('.db'):
                    db_files.append(os.path.join(root, file))
        
        if not db_files:
            print(f"No .db files found in {args.directory}")
            sys.exit(1)
    
    # Validate databases
    validate_databases(db_files, args.verbose)

if __name__ == "__main__":
    main()
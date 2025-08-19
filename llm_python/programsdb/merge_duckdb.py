#!/usr/bin/env python3
"""
DuckDB Merge Script
Merges multiple DuckDB databases into a single consolidated database.
"""

import duckdb
import os
import sys
import argparse
from pathlib import Path

def get_db_schema(db_path):
    """Get the schema information for a DuckDB database"""
    try:
        conn = duckdb.connect(db_path, read_only=True)
        tables = conn.execute("SHOW TABLES").fetchall()
        
        schema_info = {}
        for table in tables:
            table_name = table[0]
            columns = conn.execute(f"DESCRIBE {table_name}").fetchall()
            schema_info[table_name] = [(col[0], col[1]) for col in columns]
        
        conn.close()
        return schema_info
    except Exception as e:
        return None

def merge_databases(source_dbs, target_db, table_name='programs', key_column='key', verbose=False):
    """
    Merge multiple DuckDB databases into a single database.
    
    Args:
        source_dbs: List of source database paths
        target_db: Target database path
        table_name: Name of the table to merge (default: 'programs')
        key_column: Column to use for deduplication (default: 'key')
        verbose: Print detailed progress information
    """
    print(f"🎯 Creating consolidated database at: {target_db}")
    print("=" * 60)
    
    # Filter out non-existent files
    valid_dbs = [db for db in source_dbs if os.path.exists(db)]
    if len(valid_dbs) < len(source_dbs):
        print(f"⚠️  {len(source_dbs) - len(valid_dbs)} database files not found")
    
    if not valid_dbs:
        print("❌ No valid database files to merge")
        return False
    
    # Check if target database exists and warn user
    if os.path.exists(target_db):
        try:
            conn = duckdb.connect(target_db, read_only=True)
            existing_count = 0
            existing_tables = []
            try:
                tables = conn.execute("SHOW TABLES").fetchall()
                for table in tables:
                    table_name_check = table[0]
                    count = conn.execute(f"SELECT COUNT(*) FROM {table_name_check}").fetchone()[0]
                    existing_count += count
                    existing_tables.append(f"{table_name_check} ({count} rows)")
            except:
                pass
            conn.close()
            
            if existing_count > 0:
                print(f"⚠️  WARNING: Target database '{target_db}' already exists!")
                print(f"   It contains {existing_count} total rows across tables: {', '.join(existing_tables)}")
                print(f"   This operation will DELETE all existing data in '{target_db}'")
                response = input("   Do you want to continue? (yes/no): ").strip().lower()
                if response not in ['yes', 'y']:
                    print("❌ Merge cancelled by user")
                    return False
        except:
            # If we can't read it, still warn but less detail
            print(f"⚠️  WARNING: Target database '{target_db}' already exists!")
            print(f"   This operation will DELETE it and create a new database")
            response = input("   Do you want to continue? (yes/no): ").strip().lower()
            if response not in ['yes', 'y']:
                print("❌ Merge cancelled by user")
                return False
        
        os.remove(target_db)
        print("✅ Removed existing target database")
    
    target_conn = duckdb.connect(target_db)
    
    # Analyze schemas to find union of all columns
    print(f"\n📋 Analyzing schemas for table '{table_name}'...")
    all_columns = {}
    metadata_schemas = {}
    valid_source_dbs = []
    
    for db_file in valid_dbs:
        schema = get_db_schema(db_file)
        if schema and table_name in schema:
            valid_source_dbs.append(db_file)
            for col_name, col_type in schema[table_name]:
                if col_name not in all_columns:
                    all_columns[col_name] = col_type
            # Also check for metadata table
            if 'metadata' in schema:
                metadata_schemas[db_file] = schema['metadata']
    
    if not valid_source_dbs:
        print(f"❌ No databases contain table '{table_name}'")
        target_conn.close()
        return False
    
    print(f"✅ Found {len(all_columns)} unique columns across {len(valid_source_dbs)} databases")
    if verbose:
        for col_name, col_type in all_columns.items():
            print(f"  • {col_name}: {col_type}")
    
    # Create metadata table if any source has it
    if metadata_schemas:
        print("\n🔨 Creating metadata table...")
        target_conn.execute("""
            CREATE TABLE IF NOT EXISTS metadata (
                key VARCHAR,
                value VARCHAR
            )
        """)
        target_conn.execute("INSERT INTO metadata VALUES ('consolidated', 'true')")
    
    # Create the target table with all columns
    print(f"\n🔨 Creating {table_name} table...")
    columns_def = [f"{col_name} {col_type}" for col_name, col_type in all_columns.items()]
    create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            {', '.join(columns_def)}
        )
    """
    target_conn.execute(create_table_sql)
    
    # Create unique index on key column if it exists (for ON CONFLICT support)
    if 'key' in all_columns:
        target_conn.execute(f"CREATE UNIQUE INDEX IF NOT EXISTS idx_{table_name}_key ON {table_name}(key)")
        print(f"✅ Table created with UNIQUE constraint on 'key' column")
    else:
        print("✅ Table created")
    
    # Merge data from all databases
    print(f"\n📥 Merging data from {len(valid_source_dbs)} databases...")
    total_added = 0
    total_skipped = 0
    
    for i, db_file in enumerate(valid_source_dbs, 1):
        if verbose:
            print(f"\n  [{i}/{len(valid_source_dbs)}] Processing: {db_file}")
        else:
            print(f"  [{i}/{len(valid_source_dbs)}] {os.path.basename(db_file)}...", end='')
        
        try:
            # Attach the source database
            target_conn.execute(f"ATTACH '{db_file}' AS source_db (READ_ONLY)")
            
            # Get the schema of the source table
            source_schema = target_conn.execute(f"DESCRIBE source_db.{table_name}").fetchall()
            source_columns = [col[0] for col in source_schema]
            
            # Build the SELECT statement with NULL for missing columns
            select_parts = []
            for col_name in all_columns.keys():
                if col_name in source_columns:
                    select_parts.append(f"p.{col_name}")
                else:
                    # Use appropriate NULL based on column type
                    col_type = all_columns[col_name]
                    if 'BOOLEAN[]' in col_type:
                        select_parts.append("NULL::BOOLEAN[]")
                    elif 'INTEGER[][]' in col_type:
                        select_parts.append("NULL::INTEGER[][]")
                    elif 'INTEGER[][][]' in col_type:
                        select_parts.append("NULL::INTEGER[][][]")
                    else:
                        select_parts.append("NULL")
            
            # Count rows before insertion
            count_before = target_conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
            
            # Insert data, handling duplicates by key
            if key_column and key_column in source_columns:
                insert_sql = f"""
                    INSERT INTO {table_name} 
                    SELECT {', '.join(select_parts)}
                    FROM source_db.{table_name} p
                    WHERE NOT EXISTS (
                        SELECT 1 FROM {table_name} 
                        WHERE {table_name}.{key_column} = p.{key_column}
                    )
                """
            else:
                # No deduplication if key column not specified or not present
                insert_sql = f"""
                    INSERT INTO {table_name} 
                    SELECT {', '.join(select_parts)}
                    FROM source_db.{table_name} p
                """
            
            target_conn.execute(insert_sql)
            
            # Count rows after insertion
            count_after = target_conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
            rows_added = count_after - count_before
            
            # Get total rows in source for reporting
            source_count = target_conn.execute(f"SELECT COUNT(*) FROM source_db.{table_name}").fetchone()[0]
            rows_skipped = source_count - rows_added
            
            total_added += rows_added
            total_skipped += rows_skipped
            
            if verbose:
                print(f"    ✅ Source rows: {source_count}, Added: {rows_added}, Skipped (duplicates): {rows_skipped}")
            else:
                print(f" ✅ {rows_added} added, {rows_skipped} skipped")
            
            # Detach the source database
            target_conn.execute("DETACH source_db")
            
        except Exception as e:
            print(f"\n    ❌ Error: {e}")
            try:
                target_conn.execute("DETACH source_db")
            except:
                pass
    
    # Get final statistics
    final_count = target_conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
    
    # Additional statistics if specific columns exist
    stats = []
    if key_column in all_columns:
        unique_keys = target_conn.execute(f"SELECT COUNT(DISTINCT {key_column}) FROM {table_name} WHERE {key_column} IS NOT NULL").fetchone()[0]
        stats.append(f"Unique {key_column}s: {unique_keys:,}")
    
    if 'task_id' in all_columns:
        unique_tasks = target_conn.execute(f"SELECT COUNT(DISTINCT task_id) FROM {table_name} WHERE task_id IS NOT NULL").fetchone()[0]
        stats.append(f"Unique task IDs: {unique_tasks:,}")
    
    if 'model' in all_columns:
        unique_models = target_conn.execute(f"SELECT COUNT(DISTINCT model) FROM {table_name} WHERE model IS NOT NULL").fetchone()[0]
        stats.append(f"Unique models: {unique_models:,}")
    
    print("\n" + "=" * 60)
    print("✅ Consolidation complete!")
    print(f"📊 Final statistics:")
    print(f"  • Total rows in {table_name}: {final_count:,}")
    print(f"  • Rows added: {total_added:,}")
    print(f"  • Rows skipped (duplicates): {total_skipped:,}")
    for stat in stats:
        print(f"  • {stat}")
    print(f"  • Database location: {target_db}")
    
    # Close the connection
    target_conn.close()
    return True

def main():
    parser = argparse.ArgumentParser(description='Merge multiple DuckDB databases')
    parser.add_argument('target', help='Target database path (will be created/overwritten)')
    parser.add_argument('sources', nargs='*', help='Source database files to merge (if none provided, merges all .db files in current directory)')
    parser.add_argument('-t', '--table', default='programs', help='Table name to merge (default: programs)')
    parser.add_argument('-k', '--key', default='key', help='Column to use for deduplication (default: key)')
    parser.add_argument('-d', '--directory', default='.', help='Directory to search for .db files (default: current directory)')
    parser.add_argument('-v', '--verbose', action='store_true', help='Show detailed progress information')
    parser.add_argument('--no-dedup', action='store_true', help='Disable deduplication (include all rows)')
    
    args = parser.parse_args()
    
    # Get source database files
    if args.sources:
        source_dbs = args.sources
    else:
        # Find all .db files in the specified directory (excluding target)
        source_dbs = []
        target_name = os.path.basename(args.target)
        for root, dirs, files in os.walk(args.directory):
            for file in files:
                if file.endswith('.db') and file != target_name:
                    source_dbs.append(os.path.join(root, file))
        
        if not source_dbs:
            print(f"No .db files found in {args.directory}")
            sys.exit(1)
    
    # Set key to None if deduplication is disabled
    key_column = None if args.no_dedup else args.key
    
    # Merge databases
    success = merge_databases(source_dbs, args.target, args.table, key_column, args.verbose)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
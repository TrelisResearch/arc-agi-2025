#!/usr/bin/env python3
"""
Migration script to update database schema for transductive field naming.

Changes:
- Renames 'is_test_transductive' column to 'is_transductive' if it exists
- Adds 'is_transductive' column with default False if neither exists
- Handles both cases for backward compatibility

Usage:
    python migrate_transductive_field.py [database_path]
    python migrate_transductive_field.py --all  # Migrate all .db files in programsdb/
"""

import argparse
import duckdb
import os
from pathlib import Path
from typing import List


def get_column_names(connection: duckdb.DuckDBPyConnection, table_name: str = "programs") -> List[str]:
    """Get list of column names for a table."""
    try:
        result = connection.execute(f"PRAGMA table_info({table_name})").fetchall()
        return [row[1] for row in result]  # Column name is at index 1
    except Exception:
        return []


def migrate_database(db_path: str) -> bool:
    """
    Migrate a single database file to use 'is_transductive' field.
    
    Returns:
        bool: True if migration was performed, False if no changes needed
    """
    print(f"\n🔍 Analyzing database: {db_path}")
    
    if not os.path.exists(db_path):
        print(f"❌ Database file not found: {db_path}")
        return False
    
    try:
        connection = duckdb.connect(db_path)
        
        # Check if programs table exists
        tables = connection.execute("SHOW TABLES").fetchall()
        table_names = [row[0] for row in tables]
        
        if "programs" not in table_names:
            print("⚠️  No 'programs' table found - skipping")
            connection.close()
            return False
        
        # Get current column names
        columns = get_column_names(connection, "programs")
        print(f"📋 Current columns: {columns}")
        
        has_is_transductive = "is_transductive" in columns
        has_is_test_transductive = "is_test_transductive" in columns
        
        if has_is_transductive and not has_is_test_transductive:
            print("✅ Database already has 'is_transductive' field - no migration needed")
            connection.close()
            return False
        
        elif has_is_test_transductive and not has_is_transductive:
            print("🔄 Migrating 'is_test_transductive' to 'is_transductive'")
            
            try:
                # Try simple rename first
                connection.execute("ALTER TABLE programs RENAME COLUMN is_test_transductive TO is_transductive")
                print("✅ Column renamed successfully")
                connection.close()
                return True
            except Exception as e:
                print(f"⚠️  Simple rename failed ({e}), using table recreation method...")
                
                # Table recreation method for databases with constraints
                # Create new table with correct schema
                connection.execute("""
                    CREATE TABLE programs_new (
                        key VARCHAR PRIMARY KEY,
                        task_id VARCHAR NOT NULL,
                        reasoning TEXT,
                        code TEXT NOT NULL,
                        correct_train_input BOOLEAN[] NOT NULL,
                        correct_test_input BOOLEAN[] NOT NULL,
                        predicted_train_output INTEGER[][][] NOT NULL,
                        predicted_test_output INTEGER[][][] NOT NULL,
                        model VARCHAR NOT NULL,
                        is_transductive BOOLEAN NOT NULL DEFAULT FALSE
                    )
                """)
                
                # Copy data, mapping is_test_transductive to is_transductive
                connection.execute("""
                    INSERT INTO programs_new 
                    SELECT key, task_id, reasoning, code, correct_train_input, correct_test_input,
                           predicted_train_output, predicted_test_output, model, 
                           COALESCE(is_test_transductive, FALSE) as is_transductive
                    FROM programs
                """)
                
                # Drop old table and rename new one
                connection.execute("DROP TABLE programs")
                connection.execute("ALTER TABLE programs_new RENAME TO programs")
                
                print("✅ Table recreated with correct schema")
                connection.close()
                return True
        
        elif has_is_test_transductive and has_is_transductive:
            print("⚠️  Both columns exist - removing 'is_test_transductive' and keeping 'is_transductive'")
            
            # Update is_transductive with values from is_test_transductive where is_transductive is NULL
            connection.execute("""
                UPDATE programs 
                SET is_transductive = COALESCE(is_transductive, is_test_transductive, false)
            """)
            
            # Drop the old column
            connection.execute("ALTER TABLE programs DROP COLUMN is_test_transductive")
            print("✅ Duplicate column removed, data preserved")
            connection.close()
            return True
        
        else:
            print("➕ Adding 'is_transductive' column with default FALSE")
            
            # Add the new column with default value
            connection.execute("ALTER TABLE programs ADD COLUMN is_transductive BOOLEAN NOT NULL DEFAULT FALSE")
            print("✅ Column added successfully")
            connection.close()
            return True
    
    except Exception as e:
        print(f"❌ Migration failed for {db_path}: {e}")
        return False


def find_database_files(directory: str) -> List[str]:
    """Find all .db files in the programsdb directory."""
    db_dir = Path(directory)
    db_files = []
    
    # Main directory
    for db_file in db_dir.glob("*.db"):
        if db_file.is_file():
            db_files.append(str(db_file))
    
    # Backup directory
    backup_dir = db_dir / "backup"
    if backup_dir.exists():
        for db_file in backup_dir.glob("*.db"):
            if db_file.is_file():
                db_files.append(str(db_file))
        for backup_file in backup_dir.glob("*.backup"):
            if backup_file.is_file():
                db_files.append(str(backup_file))
    
    return sorted(db_files)


def main():
    parser = argparse.ArgumentParser(description="Migrate database transductive field naming")
    parser.add_argument("database_path", nargs="?", help="Path to database file to migrate")
    parser.add_argument("--all", action="store_true", help="Migrate all .db files in programsdb directory")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be migrated without making changes")
    
    args = parser.parse_args()
    
    if args.all:
        # Find programsdb directory
        script_dir = Path(__file__).parent
        db_files = find_database_files(str(script_dir))
        
        if not db_files:
            print("❌ No database files found in programsdb directory")
            return
        
        print(f"🎯 Found {len(db_files)} database files to migrate:")
        for db_file in db_files:
            print(f"  • {db_file}")
        
        if args.dry_run:
            print("\n🔍 DRY RUN - No changes will be made")
        else:
            print(f"\n🚀 Starting migration of {len(db_files)} databases...")
        
        migrated_count = 0
        for db_file in db_files:
            if args.dry_run:
                print(f"\n[DRY RUN] Would migrate: {db_file}")
            else:
                if migrate_database(db_file):
                    migrated_count += 1
        
        if args.dry_run:
            print(f"\n🔍 DRY RUN COMPLETE - Would migrate {len(db_files)} databases")
        else:
            print(f"\n🎉 Migration complete! {migrated_count} databases migrated, {len(db_files) - migrated_count} already up-to-date")
    
    elif args.database_path:
        if args.dry_run:
            print(f"🔍 DRY RUN - Would migrate: {args.database_path}")
        else:
            success = migrate_database(args.database_path)
            if success:
                print(f"\n🎉 Migration successful for {args.database_path}")
            else:
                print(f"\n✅ No migration needed for {args.database_path}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
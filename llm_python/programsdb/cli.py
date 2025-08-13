"""
CLI commands for the program database.
"""

import argparse

from .sync import sync_database_to_cloud, import_from_parquet


def sync_command(args):
    """Handle the sync command."""
    try:
        gcs_path = sync_database_to_cloud(args.db_path)
        if gcs_path:
            print(f"Sync completed successfully to {gcs_path}")
        else:
            print("No data to sync")
    except Exception as e:
        print(f"Error during sync: {e}")
        return 1
    return 0


def import_command(args):
    """Handle the import command."""
    try:
        count = import_from_parquet(args.parquet_path, args.db_path)
        print(f"Import completed successfully - imported {count} programs")
    except Exception as e:
        print(f"Error during import: {e}")
        return 1
    return 0


def clear_command(args):
    """Handle the clear command."""
    from pathlib import Path
    import os
    
    # Determine the database path
    if args.db_path:
        db_path = Path(args.db_path)
    else:
        # Use default location
        current_dir = Path(__file__).parent
        db_path = current_dir / "local.db"
    
    # Check if the database file exists
    if not db_path.exists():
        print(f"Database file does not exist: {db_path}")
        return 0
    
    # Get confirmation from user
    print(f"This will permanently delete the database file: {db_path}")
    print("All stored program data will be lost.")
    confirmation = input("Are you sure you want to continue? (yes/no): ").strip().lower()
    
    if confirmation in ['yes', 'y']:
        try:
            os.remove(db_path)
            print(f"Database file successfully deleted: {db_path}")
            return 0
        except Exception as e:
            print(f"Error deleting database file: {e}")
            return 1
    else:
        print("Operation cancelled.")
        return 0


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Program database CLI")
    parser.add_argument(
        "--db-path", 
        type=str, 
        help="Path to the local database file (optional)"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Sync command
    subparsers.add_parser("sync", help="Sync local database to Google Cloud Storage")
    
    # Import command
    import_parser = subparsers.add_parser("import", help="Import programs from parquet file")
    import_parser.add_argument(
        "parquet_path",
        type=str,
        help="Path to parquet file (local path or gs:// URL)"
    )
    
    # Clear command
    subparsers.add_parser("clear", help="Clear the local database (delete the database file)")
    
    args = parser.parse_args()
    
    if args.command == "sync":
        return sync_command(args)
    elif args.command == "import":
        return import_command(args)
    elif args.command == "clear":
        return clear_command(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    exit(main())

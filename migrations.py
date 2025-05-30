"""
Run this script to initialize database migrations:
python migrations.py
"""
import os
import sys
from flask_migrate import init, migrate, upgrade
from app import app, db

if __name__ == '__main__':
    with app.app_context():
        if len(sys.argv) > 1:
            command = sys.argv[1]
            if command == 'init':
                if not os.path.exists('migrations'):
                    init()
                    print("Initialized migrations")
                else:
                    print("Migrations already initialized")
            elif command == 'migrate':
                migrate(message='Auto migration')
                print("Created new migration")
            elif command == 'upgrade':
                upgrade()
                print("Applied migrations")
        else:
            print("Usage: python migrations.py [init|migrate|upgrade]")

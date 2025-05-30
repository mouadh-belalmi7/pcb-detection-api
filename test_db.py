<<<<<<< HEAD
import psycopg2
from sqlalchemy import create_engine, text

# Test direct connection
try:
    conn = psycopg2.connect(
        host="52.178.110.198",
        database="guest",
        user="guest",
        password="Gue5t_P0stgr3s!_2025",
        port="5432"
    )
    print("✅ Direct PostgreSQL connection successful!")
    
    # Test a simple query
    cursor = conn.cursor()
    cursor.execute("SELECT version();")
    record = cursor.fetchone()
    print(f"✅ PostgreSQL version: {record[0].split(',')[0]}")
    
    cursor.close()
    conn.close()
except Exception as e:
    print(f"❌ Direct connection failed: {e}")

# Test SQLAlchemy connection
try:
    engine = create_engine(
        "postgresql://guest:Gue5t_P0stgr3s!_2025@52.178.110.198:5432/guest"
    )
    with engine.connect() as conn:
        # Fixed: use text() for raw SQL
        result = conn.execute(text("SELECT 1"))
        print("✅ SQLAlchemy connection successful!")
        
        # Test creating a table
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS test_connection (
                id SERIAL PRIMARY KEY,
                test_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """))
        print("✅ Can create tables successfully!")
        
        # Clean up test table
        conn.execute(text("DROP TABLE IF EXISTS test_connection"))
        conn.commit()
        
except Exception as e:
    print(f"❌ SQLAlchemy connection failed: {e}")

=======
import psycopg2
from sqlalchemy import create_engine, text

# Test direct connection
try:
    conn = psycopg2.connect(
        host="52.178.110.198",
        database="guest",
        user="guest",
        password="Gue5t_P0stgr3s!_2025",
        port="5432"
    )
    print("✅ Direct PostgreSQL connection successful!")
    
    # Test a simple query
    cursor = conn.cursor()
    cursor.execute("SELECT version();")
    record = cursor.fetchone()
    print(f"✅ PostgreSQL version: {record[0].split(',')[0]}")
    
    cursor.close()
    conn.close()
except Exception as e:
    print(f"❌ Direct connection failed: {e}")

# Test SQLAlchemy connection
try:
    engine = create_engine(
        "postgresql://guest:Gue5t_P0stgr3s!_2025@52.178.110.198:5432/guest"
    )
    with engine.connect() as conn:
        # Fixed: use text() for raw SQL
        result = conn.execute(text("SELECT 1"))
        print("✅ SQLAlchemy connection successful!")
        
        # Test creating a table
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS test_connection (
                id SERIAL PRIMARY KEY,
                test_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """))
        print("✅ Can create tables successfully!")
        
        # Clean up test table
        conn.execute(text("DROP TABLE IF EXISTS test_connection"))
        conn.commit()
        
except Exception as e:
    print(f"❌ SQLAlchemy connection failed: {e}")

>>>>>>> 518cca968ba8bfdd0ffde9866e726a735025351d
print("\n🎉 Database is ready to use with your PCB Detection API!")
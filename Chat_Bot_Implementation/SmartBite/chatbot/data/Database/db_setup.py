import sqlite3

def create_database(db_name="SmartBite/chatbot/data/Database/recipes.db"):
    """
    Create a SQLite database and set up the schema.

    This function connects to the specified SQLite database file (creating it if it doesn't exist),
    reads the schema definition from a separate SQL file, and initializes the database with the
    required tables and relationships.

    Args:
        db_name (str): Path to the SQLite database file. Defaults to 
                       "SmartBite/chatbot/data/Database/recipes.db".

    Raises:
        Exception: If an error occurs during database creation or schema execution.
    """

    try:
        # Connect to SQLite (creates the file if it doesn't exist)
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()

        # Read and execute the schema file
        with open("SmartBite/chatbot/data/Database/schema.sql", "r") as schema_file:
            schema = schema_file.read()
            cursor.executescript(schema)

        # Commit changes and close connection
        conn.commit()
        conn.close()
        print(f"Database '{db_name}' has been set up successfully.")
    except Exception as e:
        print(f"Error setting up the database: {e}")


if __name__ == "__main__":
    create_database()


    
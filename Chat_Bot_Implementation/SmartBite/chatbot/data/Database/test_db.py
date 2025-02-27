import sqlite3

def test_get_all_rows_database(db_name="recipes.db"):
    """
    Test the contents of the database by querying a limited number of rows from each table.

    This function verifies that the database tables contain data and checks their structure
    by fetching up to 5 rows from each table.

    Args:
        db_name (str): Path to the SQLite database file. Defaults to "recipes.db".

    Raises:
        Exception: If an error occurs during the database queries.
    """

    try:
        # Connect to the SQLite database
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()

        # Test users table
        print("--- Users ---")
        cursor.execute("SELECT * FROM users LIMIT 5")
        for row in cursor.fetchall():
            print(row)

        # Test recipes table
        print("--- Recipes ---")
        cursor.execute("SELECT * FROM recipes LIMIT 5")
        for row in cursor.fetchall():
            print(row)

        # Test ingredients table
        print("\n--- Ingredients ---")
        cursor.execute("SELECT * FROM ingredients LIMIT 5")
        for row in cursor.fetchall():
            print(row)

        # Test nutrition_info table
        print("\n--- Nutrition Info ---")
        cursor.execute("SELECT * FROM nutrition_info LIMIT 5")
        for row in cursor.fetchall():
            print(row)

        # Test cooking_steps table
        print("\n--- Cooking Steps ---")
        cursor.execute("SELECT * FROM cooking_steps LIMIT 5")
        for row in cursor.fetchall():
            print(row)

        # Test tags table
        print("\n--- Tags ---")
        cursor.execute("SELECT * FROM tags LIMIT 5")
        for row in cursor.fetchall():
            print(row)

        # Test recipe_tags table
        print("\n--- Recipe Tags ---")
        cursor.execute("SELECT * FROM recipe_tags LIMIT 5")
        for row in cursor.fetchall():
            print(row)

        # Test user_tags table
        print("\n--- User Tags ---")
        cursor.execute("""
            SELECT 
                users.username, 
                tags.tag_name
            FROM 
                users
            INNER JOIN 
                user_tags ON users.user_id = user_tags.user_id
            INNER JOIN 
                tags ON user_tags.tag_id = tags.tag_id
            LIMIT 5;
        """)
        for row in cursor.fetchall():
            print(row)

        # Close the connection
        conn.close()
    except Exception as e:
        print(f"Error querying the database: {e}")

def test_complex_queries(db_name="recipes.db"):
    """
    Test complex queries on the database to validate its relationships and functionality.

    This function runs a series of queries to verify the integrity of relationships between
    database tables and ensures correct data retrieval for specific scenarios.

    Queries include:
    - Retrieving recipes with their ingredients.
    - Fetching user tags.
    - Identifying recipes tagged as 'Gluten-free'.
    - Listing recipes with multiple tags.
    - Calculating average calories across recipes.
    - Finding recipes preferred by specific users based on tags.

    Args:
        db_name (str): Path to the SQLite database file. Defaults to "recipes.db".

    Raises:
        Exception: If an error occurs during the database queries.
    """

    try:
        # Connect to the SQLite database
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()

        # Query 1: Get all recipes with their ingredients
        print("\n--- Recipes with Ingredients ---")
        cursor.execute("""
            SELECT 
                recipes.title,
                ingredients.ingredient_name,
                ingredients.quantity,
                ingredients.unit
            FROM 
                recipes
            INNER JOIN 
                ingredients ON recipes.recipe_id = ingredients.recipe_id
            LIMIT 10;
        """)
        for row in cursor.fetchall():
            print(row)

        # Query 2: Get all tags linked to users
        print("\n--- Users and Their Tags ---")
        cursor.execute("""
            SELECT 
                users.username,
                tags.tag_name
            FROM 
                users
            INNER JOIN 
                user_tags ON users.user_id = user_tags.user_id
            INNER JOIN 
                tags ON user_tags.tag_id = tags.tag_id
            LIMIT 10;
        """)
        for row in cursor.fetchall():
            print(row)

        # Query 3: Recipes tagged as 'Gluten-free'
        print("\n--- Recipes Tagged 'Gluten-free' ---")
        cursor.execute("""
            SELECT 
                recipes.title
            FROM 
                recipes
            INNER JOIN 
                recipe_tags ON recipes.recipe_id = recipe_tags.recipe_id
            INNER JOIN 
                tags ON recipe_tags.tag_id = tags.tag_id
            WHERE 
                tags.tag_name = 'Gluten-free'
            LIMIT 10;
        """)
        for row in cursor.fetchall():
            print(row)

        # Query 4: Recipes with multiple tags
        print("\n--- Recipes with Multiple Tags ---")
        cursor.execute("""
            SELECT 
                recipes.title, 
                GROUP_CONCAT(tags.tag_name, ', ') AS tags
            FROM 
                recipes
            INNER JOIN 
                recipe_tags ON recipes.recipe_id = recipe_tags.recipe_id
            INNER JOIN 
                tags ON recipe_tags.tag_id = tags.tag_id
            GROUP BY 
                recipes.recipe_id
            HAVING 
                COUNT(tags.tag_id) > 1
            LIMIT 10;
        """)
        for row in cursor.fetchall():
            print(row)

        # Query 5: Average calories across recipes
        print("\n--- Average Calories ---")
        cursor.execute("""
            SELECT 
                AVG(CAST(nutrition_info.value AS FLOAT)) AS avg_calories
            FROM 
                nutrition_info
            WHERE 
                nutrition_info.category = 'energy';
        """)
        print(cursor.fetchone())

        # Query 6: Recipes preferred by 'johndoe' based on tags
        print("\n--- Recipes Preferred by 'johndoe' ---")
        cursor.execute("""
            SELECT 
                recipes.title
            FROM 
                recipes
            INNER JOIN 
                recipe_tags ON recipes.recipe_id = recipe_tags.recipe_id
            INNER JOIN 
                tags ON recipe_tags.tag_id = tags.tag_id
            INNER JOIN 
                user_tags ON tags.tag_id = user_tags.tag_id
            INNER JOIN 
                users ON user_tags.user_id = users.user_id
            WHERE 
                users.username = 'johndoe'
            LIMIT 10;
        """)
        for row in cursor.fetchall():
            print(row)

        # Close the connection
        conn.close()
    except Exception as e:
        print(f"Error querying the database: {e}")

if __name__ == "__main__":
    test_get_all_rows_database()
    test_complex_queries()
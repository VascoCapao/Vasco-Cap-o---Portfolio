import sqlite3
import json
import random

def get_all_tags():   
    """
    Retrieve all unique tags from the dataset.

    Reads the JSON data file and extracts all unique tags associated with recipes.

    Returns:
        list[str]: A list of unique tags found in the dataset.
    """
    with open("SmartBite/chatbot/data/Data_Collection/processed_data.json", "r") as file:
        data = json.load(file)
    return list(set([tag for recipe in data for tag in recipe["recipe_tags"]]))

def generate_users_data(n):
    """
    Generate random user data with usernames, passwords, and tags.

    Args:
        n (int): The number of users to generate.

    Returns:
        list[dict]: A list of dictionaries containing generated user data, each with:
            - username (str): The generated username.
            - password (str): The generated password.
            - tags (list[str]): A list of tags associated with the user.
    """

    # List of first and last names for generating usernames
    first_names = ["James", "Mary", "Michael", "Patricia", "Robert", "Jennifer", "John", "Linda", "David", "Elizabeth", "William", "Barbara"]
    last_names = ["Smith", "James", "Johnson", "Williams", "Miller", "Irving", "Westbrook", "Edwards", "Bryant"]

    # Get all available tags from the dataset
    all_tags = get_all_tags()

    # Store the generated users
    output = []

    while len(output) < n:
        # Generate a random username by combining first and last names
        username = f"{random.choice(first_names)}{random.choice(last_names)}{random.randint(100, 999)}"

        # Ensure usernames are unique
        if any(user["username"] == username for user in output):
            continue

        # Generate a random password
        password = f"{random.choice(first_names).lower()}{random.randint(1000, 9999)}"

        # Assign random tags to the user
        user_tags = random.sample(all_tags, random.randint(1, 3))  # Each user has 1-3 tags

        # Add the user to the output list
        output.append({
            "username": username,
            "password": password,
            "tags": user_tags
        })

    return output

def populate_users_table():
    """
    Populate the users table with generated example data and link users with tags.

    This function inserts user data into the `users` table and associates each user
    with tags in the `user_tags` table, creating new tags in the `tags` table if needed.

    Raises:
        Exception: If an error occurs while populating the database.
    """

    users_data = generate_users_data(50)

    try:
        conn = sqlite3.connect("SmartBite/chatbot/data/Database/recipes.db")
        cursor = conn.cursor()

        for user in users_data:
            # Insert user
            cursor.execute("""
                INSERT OR IGNORE INTO users (username, password)
                VALUES (?, ?)
            """, (user["username"], user["password"]))
            user_id = cursor.lastrowid

            # Link user with tags
            for tag in user.get("tags", []):
                # Check if the tag already exists
                cursor.execute("SELECT tag_id FROM tags WHERE tag_name = ?", (tag,))
                tag_row = cursor.fetchone()

                if tag_row:
                    tag_id = tag_row[0]
                else:
                    # Insert new tag
                    cursor.execute("INSERT INTO tags (tag_name) VALUES (?)", (tag,))
                    tag_id = cursor.lastrowid

                # Link user with tag
                cursor.execute("""
                    INSERT OR IGNORE INTO user_tags (
                        user_id, tag_id
                    ) VALUES (?, ?)
                """, (user_id, tag_id))

        conn.commit()
        print("Users table and user_tags table populated successfully.")
    except Exception as e:
        print(f"Error populating users table: {e}")
    finally:
        conn.close()

def populate_database(data_file="SmartBite/chatbot/data/Data_Collection/processed_data.json", db_name="SmartBite/chatbot/data/Database/recipes.db"):
    """
    Populate the SQLite database with data from the JSON file.

    Reads recipe data from the specified JSON file and populates the database tables:
    `recipes`, `ingredients`, `nutrition_info`, `cooking_steps`, and `tags`.
    Links recipes to their associated tags in the `recipe_tags` table.

    Args:
        data_file (str): Path to the JSON file containing recipe data. Defaults to 
                         "SmartBite/chatbot/data/Data_Collection/processed_data.json".
        db_name (str): Path to the SQLite database file. Defaults to 
                       "SmartBite/chatbot/data/Database/recipes.db".

    Raises:
        Exception: If an error occurs while populating the database.
    """

    try:
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()

        with open(data_file, "r") as file:
            data = json.load(file)

        # Insert data into the recipes table
        for recipe in data:
            cursor.execute("""
                INSERT INTO recipes (
                    title, url, collection_name, duration_prep_time, duration_cook_time, 
                    duration_notes, difficulty, makes, unit, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                recipe.get("recipe_title"),
                recipe.get("recipe_url"),
                recipe.get("collection_name"),
                recipe.get("duration_info", {}).get("prep_time"),
                recipe.get("duration_info", {}).get("cook_time"),
                recipe.get("duration_info", {}).get("notes"),
                recipe.get("extra_info", {}).get("difficulty"),
                recipe.get("extra_info", {}).get("makes"),
                recipe.get("extra_info", {}).get("unit"),
                recipe.get("extra_info", {}).get("notes")
            ))
            recipe_id = cursor.lastrowid

            # Insert data into the ingredients table
            for ingredient in recipe.get("ingredients", []):
                cursor.execute("""
                    INSERT INTO ingredients (
                        recipe_id, ingredient_name, quantity, unit, notes
                    ) VALUES (?, ?, ?, ?, ?)
                """, (
                    recipe_id,
                    ingredient.get("ingredient_name"),
                    ingredient.get("quantity"),
                    ingredient.get("unit"),
                    ingredient.get("notes")
                ))

            # Insert data into the nutrition_info table
            for nutrition in recipe.get("nutrition_info", []):
                cursor.execute("""
                    INSERT INTO nutrition_info (
                        recipe_id, category, value, unit
                    ) VALUES (?, ?, ?, ?)
                """, (
                    recipe_id,
                    nutrition.get("category"),
                    nutrition.get("value"),
                    nutrition.get("unit")
                ))

            # Insert data into the cooking_steps table
            for step in recipe.get("cooking_steps", []):
                cursor.execute("""
                    INSERT INTO cooking_steps (
                        recipe_id, step_order, instruction
                    ) VALUES (?, ?, ?)
                """, (
                    recipe_id,
                    step[0],
                    step[1]
                ))

            # Insert data into the tags table and link them with the recipe
            for tag in recipe.get("recipe_tags", []):
                cursor.execute("SELECT tag_id FROM tags WHERE tag_name = ?", (tag,))
                tag_row = cursor.fetchone()

                if tag_row:
                    tag_id = tag_row[0]
                else:
                    cursor.execute("INSERT INTO tags (tag_name) VALUES (?)", (tag,))
                    tag_id = cursor.lastrowid

                cursor.execute("""
                    INSERT INTO recipe_tags (
                        recipe_id, tag_id
                    ) VALUES (?, ?)
                """, (recipe_id, tag_id))

        conn.commit()
        print(f"Data from '{data_file}' imported successfully into '{db_name}'.")
    except Exception as e:
        print(f"Error populating the database: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    populate_database()
    populate_users_table()
import sqlite3
import re

class IngredientUpdateChain:
    def __init__(self, db_name="SmartBite/chatbot/data/Database/recipes.db"):
        """
        Initialize the IngredientUpdateChain with the database file.

        Args:
            db_name (str): Path to the SQLite database file. Defaults to 
                        "SmartBite/chatbot/data/Database/recipes.db".
        """

        self.db_name = db_name

    def execute_query(self, query, params=()):
        """
        Execute an SQL query on the database.

        Args:
            query (str): The SQL query to execute.
            params (tuple): Parameters for the SQL query. Defaults to an empty tuple.

        Returns:
            list or str: Query results as a list of tuples, or an error message if an exception occurs.
        """
        try:
            conn = sqlite3.connect(self.db_name)
            cursor = conn.cursor()
            cursor.execute(query, params)
            conn.commit()
            result = cursor.fetchall()
            return result
        except sqlite3.Error as e:
            return f"Database error: {e}"
        finally:
            conn.close()

    def validate_user(self, user_id):
        """
        Validate if the user exists in the database.

        Args:
            user_id (int): ID of the user to validate.

        Returns:
            bool: True if the user exists, False otherwise.
        """

        query = "SELECT 1 FROM users WHERE user_id = ?"
        result = self.execute_query(query, (user_id,))
        return bool(result)

    def search_ingredients(self, keyword):
        """
        Search for ingredients in the database matching a keyword.

        Args:
            keyword (str): The keyword to search for in ingredient names.

        Returns:
            list[tuple]: A list of tuples containing ingredient IDs and names.
        """
        query = """
        SELECT ingredient_id, ingredient_name 
        FROM ingredients 
        WHERE ingredient_name LIKE ?
        """
        return self.execute_query(query, (f"%{keyword}%",))

    def save_inventory_ingredients(self, user_id, ingredient_ids):
        """
        Add ingredients to the user's inventory.

        Args:
            user_id (int): ID of the user.
            ingredient_ids (list[int]): List of ingredient IDs to add.

        Returns:
            str: Messages indicating the result of each addition.
        """

        inventory_id_query = "SELECT inventory_id FROM inventory WHERE user_id = ?"
        inventory_id_result = self.execute_query(inventory_id_query, (user_id,))

        if not inventory_id_result:
            create_inventory_query = "INSERT INTO inventory (user_id) VALUES (?)"
            self.execute_query(create_inventory_query, (user_id,))
            inventory_id_result = self.execute_query(inventory_id_query, (user_id,))

        inventory_id = inventory_id_result[0][0]
        messages = []

        for ingredient_id in ingredient_ids:
            check_query = """
            SELECT 1 FROM inventory_ingredients 
            WHERE inventory_id = ? AND ingredient_id = ?
            """
            exists = self.execute_query(check_query, (inventory_id, ingredient_id))

            if exists:
                messages.append(f"Ingredient ID {ingredient_id} is already in the inventory.")
            else:
                insert_query = """
                INSERT INTO inventory_ingredients (inventory_id, ingredient_id)
                VALUES (?, ?)
                """
                self.execute_query(insert_query, (inventory_id, ingredient_id))
                messages.append(f"Added ingredient ID {ingredient_id} to inventory.")

        return "\n".join(messages)

    def remove_inventory_ingredients(self, user_id, ingredient_ids):
        """
        Remove ingredients from the user's inventory.

        Args:
            user_id (int): ID of the user.
            ingredient_ids (list[int]): List of ingredient IDs to remove.

        Returns:
            str: Messages indicating the result of each removal.
        """

        inventory_id_query = "SELECT inventory_id FROM inventory WHERE user_id = ?"
        inventory_id_result = self.execute_query(inventory_id_query, (user_id,))

        if not inventory_id_result:
            return "No inventory found for the specified user. Please set up an inventory first."

        inventory_id = inventory_id_result[0][0]
        messages = []

        for ingredient_id in ingredient_ids:
            check_query = """
            SELECT 1 FROM inventory_ingredients 
            WHERE inventory_id = ? AND ingredient_id = ?
            """
            exists = self.execute_query(check_query, (inventory_id, ingredient_id))

            if exists:
                delete_query = """
                DELETE FROM inventory_ingredients 
                WHERE inventory_id = ? AND ingredient_id = ?
                """
                self.execute_query(delete_query, (inventory_id, ingredient_id))
                messages.append(f"Removed ingredient ID {ingredient_id} from inventory.")
            else:
                messages.append(f"Ingredient ID {ingredient_id} is not in the inventory.")

        return "\n".join(messages)


    def parse_input(self, user_input):
        """
        Parse user input to extract the action ('add' or 'remove') 
        and the ingredient(s) mentioned.

        Args:
            user_input (str): The user's input string.

        Returns:
            tuple: A tuple containing the action (str) and the keyword(s) (str), 
                or (None, None) if no valid action is found.
        """

        # Define patterns for different actions
        patterns = {
            "add": r"(add|bought|acquired|get|got|save)\s(.+)",
            "remove": r"(remove|ran out of|don't have|used up|no more|finished)\s(.+)",
        }

        for action, pattern in patterns.items():
            match = re.search(pattern, user_input.lower())
            if match:
                return action, match.group(2).strip()

        return None, None

    def run(self, user_input, user_id):
        """
        Run the chain to process the user's request for managing inventory.

        Args:
            user_input (str): The user's input describing the action and ingredients.
            user_id (int): ID of the user making the request.

        Returns:
            str: Response message indicating the result of the action.
        """

        if not self.validate_user(user_id):
            return "User ID is invalid. Please ensure you are registered."

        action, keyword = self.parse_input(user_input)
        if not action or not keyword:
            return "I couldn't understand your request. Please specify the action (add/remove) and the ingredient(s)."

        found_ingredients = self.search_ingredients(keyword)

        if not found_ingredients:
            return f"I couldn't find any ingredient matching '{keyword}'. Please ensure the ingredient name is correct."

        ingredient_ids = [ingredient_id for ingredient_id, _ in found_ingredients]
        if action == "add":
            return self.save_inventory_ingredients(user_id, ingredient_ids)
        elif action == "remove":
            return self.remove_inventory_ingredients(user_id, ingredient_ids)
        else:
            return "Invalid action specified. Please use 'add' or 'remove'."

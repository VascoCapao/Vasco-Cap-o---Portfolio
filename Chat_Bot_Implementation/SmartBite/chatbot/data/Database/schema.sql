-- Users Table
CREATE TABLE IF NOT EXISTS users (
    user_id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    password TEXT NOT NULL
);

-- Recipes Table
CREATE TABLE IF NOT EXISTS recipes (
    recipe_id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    url TEXT,
    collection_name TEXT,
    duration_prep_time TEXT,
    duration_cook_time TEXT,
    duration_notes TEXT,
    difficulty TEXT,
    makes TEXT,
    unit TEXT,
    notes TEXT
);

-- Ingredients Table
CREATE TABLE IF NOT EXISTS ingredients (
    ingredient_id INTEGER PRIMARY KEY AUTOINCREMENT,
    recipe_id INTEGER NOT NULL,
    ingredient_name TEXT NOT NULL,
    quantity TEXT,
    unit TEXT,
    notes TEXT,
    FOREIGN KEY (recipe_id) REFERENCES recipes(recipe_id)
);

-- Nutrition Info Table
CREATE TABLE IF NOT EXISTS nutrition_info (
    nutrition_id INTEGER PRIMARY KEY AUTOINCREMENT,
    recipe_id INTEGER NOT NULL,
    category TEXT NOT NULL,
    value TEXT,
    unit TEXT,
    FOREIGN KEY (recipe_id) REFERENCES recipes(recipe_id)
);

-- Cooking Steps Table
CREATE TABLE IF NOT EXISTS cooking_steps (
    step_id INTEGER PRIMARY KEY AUTOINCREMENT,
    recipe_id INTEGER NOT NULL,
    step_order INTEGER NOT NULL,
    instruction TEXT NOT NULL,
    FOREIGN KEY (recipe_id) REFERENCES recipes(recipe_id)
);

-- Tags Table
CREATE TABLE IF NOT EXISTS tags (
    tag_id INTEGER PRIMARY KEY AUTOINCREMENT,
    tag_name TEXT UNIQUE NOT NULL
);

-- Recipe Tags Table
CREATE TABLE IF NOT EXISTS recipe_tags (
    recipe_id INTEGER NOT NULL,
    tag_id INTEGER NOT NULL,
    FOREIGN KEY (recipe_id) REFERENCES recipes(recipe_id),
    FOREIGN KEY (tag_id) REFERENCES tags(tag_id),
    PRIMARY KEY (recipe_id, tag_id)
);

-- User Tags Table
CREATE TABLE IF NOT EXISTS user_tags (
    user_id INTEGER NOT NULL,
    tag_id INTEGER NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(user_id),
    FOREIGN KEY (tag_id) REFERENCES tags(tag_id),
    PRIMARY KEY (user_id, tag_id)
);

-- Inventory Table
CREATE TABLE IF NOT EXISTS inventory (
    inventory_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL UNIQUE,
    FOREIGN KEY (user_id) REFERENCES users(user_id)
);

-- Inventory Ingredients Table
CREATE TABLE IF NOT EXISTS inventory_ingredients (
    inventory_id INTEGER NOT NULL,
    ingredient_id INTEGER NOT NULL,
    FOREIGN KEY (inventory_id) REFERENCES inventory(inventory_id),
    FOREIGN KEY (ingredient_id) REFERENCES ingredients(ingredient_id),
    PRIMARY KEY (inventory_id, ingredient_id)
);

-- Favorite Recipe Table
CREATE TABLE IF NOT EXISTS favorite_recipe (
    favorite_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    recipe_id INTEGER NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(user_id),
    FOREIGN KEY (recipe_id) REFERENCES recipes(recipe_id),
    UNIQUE(user_id, recipe_id) -- Ensure no duplicate favorite entries for the same user and recipe
);

CREATE TRIGGER create_inventory_after_user_insert
AFTER INSERT ON users
BEGIN
    INSERT INTO inventory (user_id) VALUES (NEW.user_id);
END;
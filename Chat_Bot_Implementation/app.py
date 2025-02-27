import streamlit as st
import sqlite3
import bcrypt  # For secure password hashing
from SmartBite.chatbot.bot import SmartBiteBot  # Import the chatbot class
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(layout="wide")

# Set custom theme for sidebar background color
st.markdown(
    """
    <style>
        [data-testid="stSidebar"] {
            background-color: #FFE4B5; /* Light orange color */
            display: flex;
            flex-direction: column;
            justify-content: space-between; /* Ensure items are spaced */
        }
        .sidebar-image-container {
            margin-top: auto; /* Push the image to the bottom */
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Database setup with hashed passwords
def init_db():
    """
    Initialize the SQLite database:
    - Creates a 'users' table if it doesn't already exist.
    - The 'users' table stores usernames and hashed passwords.
    """
    conn = sqlite3.connect('SmartBite/chatbot/data/Database/recipes.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (
                    username TEXT PRIMARY KEY,
                    password TEXT
                )''')
    conn.commit()
    conn.close()

def hash_password(password):
    """
    Hashes a simple text password using bcrypt.
    """
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

def verify_password(password, hashed_password):
    """
    Verifies a password against a hashed version.
    Returns True if they match, False otherwise.
    """
    return bcrypt.checkpw(password.encode('utf-8'), hashed_password)

def register_user(username, password):
    """
    Registers a new user with a hashed password.
    Displays success or error messages based on the outcome.
    """
    # Connect to the SQLite database
    conn = sqlite3.connect('SmartBite/chatbot/data/Database/recipes.db')
    c = conn.cursor()

    # Get the system's current language
    current_language = st.session_state.language if "language" in st.session_state else "EN"

    # Dictionary of translations
    translations = {
        "EN": {
            "register_success": "User registered successfully!",
            "register_login_info": "Please log in to start your session.",
            "username_exists": "Username already exists!"
        },
        "PT": {
            "register_success": "Usuário registrado com sucesso!",
            "register_login_info": "Por favor, faça login para iniciar sua sessão.",
            "username_exists": "Nome de usuário já existe!"
        }
    }

    # Select translations based on the current language
    t = translations[current_language]

    hashed_pw = hash_password(password)
    try:
        # Insert the new user into the database
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed_pw))
        conn.commit()
        st.success(t["register_success"])  # Display translated success message
        st.info(t["register_login_info"])  # Informational message after registration
    except sqlite3.IntegrityError:
        st.error(t["username_exists"])  # Display translated error message
    finally:
        conn.close()

def login_user(username, password):
    """
    Authenticates a user by verifying their credentials.
    Returns True if login is successful, False otherwise.
    """
    conn = sqlite3.connect('SmartBite/chatbot/data/Database/recipes.db')
    c = conn.cursor()
    c.execute("SELECT user_id, password FROM users WHERE username = ?", (username,))
    result = c.fetchone()
    conn.close()
    if result and verify_password(password, result[1]):
        st.session_state["logged_in"] = True
        st.session_state["username"] = username
        st.session_state["user_id"] = result[0]   
        return True
    return False

def set_language():
    """
    Displays a language selection dropdown in the sidebar.
    Updates the application language based on user input.
    """
    # Labels for language selection
    language_labels = {
        "EN": "Select Language",
        "PT": "Selecionar Idioma"
    }

    if "language" not in st.session_state:
        st.session_state.language = "EN"

    # Get current language
    current_language = st.session_state.language

    # Show language selector with translated label
    st.sidebar.selectbox(
        language_labels[current_language],
        ["EN", "PT"],
        index=0 if current_language == "EN" else 1,
        key="language"
    )

def main():
    """
    Handles navigation between pages and configures the sidebar.
    Displays a welcome message for logged in users.
    """
    set_language()

    # Translation for sidebar items
    sidebar_labels = {
        "EN": {
            "navigation_title": "Navigation Page:",
            "go_to": "Go to",
            "home": "Home",
            "about": "About",
            "recipe_gallery": "Recipe Gallery",
            "login_register": "Login/Register",
            "chatbot": "Chatbot",
            "faq": "FAQ",
        },
        "PT": {
            "navigation_title": "Página de Navegação:",
            "go_to": "Ir para",
            "home": "Início",
            "about": "Sobre",
            "recipe_gallery": "Galeria de Receitas",
            "login_register": "Login/Registrar",
            "chatbot": "Chatbot",
            "faq": "FAQ",
        },
    }

    # Add custom CSS to position the "Welcome, [username]" message consistently
    st.markdown(
        """
        <style>
            .welcome-message {
                position: sticky;
                top: 0; /* Posiciona no topo do layout principal */
                z-index: 999; /* Garante que fique acima das imagens */
                background-color: transparent; /* Mesma cor da barra lateral */
                padding: 10px;
                text-align: center;
                font-size: 1.2rem;
                font-weight: bold;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Display the message "Welcome, [username]" if the user is logged in
    if "logged_in" in st.session_state and st.session_state.logged_in:
        welcome_message = (
            f"Welcome, {st.session_state['username']}!" 
            if st.session_state.language == "EN" 
            else f"Bem-vindo(a), {st.session_state['username']}!"
        )
        st.markdown(f"<div class='welcome-message'>{welcome_message}</div>", unsafe_allow_html=True)

    # Sidebar configuration based on the language
    current_language = st.session_state.language
    st.sidebar.title(sidebar_labels[current_language]["navigation_title"])
    menu = st.sidebar.radio(
        sidebar_labels[current_language]["go_to"],
        [
            sidebar_labels[current_language]["home"],
            sidebar_labels[current_language]["about"],
            sidebar_labels[current_language]["recipe_gallery"],
            sidebar_labels[current_language]["login_register"],
            sidebar_labels[current_language]["chatbot"],
            sidebar_labels[current_language]["faq"],
        ]
    )

    image_path = "images/Blue_and_Black_Clean___UN_Style_Civil_Society_SDG_Progress_Report-removebg-preview.png"
    st.sidebar.image(image_path)

    # Navigation through pages
    if menu == sidebar_labels[current_language]["home"]:
        home_page()
    elif menu == sidebar_labels[current_language]["about"]:
        about_page()
    elif menu == sidebar_labels[current_language]["recipe_gallery"]:
        recipe_gallery_page()
    elif menu == sidebar_labels[current_language]["login_register"]:
        login_register_page()
    elif menu == sidebar_labels[current_language]["chatbot"]:
        chatbot_page()
    elif menu == sidebar_labels[current_language]["faq"]:
        faq_page()

def home_page():
    """
    Displays the home page of SmartBite with an introductory text.
    Includes a description of the platform's features and navigation instructions.
    """
    # Apply custom CSS to control the image height and remove padding
    st.markdown(
        """
        <style>
            .css-18e3th9 {
                padding-top: 0rem;
                padding-right: 2rem;
                padding-left: 2rem;
            }
            .block-container {  /* Text padding */
            padding: 1rem 2rem;  /* Ensure text does not touch sidebar */
            }
            img.full-width {  /* Image full-width styling */
            width: 100%;
            height: 400px; /* Adjust this height as needed */
            object-fit: cover; /* Ensure proper scaling without distortion */
            margin-left: -2rem; /* Remove text padding effect */
            margin-right: -2rem;
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Display the full-width image
    st.image(
        "images/Blue and Black Clean & UN Style Civil Society SDG Progress Report (1).png",
        use_container_width=True,
    )
    st.title("Welcome to SmartBite!" if st.session_state.language == "EN" else "Bem-vindo ao SmartBite!")
    st.write("SmartBite revolutionizes the way you cook by offering personalized, real-time recipe suggestions and step-by-step guidance. Whether you're a seasoned chef or a beginner in the kitchen, our AI chatbot is here to assist you every step of the way." if st.session_state.language == "EN" else "O SmartBite revoluciona a maneira como você cozinha, oferecendo sugestões de receitas personalizadas em tempo real e orientações passo a passo. Seja você um chef experiente ou um iniciante na cozinha, nosso chatbot de IA está aqui para ajudar em cada etapa.")
    st.write(
        """
        ### What Can SmartBite Do For You
        - *Personalized Recipes*: Get recipes tailored to your preferences, dietary needs, and the ingredients you already have.
        - *Real-Time Cooking Support*: Have questions while cooking? SmartBite provides instant answers and tips.
        - *Healthy Choices Made Easy*: Receive nutritional information and ingredient swaps to align with your health goals.
        - *Reduce Food Waste*: Use up what’s in your pantry with recipe suggestions that minimize waste.
        - *Cultural Adaptation*: Discover recipes customized to your cultural preferences.

        ### Get Started Now
        Use the navigation on the left to:
        - Learn more about *SmartBite* on the *About* page.
        - Access pre-defined recipes.
        - *Log in* or *register* to access the *Chatbot* and explore interactive cooking features.
        - See the most *frequently asked questions* from users.

        ### Let us simplify your cooking experience!
        """ 
        if st.session_state.language == "EN" else
        """
        ### O que o SmartBite pode fazer por você
        - *Receitas Personalizadas*: Obtenha receitas adaptadas às suas preferências, necessidades alimentares e aos ingredientes que você já tem.
        - *Suporte em Tempo Real para Cozinhar*: Tem dúvidas enquanto cozinha? O SmartBite fornece respostas e dicas instantâneas.
        - *Escolhas Saudáveis Facilitadas*: Receba informações nutricionais e substituições de ingredientes alinhadas aos seus objetivos de saúde.
        - *Reduza o Desperdício de Alimentos*: Use o que está na sua despensa com sugestões de receitas que minimizam o desperdício.
        - *Adaptação Cultural*: Descubra receitas personalizadas às suas preferências culturais ou no seu idioma preferido.

        ### Comece Agora
        Use a navegação à esquerda para:
        - Saber mais sobre o *SmartBite* na página *Sobre*.
        - Ter acesso a receitas já pré-definidas.
        - *Fazer login* ou *registrar-se* para acessar o *Chatbot* e explorar recursos interativos de culinária.
        - Ver as *perguntas mais frequentes* dos utilizadores.
        ### Deixe-nos simplificar sua experiência culinária!
        """
    )
def about_page():
    """
    Displays the About page of SmartBite.
    Includes information on the platform's mission, vision, and values.
    """
    # Apply custom CSS to control the image height and remove padding
    st.markdown(
        """
        <style>
            .css-18e3th9 {
                padding-top: 0rem;
                padding-right: 2rem;
                padding-left: 2rem;
            }
            .block-container {  /* Text padding */
            padding: 1rem 2rem;  /* Ensure text does not touch sidebar */
            }
            img.full-width {  /* Image full-width styling */
            width: 100%;
            height: 400px; /* Adjust this height as needed */
            object-fit: cover; /* Ensure proper scaling without distortion */
            margin-left: -2rem; /* Remove text padding effect */
            margin-right: -2rem;
        </style>
        """,
        unsafe_allow_html=True,
    )
    
    st.image("images/Blue_and_Black_Clean___UN_Style_Civil_Society_SDG_Progress_Report-removebg-preview.png")
    st.title("About SmartBite" if st.session_state.language == "EN" else "Sobre o SmartBite")
    st.write(
        """
        ### Mission
        SmartBite aims to revolutionize the cooking experience by providing personalized, real-time recipe assistance powered by AI. We strive to make cooking accessible, enjoyable, and efficient for everyone by offering tailored solutions to users' dietary needs, ingredient availability, and cooking preferences.

        ### Vision
        To become the leading AI culinary partner, empowering users worldwide to explore and enjoy cooking, regardless of their skill level. SmartBite envisions a future where meal preparation is seamless, sustainable, and inclusive for all.

        ### Values
        - *Simplicity*: Making the process of finding and preparing recipes effortless by offering intuitive solutions tailored to users’ preferences and available ingredients.
        - *Health and Wellness*: Supporting users in achieving their nutritional goals with recipes aligned to their dietary needs, promoting wellness and balanced living.
        - *Empowerment and Inclusivity*: Ensuring the cooking experience is enjoyable and accessible for everyone, regardless of skill, background, or dietary preferences. We inspire confidence in cooking through personalized guidance and step-by-step instructions.
        - *Sustainability*: Encouraging sustainable cooking habits by minimizing food waste and utilizing available ingredients, promoting environmentally friendly choices.
        - *Innovation*: Continuously innovating and adapting to user needs by leveraging cutting-edge AI technologies, ensuring a personalized and forward-thinking cooking experience.
        """ if st.session_state.language == "EN" else
        """
        ### Missão
        O SmartBite visa revolucionar a experiência culinária, oferecendo assistência personalizada em receitas em tempo real com o poder da IA. Nosso objetivo é tornar a culinária acessível, agradável e eficiente para todos, oferecendo soluções adaptadas às necessidades dietéticas, disponibilidade de ingredientes e preferências culinárias dos usuários.

        ### Visão
        Tornar-se o parceiro culinário líder em IA, capacitando usuários em todo o mundo a explorar e desfrutar da culinária, independentemente do nível de habilidade. O SmartBite visualiza um futuro onde o preparo das refeições seja contínuo, sustentável e inclusivo para todos.

        ### Valores
        - *Simplicidade*: Tornar o processo de encontrar e preparar receitas mais fácil, oferecendo soluções intuitivas adaptadas às preferências e ingredientes disponíveis dos usuários.
        - *Saúde e Bem-estar*: Apoiar os usuários a alcançar seus objetivos nutricionais com receitas alinhadas às suas necessidades dietéticas, promovendo o bem-estar e um estilo de vida equilibrado.
        - *Empoderamento e Inclusividade*: Garantir que a experiência culinária seja agradável e acessível para todos, independentemente do nível de habilidade, histórico ou preferências alimentares. Inspiramos confiança na culinária por meio de orientações personalizadas e instruções passo a passo.
        - *Sustentabilidade*: Incentivar hábitos de cozinha sustentável, minimizando o desperdício de alimentos e utilizando os ingredientes disponíveis, promovendo escolhas ambientalmente conscientes.
        - *Inovação*: Continuamente inovar e se adaptar às necessidades dos usuários aproveitando tecnologias de IA de ponta, garantindo uma experiência culinária personalizada e visionária.
        """
    )

def chatbot_page():
    """
    Displays the chatbot page where users can interact with the SmartBite AI chatbot.
    Users can type messages to the chatbot and view responses in a chat history.
    Only accessible if the user is logged in.
    """
    # Check if the user is logged in
    if "logged_in" not in st.session_state or not st.session_state.logged_in:
        st.warning("Please log in to access the chatbot." if st.session_state.language == "EN" else "Faça login para acessar o chatbot.")
        return

    st.title("Chat with SmartBite" if st.session_state.language == "EN" else "Converse com o SmartBite")
    st.write("Type your message below to interact with our intelligent chatbot." if st.session_state.language == "EN" else "Digite sua mensagem abaixo para interagir com nosso chatbot inteligente.")

    # Initialize chatbot instance if not already created
    if "chatbot_instance" not in st.session_state:
        st.session_state.chatbot_instance = SmartBiteBot(
            user_id=st.session_state.get("user_id"),
            conversation_id="session_1"
        )

    # Initialize chat history if not already created
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

        # Input box and send button inside a form
    with st.form(key="chat_form", clear_on_submit=True):
        user_input = st.text_input("Your Message:" if st.session_state.language == "EN" else "Sua Mensagem:")

        # Create two buttons in a single form
        col1, col2 = st.columns([1, 1]) 
        with col1:
            send_button = st.form_submit_button("Send" if st.session_state.language == "EN" else "Enviar")
        with col2:
            clear_button = st.form_submit_button("Clear Chat" if st.session_state.language == "EN" else "Limpar Conversa")

        if send_button and user_input.strip():
            # Process user input and bot response
            chatbot = st.session_state.chatbot_instance
            bot_response = chatbot.process_input(user_input.strip())
            # Append messages to chat history
            st.session_state.chat_history.append(("You", user_input.strip()))
            st.session_state.chat_history.append(("Bot", bot_response))

        if clear_button:
            # Clear the chat history
            st.session_state.chat_history = []

    st.subheader("Chat History" if st.session_state.language == "EN" else "Histórico de Conversas")
    for sender, message in reversed(st.session_state.chat_history):
        st.write(f"{sender}:** {message}")

def login_register_page():
    """
    Displays the login and registration page.
    Allows users to log in or register with their credentials.
    Handles both login validation and new user registration.
    """
    # Apply custom CSS to control the image height and remove padding
    st.markdown(
        """
        <style>
            .css-18e3th9 {
                padding-top: 0rem;
                padding-right: 2rem;
                padding-left: 2rem;
            }
            .block-container {  /* Text padding */
            padding: 1rem 2rem;  /* Ensure text does not touch sidebar */
            }
            img.full-width {  /* Image full-width styling */
            width: 100%;
            height: 400px; /* Adjust this height as needed */
            object-fit: cover; /* Ensure proper scaling without distortion */
            margin-left: -2rem; /* Remove text padding effect */
            margin-right: -2rem;
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Display the full-width image
    st.image(
        "images/360_F_417777825_v7o8RvkQhxpZkE0ZBD4xwzri5hGFHkO3.jpg",
        use_container_width=True,
    )
    # Check if the user is logged in.
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False  # Initial login state
    if "username" not in st.session_state:
        st.session_state.username = None  # Initialize the username

    # Check if the user is logged in
    if st.session_state.logged_in:
        logout_text = "Logout" if st.session_state.language == "EN" else "Sair"

        st.markdown(
            """
            <style>
                .center-content {
                    display: flex;
                    justify-content: center;
                    align-items: flex-start; /* Posiciona no topo */
                    height: 30vh; /* Reduz ainda mais a altura vertical */
                    margin-top: 1rem; /* Pequeno espaço acima */
                }
                .logout-button {
                    font-size: 1.2rem;
                    padding: 0.5rem 1rem;
                    background-color: #FFA500;
                    border: none;
                    color: white;
                    cursor: pointer;
                    border-radius: 5px;
                }
                .logout-button:hover {
                    background-color: #FF8C00;
                }
            </style>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<div class='center-content'>", unsafe_allow_html=True)
        if st.button(logout_text, key="logout"):
            st.session_state.logged_in = False
            st.session_state.username = None
            st.success(
                "You have been logged out."
                if st.session_state.language == "EN"
                else "Você foi desconectado."
            )
        st.markdown("</div>", unsafe_allow_html=True)

    else:
        # Login or Registration Page
        st.title(
            "Login or Register" if st.session_state.language == "EN" else "Login ou Registrar"
        )

        auth_mode = st.radio(
            "Choose an option" if st.session_state.language == "EN" else "Escolha uma opção",
            ["Login", "Register"] if st.session_state.language == "EN" else ["Login", "Registrar"],
        )

        username = st.text_input(
            "Username" if st.session_state.language == "EN" else "Usuário"
        )
        password = st.text_input(
            "Password" if st.session_state.language == "EN" else "Senha", type="password"
        )

        if auth_mode == "Login":
            if st.button("Login" if st.session_state.language == "EN" else "Entrar"):
                if username and password:
                    if login_user(username, password):
                        st.success(
                            "Logged in successfully!"
                            if st.session_state.language == "EN"
                            else "Login realizado com sucesso!"
                        )
                        st.session_state.logged_in = True
                        st.session_state.username = username
                    else:
                        st.error(
                            "Invalid username or password!"
                            if st.session_state.language == "EN"
                            else "Usuário ou senha inválidos!"
                        )
                else:
                    st.error(
                        "Please enter both username and password!"
                        if st.session_state.language == "EN"
                        else "Por favor, insira usuário e senha!"
                    )
        else:
            st.write(
                "The password must contain:"
                if st.session_state.language == "EN"
                else "A senha deve conter:"
            )
            st.markdown(
                """
                - At least 8 characters
                - At least one number
                - At least one letter
                """
                if st.session_state.language == "EN"
                else """
                - Pelo menos 8 caracteres
                - Pelo menos um número
                - Pelo menos uma letra
                """
            )
            if st.button("Register" if st.session_state.language == "EN" else "Registrar"):
                if username and password:
                    if len(password) < 8:
                        st.error(
                            "Password must be at least 8 characters long!"
                            if st.session_state.language == "EN"
                            else "A senha deve ter pelo menos 8 caracteres!"
                        )
                    elif not any(char.isdigit() for char in password):
                        st.error(
                            "Password must contain at least one number!"
                            if st.session_state.language == "EN"
                            else "A senha deve conter pelo menos um número!"
                        )
                    elif not any(char.isalpha() for char in password):
                        st.error(
                            "Password must contain at least one letter!"
                            if st.session_state.language == "EN"
                            else "A senha deve conter pelo menos uma letra!"
                        )
                    else:
                        register_user(username, password)
                else:
                    st.error(
                        "Please enter both username and password!"
                        if st.session_state.language == "EN"
                        else "Por favor, insira usuário e senha!"
                    )


def recipe_gallery_page():
    """
    Displays a recipe gallery with filtering options and allows users to view detailed recipe instructions.

    Filters:
        - Cuisine
        - Diet
        - Ingredient

    Functionality:
        - Displays a recipe gallery with images and basic information.
        - Allows the user to select a recipe to view detailed instructions.
    """
    # Apply custom CSS to control the image height and remove the padding
    st.markdown(
        """
        <style>
            .css-18e3th9 {
                padding-top: 0rem;
                padding-right: 2rem;
                padding-left: 2rem;
            }
            .block-container {  /* Text padding */
            padding: 1rem 2rem;  /* Ensure text does not touch sidebar */
            }
            img.full-width {  /* Image full-width styling */
            width: 100%;
            height: 400px; /* Adjust this height as needed */
            object-fit: cover; /* Ensure proper scaling without distortion */
            margin-left: -2rem; /* Remove text padding effect */
            margin-right: -2rem;
        </style>
        """,
        unsafe_allow_html=True,
    )

    if "selected_recipe" not in st.session_state or not st.session_state.selected_recipe:
        st.image(
            "images/360_F_594821637_Rzb8t6sMmPQBylvBX1Kme9sgB0pcoeyi.jpg",
            use_container_width=True,
        )
        st.title("Recipe Gallery" if st.session_state.language == "EN" else "Galeria de Receitas")

    if "selected_recipe" not in st.session_state or not st.session_state.selected_recipe:
        st.write(
            """
            Welcome to the Recipe Gallery!  
            This page allows you to explore pre-defined recipe options that are ready to use and may inspire your next meal.
            Use the available filters to customize your search based on cuisine, dietary preferences, or ingredients you already have.
            Start cooking with ease and discover new ideas!
            """ if st.session_state.language == "EN" else
            """
            Bem-vindo à Galeria de Receitas!  
            Esta página permite que você explore opções de receitas pré-definidas que já estão prontas para uso e podem inspirar sua próxima refeição.
            Use os filtros disponíveis para personalizar sua busca com base na cozinha, preferências alimentares ou ingredientes que você já possui.
            Comece a cozinhar com facilidade e descubra novas ideias!
            """
        )

    # Translation of labels
    labels = {
        "filters": {
            "EN": "Filters",
            "PT": "Filtros"
        },
        "cuisine": {
            "EN": "*Cuisine*",
            "PT": "*Cozinha*"
        },
        "diet": {
            "EN": "*Diet*",
            "PT": "*Dieta*"
        },
        "ingredients": {
            "EN": "*Ingredients*",
            "PT": "*Ingredientes*"
        },
        "instructions": {
            "EN": "*Instructions*",
            "PT": "*Instruções*"
        }
    }
    # Translation mapping for filters
    cuisine_translation = {
        "Todas": "All",
        "Italiana": "Italian",
        "Indiana": "Indian",
        "Chinesa": "Chinese",
        "Mexicana": "Mexican",
    }
    diet_translation = {
        "Todas": "All",
        "Vegetariana": "Vegetarian",
        "Vegana": "Vegan",
        "Sem Glúten": "Gluten-Free",
    }

    # Translation of options
    cuisine_options = {
        "EN": ["All", "Italian", "Indian", "Chinese", "Mexican"],
        "PT": ["Todas", "Italiana", "Indiana", "Chinesa", "Mexicana"]
    }
    diet_options = {
        "EN": ["All", "Vegetarian", "Vegan", "Gluten-Free"],
        "PT": ["Todas", "Vegetariana", "Vegana", "Sem Glúten"]
    }

    # Translated buttons
    button_texts = {
        "view_recipe": {
            "EN": "View Recipe",
            "PT": "Ver Receita"
        },
        "back_to_gallery": {
            "EN": "Back to Gallery",
            "PT": "Voltar para a Galeria"
        }
    }

  # Ensure that only one header is displayed.
    current_language = st.session_state.language
    st.sidebar.header(labels["filters"][current_language])
    
    # Language selection
    current_language = st.session_state.language 

    cuisine_filter = st.sidebar.selectbox(
        "Select Cuisine" if current_language == "EN" else "Selecionar Cozinha", 
        cuisine_options[current_language]
    )
    diet_filter = st.sidebar.selectbox(
        "Select Diet" if current_language == "EN" else "Selecionar Dieta", 
        diet_options[current_language]
    )
    ingredient_filter = st.sidebar.text_input(
        "Enter Ingredient" if current_language == "EN" else "Insira o Ingrediente"
    )

    # Convert the filters back to the original values used in the recipes
    if current_language == "PT":
        cuisine_filter = cuisine_translation[cuisine_filter]
        diet_filter = diet_translation[diet_filter]

    # Predefined recipes
    recipes = [
        {
            "name": {
                "EN": "Spaghetti Carbonara",
                "PT": "Espaguete Carbonara"
            },
            "cuisine": {
                "EN": "Italian",
                "PT": "Italiana"
            },
            "diet": {
                "EN": "Vegetarian",
                "PT": "Vegetariana"
            },
            "ingredients": {
                "EN": ["Spaghetti", "Eggs", "Cheese"],
                "PT": ["Espaguete", "Ovos", "Queijo"]
            },
            "image_path": "images/1500x1500-Photo-5_1950-How-to-Make-VEGETARIAN-CARBONARA-Like-an-Italian-V1.jpg",
            "instructions": {
                "EN": """
                1. Cook the Spaghetti:

                    - Bring a large pot of salted water to a boil.

                    - Add the spaghetti and cook according to the package instructions until al dente.

                    - Reserve about 1 cup of pasta water, then drain the spaghetti.

                2. Mix eggs and cheese:

                    - In a bowl, beat the eggs and mix in the grated cheese until combined. Set aside.
 
                3. Combine Pasta and Sauce:

                    - Return the drained spaghetti to the warm pot and pour the egg and cheese mixture over it.

                    - Toss vigorously to coat the spaghetti. The residual heat from the pasta will cook the eggs, creating a creamy sauce.

                    - If the mixture seems too thick, add a little of the reserved pasta water to loosen it.  

                4. Serve hot.
                """,
                "PT": """
                1. Cozinhe o Espaguete:

                    - Ferva uma panela grande com água salgada.

                    - Adicione o espaguete e cozinhe.

                    - Reserve cerca de 1 chávena da água do cozimento e escorra o espaguete.

                2. Misture os ovos e o queijo:

                    - Numa tigela, bata os ovos e misture o queijo ralado até ficar homogêneo. Reserve.

                3. Combine o Espaguete com o Molho:

                    - Coloque o espaguete escorrido na panela ainda morna e despeje a mistura de ovos e queijo sobre ele.

                    - Misture vigorosamente para envolver o espaguete. O calor residual da massa cozinhará os ovos, criando um molho cremoso.

                    - Se a mistura parecer muito grossa, adicione um pouco da água reservada para soltar o molho.

                4. Sirva quente.
                """
            },
        },
        {
            "name": {
                "EN": "Paneer Tikka",
                "PT": "Tikka de Paneer"
            },
            "cuisine": {
                "EN": "Indian",
                "PT": "Indiana"
            },
            "diet": {
                "EN": "Vegetarian",
                "PT": "Vegetariana"
            },
            "ingredients": {
                "EN": ["Paneer", "Yogurt", "Spices"],
                "PT": ["Paneer", "Iogurte", "Especiarias"]
            },
            "image_path": "images/281122_7.jpg",
            "instructions": {
                "EN": """
                1. Prepare the Marinade: 

                    - In a large bowl, mix the yogurt with all the spices: garam masala, turmeric, red chili powder, coriander powder, cumin powder, and salt.

                    - Add the lemon juice to the mixture and stir well until it forms a smooth past.

                2. Marinate the Paneer: 

                    - Add the paneer cubes to the bowl with the marinade.

                    - Gently mix, ensuring all the paneer pieces are well coated with the marinade.

                    - Cover the bowl and let it marinate for at least 30 minutes (or up to 2 hours in the refrigerator for more flavor).

                3. Grill the Paneer:

                    - Heat a pan or grill with one tablespoon of oil over medium heat.

                    - Place the marinated paneer cubes on the pan.

                    - Grill for 2–3 minutes on each side until they are golden and slightly charred.

                4. Serve.
                """,
                "PT": """
                1. Prepare a Marinada:

                    - Numa tigela grande, misture o iogurte com todas as especiarias: garam masala, cúrcuma, pimenta vermelha, coentro em pó, cominho em pó e sal.

                    - Adicione o sumo de limão à mistura e misture bem até formar uma pasta homogênea.

                2. Marine o Paneer: 

                    - Adicione os cubos de paneer à tigela com a marinada.

                    - Misture suavemente garatindo que todos os pedaços de paneer são cobertos pela marinada.

                    - Cubra a tigela e deixe marinar por pelo menos 30 minutos (ou até 2 horas no fricorífico para obter mais sabor).

                3. Grelhe o Paneer:

                    - Aqueça uma frigideira ou grelha com uma colher de sopa de óleo em lume médio.

                    - Coloque os cubos de paneer marinados na frigideira.

                    - Grelhe por 2–3 minutos de cada lado até que estejam dourados.

                4. Sirva.
                """
            }
        },
        {
            "name": {
                "EN": "Kung Pao Chicken",
                "PT": "Frango Kung Pao"
            },
            "cuisine": {
                "EN": "Chinese",
                "PT": "Chinesa"
            },
            "diet": {
                "EN": "Non-Vegetarian",
                "PT": "Não-Vegetariana"
            },
            "ingredients": {
                "EN": ["Chicken", "Peanuts", "Chili"],
                "PT": ["Frango", "Amendoins", "Pimenta"]
            },
            "image_path": "images/kungpao-chicken.jpg",
            "instructions": {
                "EN": """
                1. Prepare the Chicken:  

                    - Cut the chicken into small cubes.

                    - Season with a pinch of salt and one tablespoon of soy sauce (optional). Let it rest for 10 minutes.

                2. Sauté the Peppers and Peanuts: 

                    - Heat a large skillet or wok over medium-high heat.

                    - Add 1 tablespoon of oil and the dried peppers. Fry for 30 seconds.

                    - Add the peanuts and sauté for 1–2 minutes. Remove and set aside.

                3. Cook the Chicken:

                    - In the same skillet, add another tablespoon of oil.

                    - Add the chicken and cook until it is golden and fully cooked (about 5–7 minutes).

                4. Combine Everything:

                    - Return the peanuts and peppers to the skillet with the chicken.

                    - Add 1 teaspoon of sugar (optional) and mix well to balance the flavors.

                    - Stir everything together for 1–2 minutes until the ingredients are well combined.

                5. Serve:

                    - Transfer to a plate and serve immediately with white rice, if desired.
                """,
                "PT": """
                1. Prepare o Frango:  
        
                    - Corte o frango em cubos pequenos.

                    - Tempere com uma pitada de sal e uma colher de sopa de molho de soja (opcional). Deixe descansar por 10 minutos.

                2. Refogue a Pimenta e os Amendoins: 

                    - Aqueça uma frigideira grande ou wok em fogo médio-alto.

                    - Adicione 1 colher de sopa de óleo e as pimentas secas. Frite por 30 segundos.

                    - Acrescente os amendoins e refogue por 1–2 minutos. Retire e reserve.

                3. Cozinhe o Frango:

                    - Na mesma frigideira, adicione mais 1 colher de sopa de óleo.

                    - Coloque o frango na frigideira e cozinhe até que esteja dourado e cozido por completo (cerca de 5–7 minutos).

                4. Combine Tudo:

                    - Volte os amendoins e as pimentas para a frigideira com o frango.

                    - Adicione 1 colher de chá de açúcar (opcional) e misture bem para balancear os sabores.

                    - Mexa tudo por 1–2 minutos até os ingredientes estarem bem incorporados.

                5. Sirva:

                    - Transfira para um prato e sirva imediatamente com arroz branco, se desejar.
                """
            }
        },
        {
            "name": {
                "EN": "Tacos",
                "PT": "Tacos"
            },
            "cuisine": {
                "EN": "Mexican",
                "PT": "Mexicana"
            },
            "diet": {
                "EN": "Gluten-Free",
                "PT": "Sem Glúten"
            },
            "ingredients": {
                "EN": ["Corn Tortilla", "Beef", "Cheese"],
                "PT": ["Tortilha de Milho", "Carne Moída", "Queijo"]
            },
            "image_path": "images/2021-09-22-16.41.34-scaled.jpg",
            "instructions": {
                "EN": """
                1. Cook beef:

                    - Heat a skillet over medium heat and add the cooking oil.

                    - Add the ground beef and cook, breaking it up with a spatula.

                    - Season with salt, pepper, and optional taco seasoning.

                    - Cook until the beef is browned and fully cooked. Remove from heat.

                2. Warm the Tortillas:

                    - Heat tortillas in a dry skillet for 20–30 seconds per side or microwave wrapped in a damp paper towel for 20–30 seconds.

                3. Assemble the Tacos:

                    - Place beef in the center of each tortilla.

                    - Sprinkle cheese on top and let it melt slightly from the heat of the beef.

                4. Serve.
                """,
                "PT": """
                1. Cozinhe a carne:

                    - Aqueça uma frigideira em fogo médio e adicione um pouco de óleo de cozinha.

                    - Adicione a carne moída e cozinhe, quebrando-a em pedaços com uma espátula.

                    - Tempere com sal, pimenta e tempero para taco (opcional).

                    - Cozinhe até que a carne esteja dourada e completamente cozida.

                2. Aqueça as Tortilhas:

                    - Aqueça as tortilhas numa frigideira seca por 20 a 30 segundos de cada lado ou no micro-ondas, envoltas em um pano úmido, por 20 a 30 segundos.

                3. Monte os Tacos:

                    - Coloque a carne no centro de cada tortilha.

                    - Polvilhe queijo por cima e deixe derreter levemente com o calor da carne.

                4. Sirva.
                """
            }
        }
    ]

    # Apply filters
    filtered_recipes = [
        recipe for recipe in recipes
        if (cuisine_filter == "All" or recipe["cuisine"]["EN"] == cuisine_filter) and
           (diet_filter == "All" or recipe["diet"]["EN"] == diet_filter) and
           (ingredient_filter.lower() in " ".join(recipe["ingredients"][current_language]).lower() if ingredient_filter else True)
    ]

    # Display the filtered recipes
    if "selected_recipe" not in st.session_state:
        st.session_state.selected_recipe = None

    if st.session_state.selected_recipe:
        # Show detailed recipe
        recipe = st.session_state.selected_recipe
        st.header(recipe["name"][current_language])
        st.image(recipe["image_path"], width=300)
        st.write(f"{labels['cuisine'][current_language]}: {recipe['cuisine'][current_language]}")
        st.write(f"{labels['diet'][current_language]}: {recipe['diet'][current_language]}")
        st.write(f"{labels['ingredients'][current_language]}: {', '.join(recipe['ingredients'][current_language])}")
        st.write(f"{labels['instructions'][current_language]}:")
        st.write(recipe["instructions"][current_language])
        if st.button(button_texts["back_to_gallery"][current_language]):
            st.session_state.selected_recipe = None
    else:
        # Show recipe gallery
        if filtered_recipes:
            for i, recipe in enumerate(filtered_recipes):
                cols = st.columns([2, 1])
                with cols[0]:
                    # Display recipe details in the left column
                    st.subheader(recipe["name"][current_language])
                    st.write(f"{labels['cuisine'][current_language]}: {recipe['cuisine'][current_language]} | *{labels['diet'][current_language]}*: {recipe['diet'][current_language]}")
                    st.write(f"{labels['ingredients'][current_language]}: {', '.join(recipe['ingredients'][current_language])}")
                    if st.button(button_texts["view_recipe"][current_language], key=f"view-recipe-{i}"):
                        st.session_state.selected_recipe = recipe
                with cols[1]:
                    if "image_path" in recipe:
                        st.image(recipe["image_path"], width=200)
        else:
            st.write(
                "No recipes found. Try adjusting the filters." 
                if st.session_state.language == "EN" else 
                "Nenhuma receita encontrada. Tente ajustar os filtros."
            )

def faq_page():
    """
    Renders a FAQ page with expandable questions and answers.
    """
    # Custom CSS for styling
    st.markdown(
        """
        <style>
            .block-container {  
                padding: 1rem 2rem; 
            }
            .faq-expander {
                margin-bottom: 1rem; /* Add spacing between questions */
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Top image
    st.image(
        "images/istockphoto-1318298919-612x612.jpg",
        use_container_width=True,
    )
    
    # Title
    st.title("Frequently Asked Questions" if st.session_state.get("language", "EN") == "EN" else "Perguntas Frequentes")
    
    introduction = {
            "EN": "This FAQ section is designed to provide quick and clear answers to common questions about the SmartBite platform. It helps users understand how to use the platform's features, customize their experience, and resolve common issues.",
            "PT": "Esta seção de perguntas frequentes foi criada para fornecer respostas rápidas e claras às dúvidas comuns sobre a plataforma SmartBite. Ela ajuda os usuários a entender como usar os recursos da plataforma, personalizar sua experiência e resolver problemas comuns."
        }
    st.write(introduction[st.session_state["language"]])

    # Questions and answers
    faq = {
        "EN": [
            ("What is SmartBite?", "SmartBite is an AI-powered chatbot that offers personalized recipe recommendations, real-time assistance during preparation, ingredient substitution suggestions, and nutritional information."),
            ("How does SmartBite help me cook?", "SmartBite provides personalized recipes based on the ingredients you already have, your dietary preferences, and cooking skills. Additionally, it offers step-by-step guidance and answers questions in real time."),
            ("Can the chatbot help with dietary restrictions?", "Yes, SmartBite adapts recipes to accommodate dietary restrictions such as vegetarianism, veganism, or gluten-free diets."),
            ("Can I ask for help during recipe preparation?", "Yes, the chatbot provides real-time support for questions and instructions while you cook."),
            ("How are the calories in recipes calculated?", "SmartBite uses nutritional databases to calculate calories and provide accurate information."),
        ],
        "PT": [
            ("O que é o SmartBite?", "O SmartBite é um chatbot baseado em inteligência artificial que oferece recomendações personalizadas de receitas, assistência em tempo real durante o preparo, sugestões de substituições de ingredientes e informações nutricionais."),
            ("Como o SmartBite me ajuda a cozinhar?", "O SmartBite oferece receitas personalizadas com base nos ingredientes que você já possui, preferências dietéticas e habilidades culinárias. Além disso, fornece orientações passo a passo e responde dúvidas em tempo real."),
            ("O chatbot pode ajudar com restrições alimentares?", "Sim, o SmartBite adapta receitas para atender restrições alimentares, como vegetarianismo, veganismo, ou dietas sem glúten."),
            ("Posso pedir ajuda durante o preparo de uma receita?", "Sim, o chatbot fornece suporte em tempo real para dúvidas e instruções enquanto você cozinha."),
            ("Como são calculadas as calorias das receitas?", "O SmartBite utiliza bancos de dados nutricionais para calcular as calorias e fornecer informações precisas."),
        ]
    }

    # Current language
    current_language = st.session_state.get("language", "EN")
    
    # Show FAQ with expanders
    for question, answer in faq[current_language]:
        with st.expander(question, expanded=False):
            st.write(answer)


if __name__ == "__main__":
    init_db()
    main()
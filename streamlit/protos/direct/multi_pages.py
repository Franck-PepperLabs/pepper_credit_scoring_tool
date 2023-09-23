import streamlit as st

# Fonction pour afficher la page d'accueil
def show_homepage():
    st.title("Accueil")
    st.write("Bienvenue sur la page d'accueil !")

# Fonction pour afficher la page 1
def show_page1():
    st.title("Page 1")
    st.write("Ceci est le contenu de la page 1.")

# Fonction pour afficher la page 2
def show_page2():
    st.title("Page 2")
    st.write("Ceci est le contenu de la page 2.")

# Barre de navigation latérale pour choisir la page
page = st.sidebar.selectbox("Choisir une page", ["Accueil", "Page 1", "Page 2"])

# Afficher la page sélectionnée
if page == "Accueil":
    show_homepage()
elif page == "Page 1":
    show_page1()
elif page == "Page 2":
    show_page2()

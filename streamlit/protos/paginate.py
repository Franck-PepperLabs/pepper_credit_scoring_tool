import streamlit as st
import pandas as pd

# Données de l'exemple
data = pd.DataFrame({'A': range(1, 101), 'B': range(101, 201)})

# Paramètres de pagination
items_per_page = 10  # Nombre d'éléments par page
current_page = st.sidebar.number_input(
    "Page", 1, len(data) // items_per_page + 1, 1
)

# Calcul des indices de début et de fin de la page actuelle
start_idx = (current_page - 1) * items_per_page
end_idx = min(start_idx + items_per_page, len(data))

# Affichage de la table pour la page actuelle
st.dataframe(data.iloc[start_idx:end_idx])

# Contrôles de pagination
if current_page > 1:
    if st.button("<<"):
        current_page = 1
    if st.button("<"):
        current_page -= 1

if current_page < len(data) / items_per_page:
    if st.button(">"):
        current_page += 1
    if st.button(">>"):
        current_page = len(data) // items_per_page + 1

# Mise à jour de la page actuelle
st.sidebar.write(f"Page {current_page}/{len(data) // items_per_page + 1}")

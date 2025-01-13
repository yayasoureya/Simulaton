import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import shap
import matplotlib.pyplot as plt
import streamlit as st
import matplotlib

matplotlib.use('Agg')  # Utiliser le backend non interactif

# Fonction pour afficher les graphiques SHAP dans Streamlit
def show_shap_plot(plot_function, *args, **kwargs):
    fig, ax = plt.subplots()
    plot_function(*args, show=False, **kwargs)  # Générer le graphique SHAP
    st.pyplot(fig)  # Afficher le graphique dans Streamlit

# 1. Simulation des données
np.random.seed(42)
n_samples = 1000
n_features = 7  # Capteurs : température, mouvement, luminosité, etc.

# Générer les données capteurs
X = np.random.rand(n_samples, n_features) * 10  # Capteurs avec valeurs entre 0 et 10
y = np.random.choice(['Lumière ON', 'Lumière OFF', 'Ouvrir Porte', 'Rien', 'Allumer Clim', 'Éteindre Clim'], size=n_samples)

# Convertir en DataFrame avec de nouveaux capteurs
columns = ['Capteur_Température', 'Capteur_Mouvement', 'Capteur_Luminosité', 'Capteur_Humidité', 'Capteur_Son', 'Capteur_Présence_Salon', 'Capteur_Présence_Chambre']
df = pd.DataFrame(X, columns=columns)
df['Action'] = y

# Ajout d'une estimation de la présence dans chaque pièce (Salon vs Chambre)
df['Time_in_Salon'] = np.random.rand(n_samples) * 10  # Temps estimé passé dans le salon (en minutes)
df['Time_in_Chambre'] = np.random.rand(n_samples) * 10  # Temps estimé passé dans la chambre (en minutes)

# Diviser les données en train/test
X_train, X_test, y_train, y_test = train_test_split(df[columns + ['Time_in_Salon', 'Time_in_Chambre']], df['Action'], test_size=0.2, random_state=0)

# 2. Entraîner un modèle
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 3. Prédire sur les nouvelles données
predictions = model.predict(X_test)

# Affichage des prédictions
st.write(f"Prédictions : {predictions[:5]}")

# 4. Interprétation avec SHAP
explainer = shap.TreeExplainer(model)
#X_test_limited = X_test.iloc[:, :6]
shap_values = explainer.shap_values(X_test)

# Visualisation globale : importance des caractéristiques
st.write("### Importance des caractéristiques")
# Vérifier le nombre de classes dans shap_values
st.write(f"Nombre de classes SHAP : {len(shap_values)}")

# Vérifier la cohérence entre columns et X_test
# Inclure les colonnes supplémentaires dans la vérification
expected_columns = columns + ['Time_in_Salon', 'Time_in_Chambre']
assert len(expected_columns) == X_test.shape[1], "Les noms de colonnes et les données ne correspondent pas."


# Correction du plot SHAP (choisir une classe existante, par exemple 0)
# Visualisation SHAP : inclure les colonnes supplémentaires
shap.summary_plot(shap_values[1], X_test[expected_columns], feature_names=expected_columns)
show_shap_plot(shap.summary_plot, shap_values[1], X_test[expected_columns], feature_names=expected_columns)


# Visualisation locale : explication pour un échantillon spécifique
# Vérifiez les dimensions de shap_values et X_test
assert shap_values[0].shape[1] == len(expected_columns), "Incohérence entre SHAP values et les colonnes."

# Gestion de la taille des SHAP values
if shap_values[0].shape[1] != len(expected_columns):
    st.warning("Les SHAP values générées ne correspondent pas au nombre attendu de colonnes.")
    explanation = shap.Explanation(
        shap_values[0][0][:len(expected_columns)],
        base_values=explainer.expected_value[0],
        data=X_test.iloc[0].values[:len(expected_columns)],
        feature_names=expected_columns
    )
else:
    explanation = shap.Explanation(
        shap_values[0][0],
        base_values=explainer.expected_value[0],
        data=X_test.iloc[0].values,
        feature_names=expected_columns
    )

st.write("### Explication locale pour un échantillon spécifique")
# Convertir base_values en float, si nécessaire
if isinstance(explanation.base_values, pd.Series):
    explanation.base_values = float(explanation.base_values.iloc[0])
    
shap.plots.waterfall(explanation)
show_shap_plot(shap.plots.waterfall, explanation)

# 5. Diagramme en courbes (variations des capteurs)
st.write("### Variation des capteurs au fil du temps")
plt.figure(figsize=(10, 6))
for i, column in enumerate(columns):
    plt.plot(df.index[:100], df[column][:100], label=column)
plt.title('Variation des capteurs au fil du temps')
plt.xlabel('Index')
plt.ylabel('Valeur des capteurs')
plt.legend(loc='upper right')
plt.grid(True)
st.pyplot(plt)

# 6. Diagramme circulaire : Répartition des valeurs des capteurs
st.write("### Répartition des valeurs moyennes des capteurs")
plt.figure(figsize=(7, 7))
sensor_means = df[columns].mean()
plt.pie(sensor_means, labels=sensor_means.index, autopct='%1.1f%%', startangle=90)
plt.title('Répartition des valeurs moyennes des capteurs')
plt.axis('equal')  # Pour un cercle parfait
st.pyplot(plt)

# 7. Diagramme des prédictions futures
st.write("### Prédictions futures au fil du temps")
future_predictions = model.predict(X_test[:100])  # Prédictions sur les 100 premiers échantillons
plt.figure(figsize=(10, 6))
plt.plot(np.arange(100), future_predictions, marker='o', linestyle='-', color='b', label='Prédictions')
plt.title('Prédictions futures au fil du temps')
plt.xlabel('Index')
plt.ylabel('Prédiction')
plt.legend(loc='upper left')
plt.grid(True)
st.pyplot(plt)

# 8. Résumé psychologique de l'utilisateur
def generate_psychological_profile(df):
    profile = []
    
    if df['Capteur_Température'].mean() > 20:
        profile.append("Utilisateur préfère des températures inférieures à 20°C.")
    else:
        profile.append("Utilisateur préfère des températures supérieures à 20°C.")
    
    if df['Capteur_Présence_Salon'].mean() < df['Capteur_Présence_Chambre'].mean():
        profile.append("Utilisateur préfère passer du temps dans la chambre.")
    else:
        profile.append("Utilisateur préfère passer du temps dans le salon.")
    
    if df['Capteur_Luminosité'].mean() < 5:
        profile.append("Utilisateur préfère rester dans l'obscurité plutôt que dans la lumière.")
    else:
        profile.append("Utilisateur préfère des environnements lumineux.")
    
    return " ".join(profile)

profile = generate_psychological_profile(df)
st.write("### Profil psychologique de l'utilisateur")
st.write(profile)

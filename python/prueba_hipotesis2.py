import numpy as np
from scipy import stats

# Importancias extraídas del Random Forest (resultados_avanzados.qmd)
importancias = {
    'Copas Iniciales': 0.1899,
    'Elíxir Promedio': 0.1536,
    'Nivel de Cartas': 0.1374,
    'Épicas': 0.0943,
    'Especiales (Raras)': 0.0897,
    'Comunes': 0.0875,
    'Legendarias': 0.0837,
    'Tropas': 0.0626,
    'Hechizos': 0.0519,
    'Estructuras': 0.0494,
}

# Grupo A: Nivel de cartas + Copas (factores de habilidad/progresión)
grupo_a = np.array([importancias['Copas Iniciales'], importancias['Nivel de Cartas']])

# Grupo B: Rarezas (comunes, raras, épicas, legendarias)
grupo_b = np.array([importancias['Comunes'], importancias['Especiales (Raras)'],
                     importancias['Épicas'], importancias['Legendarias']])

print("Grupo A (Habilidad/Progresión):", grupo_a)
print("  Media:", np.mean(grupo_a))
print("Grupo B (Rarezas):", grupo_b)
print("  Media:", np.mean(grupo_b))

# Prueba t de Welch (no asume varianzas iguales)
t_stat, p_value = stats.ttest_ind(grupo_a, grupo_b, equal_var=False)

print("\n" + "="*60)
print("PRUEBA t DE WELCH: Habilidad/Progresión vs Rarezas")
print("="*60)
print(f"Media grupo A (Copas+Nivel):  {np.mean(grupo_a):.4f}")
print(f"Media grupo B (Rarezas):       {np.mean(grupo_b):.4f}")
print(f"Diferencia (A - B):           {np.mean(grupo_a) - np.mean(grupo_b):.4f}")
print(f"Estadístico t:                {t_stat:.4f}")
print(f"Valor p (bilateral):          {p_value:.6f}")
print(f"Valor p (unilateral >):       {p_value/2:.6f}")

if p_value/2 < 0.05:
    print("\nRESULTADO: La importancia de Habilidad/Progresión ES significativamente")
    print("mayor que la de Rarezas (p < 0.05, prueba unilateral)")
else:
    print("\nRESULTADO: No hay evidencia suficiente para afirmar que")
    print("Habilidad/Progresión es significativamente mayor que Rarezas")
print("="*60)

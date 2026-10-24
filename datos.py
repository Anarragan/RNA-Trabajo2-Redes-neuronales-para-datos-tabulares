import kagglehub
import pandas as pd
import os
import glob
import random
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Descargar la última versión del dataset
path = kagglehub.dataset_download("ranadeep/credit-risk-dataset")
print("Ruta principal de descarga:", path)

# 2. Buscar dinámicamente cualquier archivo .csv y .xlsx en todas las subcarpetas
archivos_csv = glob.glob(os.path.join(path, '**', '*.csv'), recursive=True)
archivos_xlsx = glob.glob(os.path.join(path, '**', '*.xlsx'), recursive=True)

# 3. Leer los Datos de Préstamos (.csv)
if archivos_csv:
    ruta_csv = archivos_csv[0]
    print(f"\n✅ Leyendo CSV desde: {ruta_csv}")
    df = pd.read_csv(ruta_csv)
    print("--- Datos de Préstamos ---")
    print(df.head())
else:
    print("\n❌ Error: No se encontró ningún archivo .csv en el dataset.")

# 4. Leer el Diccionario de Datos (.xlsx)
if archivos_xlsx:
    ruta_xlsx = archivos_xlsx[0]
    print(f"\n✅ Leyendo Excel desde: {ruta_xlsx}")
    # Leemos específicamente la hoja 'LoanStats'
    df_diccionario = pd.read_excel(ruta_xlsx, sheet_name="LoanStats") 
    print("--- Diccionario de Datos ---")
    print(df_diccionario.head())
else:
    print("\n❌ Error: No se encontró ningún archivo .xlsx en el dataset.")

    # 1. Definimos el diccionario basándonos en tu tabla de reglas
mapeo_riesgo = {
    "Fully Paid": 0,
    "Does not meet the credit policy. Status:Fully Paid": 0,
    "Charged Off": 1,
    "Default": 1,
    "Late (31-120 days)": 1,
    "Does not meet the credit policy. Status:Charged Off": 1
}

# 2. Aplicamos la vectorización. 
# Lo que no esté en el diccionario (Current, Issued, etc.) se vuelve NaN automáticamente.
df["loan_status"] = df["loan_status"].map(mapeo_riesgo)

# 3. Eliminamos de un solo golpe todas las filas que quedaron con NaN en esta columna
df = df.dropna(subset=["loan_status"])

# 1. Filtramos solo las columnas que contienen texto (tipo 'object')
columnas_categoricas = df.select_dtypes(include=['object']).columns

# 2. Imprimimos el nombre de la columna y cuántos valores de texto diferentes tiene
for col in columnas_categoricas:
    print(f"Columna: '{col}' | Valores únicos: {df[col].nunique()}")


    # 1. Definimos las columnas que son ruido, texto libre o tienen varianza cero
columnas_a_eliminar = [
    'emp_title', 'url', 'desc', 'title', 'zip_code', 
    'verification_status_joint', 'pymnt_plan', 
    'initial_list_status', 'application_type',
    'earliest_cr_line', 'last_pymnt_d', 'next_pymnt_d', 'last_credit_pull_d', 'issue_d' # Fechas complejas que requieren otro tratamiento
]

# Las borramos del DataFrame
df = df.drop(columns=columnas_a_eliminar, errors='ignore')

# 2. Identificamos las columnas de texto que sobrevivieron al filtro
columnas_categoricas_utiles = df.select_dtypes(include=['object']).columns

# 3. Aplicamos One-Hot Encoding (pd.get_dummies)
# drop_first=True elimina la primera categoría de cada variable para evitar multicolinealidad
df = pd.get_dummies(df, columns=columnas_categoricas_utiles, drop_first=True)

# 4. Verificamos las nuevas dimensiones de la matriz de entrada
print(f"La capa de entrada de tu red tendrá {df.shape[1] - 1} neuronas (características).")

# 1. Separar la matriz de características (X) y el vector objetivo (y)
X = df.drop(columns=['loan_status'])
y = df['loan_status']

# 1. Primera división: Separamos el Test (15%) y guardamos el resto (85%) para Train/Val
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)

# 2. Segunda división: Del 85% restante, separamos un 15% (del total original) para Validación
# Matemáticamente, 0.15 / 0.85 = 0.1764 (aprox 17.65% del X_temp)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.1765, random_state=42, stratify=y_temp
)

# 3. Escalar matemáticamente (El fit SOLO se hace en Train)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)      # Solo transform!
X_test_scaled = scaler.transform(X_test)    # Solo transform!

# 4. Convertir a Tensores de PyTorch
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)

X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).view(-1, 1)

X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

#Convertir los valores nulos a 0.0 para evitar problemas durante el entrenamiento

X_train_tensor = torch.nan_to_num(X_train_tensor, nan=0.0)
X_val_tensor = torch.nan_to_num(X_val_tensor, nan=0.0)
X_test_tensor = torch.nan_to_num(X_test_tensor, nan=0.0)


print(f"Entrenamiento: {X_train_tensor.shape[0]} muestras")
print(f"Validación: {X_val_tensor.shape[0]} muestras")
print(f"Prueba (Oculto): {X_test_tensor.shape[0]} muestras")


# 1. Creamos una arquitectura dinámica que acepta cualquier cantidad de capas
class PerceptronDinamico(nn.Module):
    def __init__(self, num_caracteristicas, capas_ocultas, p_dropout):
        super(PerceptronDinamico, self).__init__()
        
        capas = []
        entrada_actual = num_caracteristicas
        
        # El bucle crea las capas ocultas y de regularización que le pasemos en la lista
        for neuronas in capas_ocultas:
            capas.append(nn.Linear(entrada_actual, neuronas))
            capas.append(nn.ReLU())
            capas.append(nn.Dropout(p=p_dropout))
            entrada_actual = neuronas
            
        # Agregamos la capa de salida binaria
        capas.append(nn.Linear(entrada_actual, 1))
        capas.append(nn.Sigmoid())
        
        # nn.Sequential empaqueta todas las capas en orden
        self.red = nn.Sequential(*capas)

    def forward(self, x):
        return self.red(x)

# 2. Definimos nuestro "universo" de hiperparámetros a explorar
espacio_busqueda = {
    'arquitectura': [[64, 32], [128, 64, 32], [32, 16], [128, 64]], # Desde 2 hasta 3 capas
    'dropout': [0.1, 0.2, 0.3, 0.4],                                # Porcentaje de neuronas a apagar
    'lr': [0.01, 0.001, 0.0005],                                    # Tasa de aprendizaje
    'l2_penalty': [0.0, 1e-4, 1e-5]                                 # Regularización L2
}

# 3. Lógica de Random Search
intentos = 10 # Evaluaremos 10 configuraciones aleatorias para no saturar Colab
mejor_configuracion = None
mejor_loss_global = float('inf')
mejores_pesos_globales = None

print(f"Iniciando Random Search ({intentos} combinaciones)...\n")

for i in range(intentos):
    # A. Seleccionar una combinación de hiperparámetros al azar
    config = {
        'arquitectura': random.choice(espacio_busqueda['arquitectura']),
        'dropout': random.choice(espacio_busqueda['dropout']),
        'lr': random.choice(espacio_busqueda['lr']),
        'l2_penalty': random.choice(espacio_busqueda['l2_penalty'])
    }
    
    # B. Construir el modelo dinámico y el motor matemático para este intento
    modelo_prueba = PerceptronDinamico(X_train_tensor.shape[1], config['arquitectura'], config['dropout'])
    
    # Aquí inyectamos la Penalización L2 (weight_decay) directo al optimizador
    optimizador_prueba = optim.Adam(modelo_prueba.parameters(), lr=config['lr'], weight_decay=config['l2_penalty'])
    criterio = nn.BCELoss()
    
    # C. Bucle de Entrenamiento Rápido con Parada Temprana
    mejor_loss_val = float('inf')
    paciencia, contador = 10, 0
    mejores_pesos_intento = None
    
    for epoca in range(250): # Dejamos correr hasta 250 épocas por intento
        # Entrenar
        modelo_prueba.train()
        optimizador_prueba.zero_grad()
        perdida_train = criterio(modelo_prueba(X_train_tensor), y_train_tensor)
        perdida_train.backward()
        optimizador_prueba.step()
        
        # Validar
        modelo_prueba.eval()
        with torch.no_grad():
            perdida_val = criterio(modelo_prueba(X_val_tensor), y_val_tensor).item()
            
        # Lógica Early Stopping
        if perdida_val < mejor_loss_val:
            mejor_loss_val = perdida_val
            contador = 0
            import copy
            mejores_pesos_intento = copy.deepcopy(modelo_prueba.state_dict())
        else:
            contador += 1
            if contador >= paciencia:
                break # Abortar este intento si deja de aprender
                
    print(f"Intento {i+1:2d}/10 | Mejor Error Val: {mejor_loss_val:.4f} | Config: {config}")
    
    # D. ¿Es esta la mejor configuración de todos los intentos?
    if mejor_loss_val < mejor_loss_global:
        mejor_loss_global = mejor_loss_val
        mejor_configuracion = config
        mejores_pesos_globales = mejores_pesos_intento

print("\n🚀 Búsqueda finalizada.")
print(f"Mejor configuración encontrada: {mejor_configuracion}")
print(f"Con un error matemático mínimo de: {mejor_loss_global:.4f}")

# Cargar en memoria el mejor modelo absoluto encontrado
modelo_final = PerceptronDinamico(X_train_tensor.shape[1], mejor_configuracion['arquitectura'], mejor_configuracion['dropout'])
modelo_final.load_state_dict(mejores_pesos_globales)

from sklearn.metrics import classification_report, confusion_matrix

print("Evaluando el modelo final en los datos ocultos de Prueba (Test Set)...")

# 1. Ponemos el modelo ganador en modo evaluación
modelo_final.eval()

# 2. Apagamos el motor de gradientes (ya no aprende, solo predice)
with torch.no_grad():
    # Pasamos los datos de prueba (que ya habíamos mudado a la GPU)
    predicciones_test = modelo_final(X_test_tensor)
    
    # Transformamos las probabilidades en 0 o 1
    clases_predichas = (predicciones_test >= 0.5).float()

# 3. Traemos las predicciones de vuelta a la CPU para poder usar Scikit-Learn
clases_predichas_cpu = clases_predichas.cpu().numpy()
y_test_cpu = y_test_tensor.cpu().numpy()

# 4. Calculamos la precisión total
aciertos = (clases_predichas_cpu == y_test_cpu).sum()
total = len(y_test_cpu)
precision = (aciertos / total) * 100

print(f"\n¡Precisión final del modelo: {precision:.2f}%!\n")

# 5. Reporte detallado
print("Reporte de Clasificación:")
print(classification_report(y_test_cpu, clases_predichas_cpu, target_names=["Buen Pagador (0)", "Mal Pagador (1)"]))

print("\nMatriz de Confusión:")
matriz = confusion_matrix(y_test_cpu, clases_predichas_cpu)
print(matriz)



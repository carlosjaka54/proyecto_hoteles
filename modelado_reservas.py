"""
modelado_reservas.py
====================
Pipeline de machine learning: predecir si una reserva de hotel será CANCELADA.

Entrada : PostgreSQL, tabla "hotel_analitica"  (o datos/hotel_analitica.csv con --fuente csv)
Salida  : PostgreSQL, tabla "predicciones_2017" (probabilidad y decisión por reserva)
Modelo  : Regresión logística (solver lbfgs)

Pasos del pipeline:
  1. Auditar y tratar nulos (imputación con la mediana del conjunto de entrenamiento).
  2. Separar X y target + split CRONOLÓGICO:
       - Entrenamiento: llegadas de jul-2015 a dic-2016
       - Prueba (test): llegadas de ene-2017 a ago-2017
  3. Escalar variables con StandardScaler (ajustado SOLO con entrenamiento).
  4. Entrenar regresión logística (solver='lbfgs').
  5. Evaluar: reporte de clasificación, matriz de confusión, ROC AUC y
     comparación contra un baseline que siempre predice "no cancela".
  6. Función tipada de inferencia: si P(cancelar) >= 60% -> sobrevender la habitación.

Ejecutar:  python modelado_reservas.py              (lee y guarda en PostgreSQL)
           python modelado_reservas.py --fuente csv (solo CSV, sin base de datos)
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

RUTA_DATOS = Path("datos/hotel_analitica.csv")
FECHA_CORTE = pd.Timestamp("2017-01-01")  # antes -> entrenamiento, desde aquí -> prueba
UMBRAL_DECISION = 0.60
TARGET = "cancelada"

VARIABLES_X: list[str] = [
    "dias_anticipacion",
    "noches_totales",
    "adultos",
    "ninos",
    "tarifa_diaria",
    "cliente_repetido",
    "cancelaciones_previas",
    "reservas_previas_no_canceladas",
    "dias_lista_espera",
    "parqueaderos_solicitados",
    "solicitudes_especiales",
    "es_city_hotel",
    "deposito_no_reembolsable",
    "canal_agencia_online",
    "cliente_transitorio",
]


# --------------------------------------------------------------------------- #
# 0. Carga
# --------------------------------------------------------------------------- #
def cargar_datos(fuente: str) -> pd.DataFrame:
    if fuente == "postgres":
        from conexion_db import obtener_engine

        df = pd.read_sql("SELECT * FROM hotel_analitica ORDER BY fecha_llegada, id_reserva",
                         obtener_engine(), parse_dates=["fecha_llegada"])
        origen = "PostgreSQL (tabla 'hotel_analitica')"
    else:
        df = pd.read_csv(RUTA_DATOS, parse_dates=["fecha_llegada"])
        origen = str(RUTA_DATOS)
    print(f"[CARGA] {len(df):,} reservas leídas de {origen}")
    return df


# --------------------------------------------------------------------------- #
# 1 y 2. Split cronológico + auditoría/imputación de nulos
# --------------------------------------------------------------------------- #
def split_cronologico(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Separa X / y y divide por fecha (no aleatorio) para evitar fuga temporal."""
    entrenamiento = df[df["fecha_llegada"] < FECHA_CORTE]
    prueba = df[df["fecha_llegada"] >= FECHA_CORTE]
    print(
        f"\n[SPLIT CRONOLÓGICO]\n"
        f"  Entrenamiento: {entrenamiento['fecha_llegada'].min():%Y-%m-%d} a "
        f"{entrenamiento['fecha_llegada'].max():%Y-%m-%d} -> {len(entrenamiento):,} reservas\n"
        f"  Prueba       : {prueba['fecha_llegada'].min():%Y-%m-%d} a "
        f"{prueba['fecha_llegada'].max():%Y-%m-%d} -> {len(prueba):,} reservas"
    )
    return (
        entrenamiento[VARIABLES_X].copy(),
        prueba[VARIABLES_X].copy(),
        entrenamiento[TARGET],
        prueba[TARGET],
    )


def imputar_nulos(
    X_train: pd.DataFrame, X_test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Imputa nulos con la mediana calculada SOLO sobre entrenamiento."""
    print("\n[AUDITORÍA DE NULOS]")
    nulos_train = X_train.isna().sum()
    nulos_test = X_test.isna().sum()
    medianas = X_train.median()
    for col in VARIABLES_X:
        if nulos_train[col] or nulos_test[col]:
            print(
                f"  {col}: {nulos_train[col]} nulos en entrenamiento, {nulos_test[col]} en prueba "
                f"-> imputados con la mediana de entrenamiento = {medianas[col]}"
            )
    X_train = X_train.fillna(medianas)
    X_test = X_test.fillna(medianas)
    print(f"  Nulos restantes: {int(X_train.isna().sum().sum() + X_test.isna().sum().sum())}")
    return X_train, X_test, medianas


# --------------------------------------------------------------------------- #
# 3 y 4. Escalado + entrenamiento
# --------------------------------------------------------------------------- #
def escalar_y_entrenar(
    X_train: pd.DataFrame, y_train: pd.Series
) -> tuple[StandardScaler, LogisticRegression]:
    escalador = StandardScaler().fit(X_train)  # ajustado SOLO con entrenamiento
    modelo = LogisticRegression(solver="lbfgs", max_iter=1000)
    modelo.fit(escalador.transform(X_train), y_train)

    print("\n[MODELO] Regresión logística entrenada. Coeficientes (variables escaladas):")
    coef = pd.Series(modelo.coef_[0], index=VARIABLES_X).sort_values(key=np.abs, ascending=False)
    for var, valor in coef.items():
        efecto = "aumenta" if valor > 0 else "reduce"
        print(f"  {var:<32} {valor:+.3f}  ({efecto} la probabilidad de cancelar)")
    print(f"  Intercepto: {modelo.intercept_[0]:+.3f}")
    return escalador, modelo


# --------------------------------------------------------------------------- #
# 5. Evaluación y baseline
# --------------------------------------------------------------------------- #
def evaluar(
    modelo: LogisticRegression,
    escalador: StandardScaler,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> np.ndarray:
    proba = modelo.predict_proba(escalador.transform(X_test))[:, 1]
    pred = (proba >= 0.5).astype(int)

    print("\n[EVALUACIÓN EN PRUEBA (ene-ago 2017)]")
    print("Reporte de clasificación (umbral estándar 0.5):")
    print(classification_report(y_test, pred, target_names=["No cancela", "Cancela"], digits=3))

    cm = confusion_matrix(y_test, pred)
    print("Matriz de confusión:")
    print(f"                      Predice NO cancela   Predice cancela")
    print(f"  Real NO cancela     {cm[0, 0]:>18,}   {cm[0, 1]:>15,}")
    print(f"  Real cancela        {cm[1, 0]:>18,}   {cm[1, 1]:>15,}")

    auc = roc_auc_score(y_test, proba)
    print(f"\nROC AUC: {auc:.3f}  (0.5 = adivinar al azar, 1.0 = perfecto)")

    # Baseline: siempre predecir la clase mayoritaria del entrenamiento ("no cancela")
    acc_modelo = accuracy_score(y_test, pred)
    acc_baseline = accuracy_score(y_test, np.zeros_like(y_test))
    print("\n[COMPARACIÓN CONTRA BASELINE]")
    print(f"  Baseline (siempre 'no cancela'): exactitud = {acc_baseline:.1%}, ROC AUC = 0.500")
    print(f"  Regresión logística           : exactitud = {acc_modelo:.1%}, ROC AUC = {auc:.3f}")
    print(f"  Mejora sobre el baseline       : {acc_modelo - acc_baseline:+.1%} puntos de exactitud")

    decide = proba >= UMBRAL_DECISION
    print(f"\n[UMBRAL DE DECISIÓN {UMBRAL_DECISION:.0%}]")
    print(f"  Reservas marcadas para sobreventa: {decide.sum():,} de {len(decide):,}")
    print(f"  De esas, realmente se cancelaron: {y_test[decide].mean():.1%} (precisión)")
    return proba


def guardar_predicciones(df_prueba: pd.DataFrame, proba: np.ndarray) -> None:
    """Guarda en PostgreSQL la probabilidad y la decisión de cada reserva de prueba."""
    from conexion_db import obtener_engine

    resultado = pd.DataFrame(
        {
            "id_reserva": df_prueba["id_reserva"].to_numpy(),
            "fecha_llegada": df_prueba["fecha_llegada"].dt.date.to_numpy(),
            "hotel": df_prueba["hotel"].to_numpy(),
            "cancelada_real": df_prueba[TARGET].to_numpy(),
            "probabilidad_cancelacion": proba.round(4),
            "decision_sobrevender": proba >= UMBRAL_DECISION,
        }
    )
    resultado.to_sql("predicciones_2017", obtener_engine(), if_exists="replace", index=False,
                     chunksize=5000, method="multi")
    print(f"\n[GUARDADO] Tabla 'predicciones_2017' creada en PostgreSQL ({len(resultado):,} filas)")


# --------------------------------------------------------------------------- #
# 6. Función tipada de inferencia
# --------------------------------------------------------------------------- #
@dataclass
class Reserva:
    dias_anticipacion: int
    noches_totales: int
    adultos: int
    ninos: float
    tarifa_diaria: float
    cliente_repetido: int
    cancelaciones_previas: int
    reservas_previas_no_canceladas: int
    dias_lista_espera: int
    parqueaderos_solicitados: int
    solicitudes_especiales: int
    es_city_hotel: int
    deposito_no_reembolsable: int
    canal_agencia_online: int
    cliente_transitorio: int


@dataclass
class Decision:
    probabilidad_cancelacion: float
    sobrevender: bool
    recomendacion: str


def predecir_reserva(
    reserva: Reserva,
    modelo: LogisticRegression,
    escalador: StandardScaler,
    umbral: float = UMBRAL_DECISION,
) -> Decision:
    """Devuelve la probabilidad de cancelación y la decisión con el umbral dado."""
    x = pd.DataFrame([vars(reserva)])[VARIABLES_X]
    p = float(modelo.predict_proba(escalador.transform(x))[0, 1])
    sobrevender = p >= umbral
    texto = (
        "Alta probabilidad de cancelación: se puede sobrevender esta habitación."
        if sobrevender
        else "No supera el umbral: mantener la habitación reservada."
    )
    return Decision(round(p, 3), sobrevender, texto)


def demo_inferencia(modelo: LogisticRegression, escalador: StandardScaler) -> None:
    casos: dict[str, Reserva] = {
        "Caso A - reserva riesgosa (agencia online, 9 meses de anticipación, depósito no reembolsable)": Reserva(
            270, 3, 2, 0, 95.0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1
        ),
        "Caso B - caso intermedio (City Hotel, 2 meses de anticipación, sin solicitudes)": Reserva(
            60, 3, 2, 0, 110.0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1
        ),
        "Caso C - reserva segura (cliente repetido, parqueadero, 2 solicitudes especiales)": Reserva(
            10, 2, 2, 1, 120.0, 1, 0, 3, 0, 1, 2, 0, 0, 0, 1
        ),
    }
    print(f"\n[FUNCIÓN DE INFERENCIA - umbral {UMBRAL_DECISION:.0%}]")
    for nombre, reserva in casos.items():
        d = predecir_reserva(reserva, modelo, escalador)
        print(f"  {nombre}")
        print(f"    P(cancelar) = {d.probabilidad_cancelacion:.1%} -> sobrevender = {d.sobrevender}. {d.recomendacion}")


# --------------------------------------------------------------------------- #
# Pipeline principal
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="Modelo de cancelación de reservas")
    parser.add_argument("--fuente", choices=["postgres", "csv"], default="postgres")
    fuente = parser.parse_args().fuente

    df = cargar_datos(fuente)
    X_train, X_test, y_train, y_test = split_cronologico(df)
    X_train, X_test, _ = imputar_nulos(X_train, X_test)
    escalador, modelo = escalar_y_entrenar(X_train, y_train)
    proba = evaluar(modelo, escalador, X_test, y_test)
    demo_inferencia(modelo, escalador)
    if fuente == "postgres":
        guardar_predicciones(df[df["fecha_llegada"] >= FECHA_CORTE], proba)


if __name__ == "__main__":
    main()

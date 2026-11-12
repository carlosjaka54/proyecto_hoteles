"""
etl_auditoria.py
================
Fase ETL + auditoría del dataset "Hotel Booking Demand" (Kaggle).

Entrada : datos/hotel_bookings.csv   (119.390 reservas, 32 columnas, jul-2015 a ago-2017)
Salida  : datos/hotel_analitica.csv  (matriz analítica plana lista para el modelo)

Pasos:
  1. Extraer  -> leer el CSV original.
  2. Auditar  -> nulos, duplicados, valores fuera de rango.
  3. Transformar -> limpiar anomalías, quitar columnas con fuga de datos,
                    crear variables nuevas y la fecha de llegada.
  4. Cargar   -> guardar la matriz analítica ordenada cronológicamente.

Ejecutar:  python etl_auditoria.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

RUTA_ENTRADA = Path("datos/hotel_bookings.csv")
RUTA_SALIDA = Path("datos/hotel_analitica.csv")

MESES = {
    "January": 1, "February": 2, "March": 3, "April": 4, "May": 5, "June": 6,
    "July": 7, "August": 8, "September": 9, "October": 10, "November": 11, "December": 12,
}

# Columnas que se conocen DESPUÉS de saber si la reserva se canceló (o que cambian
# hasta el día de llegada). Si se dejan, el modelo "hace trampa" (fuga de datos).
COLUMNAS_CON_FUGA = [
    "reservation_status",       # dice literalmente "Canceled" / "Check-Out"
    "reservation_status_date",  # fecha en que se canceló o salió
    "assigned_room_type",       # se asigna al llegar al hotel
    "booking_changes",          # cambia hasta el día de llegada
]


def extraer(ruta: Path) -> pd.DataFrame:
    """Lee el CSV original de Kaggle."""
    df = pd.read_csv(ruta)
    print(f"[EXTRACCIÓN] {len(df):,} filas x {df.shape[1]} columnas leídas de {ruta}")
    return df


def auditar(df: pd.DataFrame) -> None:
    """Imprime un reporte de calidad de datos (no modifica nada)."""
    print("\n[AUDITORÍA]")
    nulos = df.isna().sum()
    nulos = nulos[nulos > 0]
    print("  Nulos por columna:")
    for col, n in nulos.items():
        print(f"    - {col:<10} {n:>7,}  ({n / len(df):.1%})")

    # Nota: el dataset no tiene ID de reserva, así que reservas distintas con los mismos
    # datos (p. ej. grupos) aparecen "duplicadas". Por eso NO se eliminan.
    print(f"  Filas duplicadas exactas : {df.duplicated().sum():,}  (reservas de grupo sin ID; se conservan)")
    huespedes = df["adults"] + df["children"].fillna(0) + df["babies"]
    print(f"  Reservas con 0 huéspedes : {(huespedes == 0).sum():,}  (anomalía)")
    print(f"  Tarifa diaria (adr) < 0  : {(df['adr'] < 0).sum():,}  (anomalía)")
    print(f"  Tarifa diaria (adr) > 1000: {(df['adr'] > 1000).sum():,}  (valor atípico, máx = {df['adr'].max():,.0f})")
    print(f"  Tasa de cancelación global: {df['is_canceled'].mean():.1%}")


def transformar(df: pd.DataFrame) -> pd.DataFrame:
    """Limpia anomalías y construye la matriz analítica plana."""
    print("\n[TRANSFORMACIÓN]")
    n0 = len(df)

    # 1) Eliminar anomalías detectadas en la auditoría
    huespedes = df["adults"] + df["children"].fillna(0) + df["babies"]
    df = df[(huespedes > 0) & (df["adr"] >= 0) & (df["adr"] <= 1000)].copy()
    print(f"  Filas eliminadas por anomalías: {n0 - len(df):,}")

    # 2) Fecha de llegada (necesaria para el split cronológico)
    df["fecha_llegada"] = pd.to_datetime(
        dict(
            year=df["arrival_date_year"],
            month=df["arrival_date_month"].map(MESES),
            day=df["arrival_date_day_of_month"],
        )
    )

    # 3) Quitar columnas con fuga de datos
    df = df.drop(columns=COLUMNAS_CON_FUGA)
    print(f"  Columnas eliminadas por fuga de datos: {COLUMNAS_CON_FUGA}")

    # 4) Variables nuevas (todas numéricas, para la regresión logística)
    analitica = pd.DataFrame(
        {
            "fecha_llegada": df["fecha_llegada"].dt.date,
            "hotel": df["hotel"],
            "dias_anticipacion": df["lead_time"],
            "noches_totales": df["stays_in_weekend_nights"] + df["stays_in_week_nights"],
            "adultos": df["adults"],
            "ninos": df["children"],  # se dejan sus 4 nulos: el pipeline los imputa con la mediana
            "tarifa_diaria": df["adr"].round(2),
            "cliente_repetido": df["is_repeated_guest"],
            "cancelaciones_previas": df["previous_cancellations"],
            "reservas_previas_no_canceladas": df["previous_bookings_not_canceled"],
            "dias_lista_espera": df["days_in_waiting_list"],
            "parqueaderos_solicitados": df["required_car_parking_spaces"],
            "solicitudes_especiales": df["total_of_special_requests"],
            "es_city_hotel": (df["hotel"] == "City Hotel").astype(int),
            "deposito_no_reembolsable": (df["deposit_type"] == "Non Refund").astype(int),
            "canal_agencia_online": (df["market_segment"] == "Online TA").astype(int),
            "cliente_transitorio": (df["customer_type"] == "Transient").astype(int),
            # Variable predictiva (target binario): 1 = canceló, 0 = no canceló
            "cancelada": df["is_canceled"],
        }
    )
    analitica = analitica.sort_values("fecha_llegada").reset_index(drop=True)
    print(f"  Matriz analítica: {len(analitica):,} filas x {analitica.shape[1]} columnas")
    return analitica


def cargar(df: pd.DataFrame, ruta: Path) -> None:
    """Guarda la matriz analítica plana."""
    df.to_csv(ruta, index=False)
    print(f"\n[CARGA] Archivo plano guardado en {ruta}")


def main() -> None:
    df = extraer(RUTA_ENTRADA)
    auditar(df)
    analitica = transformar(df)
    cargar(analitica, RUTA_SALIDA)


if __name__ == "__main__":
    main()

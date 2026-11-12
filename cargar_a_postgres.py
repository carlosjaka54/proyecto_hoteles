"""
cargar_a_postgres.py
====================
Carga el CSV original de Kaggle en PostgreSQL, en la tabla "reservas".
Esta tabla hace el papel del almacén transaccional (como el SQLite del profesor).

Se ejecuta UNA sola vez (si lo vuelves a correr, reemplaza la tabla).

Ejecutar:  python cargar_a_postgres.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sqlalchemy import text

from conexion_db import obtener_engine

RUTA_CSV = Path("datos/hotel_bookings.csv")
TABLA = "reservas"


def main() -> None:
    df = pd.read_csv(RUTA_CSV)
    print(f"Leídas {len(df):,} reservas de {RUTA_CSV}")

    engine = obtener_engine()
    df.index.name = "id_reserva"  # se agrega un ID, porque el CSV no trae uno
    df.to_sql(TABLA, engine, if_exists="replace", index=True, chunksize=5000, method="multi")

    with engine.begin() as conn:
        conn.execute(text(f"ALTER TABLE {TABLA} ADD PRIMARY KEY (id_reserva)"))
        total = conn.execute(text(f"SELECT COUNT(*) FROM {TABLA}")).scalar()
    print(f"Tabla '{TABLA}' creada en PostgreSQL con {total:,} filas.")


if __name__ == "__main__":
    main()

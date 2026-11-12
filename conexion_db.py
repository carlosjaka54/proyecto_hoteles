"""
conexion_db.py
==============
Conexión a PostgreSQL (la misma base de datos que ves en pgAdmin).

Los datos de conexión se leen del archivo .env (que NO se sube a GitHub).
Copia .env.example como .env y pon tu contraseña de PostgreSQL.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from sqlalchemy import Engine, create_engine, text
from sqlalchemy.engine import URL

load_dotenv()


def obtener_engine() -> Engine:
    """Crea la conexión a PostgreSQL con los datos del archivo .env."""
    url = URL.create(
        drivername="postgresql+psycopg2",
        username=os.getenv("DB_USUARIO", "postgres"),
        password=os.getenv("DB_CLAVE"),
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PUERTO", "5432")),
        database=os.getenv("DB_NOMBRE", "hoteles"),
    )
    return create_engine(url)


def probar_conexion() -> None:
    """Verifica que la base de datos responde."""
    engine = obtener_engine()
    with engine.connect() as conn:
        version = conn.execute(text("SELECT version()")).scalar()
    print(f"Conexión exitosa a '{engine.url.database}' en {engine.url.host}:{engine.url.port}")
    print(f"  {version}")


if __name__ == "__main__":
    probar_conexion()

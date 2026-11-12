-- =====================================================================
-- Consultas para revisar el proyecto en pgAdmin (base de datos: hoteles)
-- Abre: hoteles > clic derecho > Query Tool, pega una consulta y presiona F5
-- =====================================================================

-- 1. Tabla original (almacén transaccional): ¿cuántas reservas hay?
SELECT COUNT(*) AS total_reservas FROM reservas;

-- 2. Auditoría: tasa de cancelación por hotel
SELECT hotel,
       COUNT(*) AS reservas,
       ROUND((AVG(is_canceled) * 100)::numeric, 1) AS porcentaje_canceladas
FROM reservas
GROUP BY hotel;

-- 3. Auditoría: anomalías detectadas
SELECT
  SUM(CASE WHEN adults + COALESCE(children, 0) + babies = 0 THEN 1 ELSE 0 END) AS sin_huespedes,
  SUM(CASE WHEN adr < 0 THEN 1 ELSE 0 END)                                    AS tarifa_negativa,
  SUM(CASE WHEN adr > 1000 THEN 1 ELSE 0 END)                                 AS tarifa_atipica,
  SUM(CASE WHEN children IS NULL THEN 1 ELSE 0 END)                           AS ninos_nulos
FROM reservas;

-- 4. Matriz analítica (salida del ETL): primeras filas
SELECT * FROM hotel_analitica ORDER BY fecha_llegada LIMIT 20;

-- 5. Predicciones 2017: reservas con mayor probabilidad de cancelar
SELECT * FROM predicciones_2017
ORDER BY probabilidad_cancelacion DESC
LIMIT 20;

-- 6. ¿Qué tan bien funcionó la decisión de sobrevender (umbral 60%)?
SELECT decision_sobrevender,
       COUNT(*) AS reservas,
       SUM(cancelada_real) AS realmente_canceladas,
       ROUND((AVG(cancelada_real) * 100)::numeric, 1) AS porcentaje_acierto
FROM predicciones_2017
GROUP BY decision_sobrevender;

-- 7. Resultado por mes de 2017
SELECT DATE_TRUNC('month', fecha_llegada)::date AS mes,
       COUNT(*) AS reservas,
       SUM(CASE WHEN decision_sobrevender THEN 1 ELSE 0 END) AS marcadas_sobreventa,
       ROUND((AVG(probabilidad_cancelacion) * 100)::numeric, 1) AS prob_promedio
FROM predicciones_2017
GROUP BY 1
ORDER BY 1;

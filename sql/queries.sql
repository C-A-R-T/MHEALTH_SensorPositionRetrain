-- Step 3: sql/queries.sql
CREATE OR REPLACE TABLE har AS SELECT * FROM 'data/processed/mhealth.parquet';
-- Q1: windows per person and activity (balance of the protocol)
SELECT subject, COUNT(*) AS n, COUNT(DISTINCT activity) AS activities FROM har GROUP BY 1 ORDER BY 1;
-- Q2: which body position moves in which activity, and how the heart signal changes
SELECT activity, COUNT(*) AS n, AVG(chest_accmag_std) AS chest_motion, AVG(ankle_accmag_std) AS ankle_motion,
       AVG(wrist_accmag_std) AS wrist_motion, AVG(ecg_lead1_std) AS ecg_spread
FROM har GROUP BY 1 HAVING COUNT(*) > 100 ORDER BY ankle_motion;
-- Q3: spread between people inside one activity for each body position (the source of person shift), cv = sd / mean
WITH s AS (SELECT activity, subject, AVG(chest_accmag_std) AS chest, AVG(ankle_accmag_std) AS ankle,
                  AVG(wrist_accmag_std) AS wrist FROM har GROUP BY 1, 2)
SELECT activity, AVG(chest) AS chest_mean, STDDEV(chest) / AVG(chest) AS chest_cv, AVG(ankle) AS ankle_mean,
       STDDEV(ankle) / AVG(ankle) AS ankle_cv, AVG(wrist) AS wrist_mean, STDDEV(wrist) / AVG(wrist) AS wrist_cv
FROM s GROUP BY 1 ORDER BY wrist_cv DESC;
-- Q4: model table; the motion trend uses only earlier windows of the same run (LAG, no leakage)
CREATE OR REPLACE TABLE feat AS
SELECT * EXCLUDE (label),
       COALESCE(ankle_accmag_std - LAG(ankle_accmag_std, 10) OVER w, 0) AS ankle_trend,
       COALESCE(wrist_accmag_std - LAG(wrist_accmag_std, 10) OVER w, 0) AS wrist_trend
FROM har WINDOW w AS (PARTITION BY subject, run ORDER BY win);
-- Q5: windows per activity in the model table
SELECT activity, COUNT(*) AS n FROM feat GROUP BY 1 ORDER BY 2 DESC, 1;   -- ties sorted by name

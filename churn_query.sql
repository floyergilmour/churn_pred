WITH base AS(
SELECT
  geography,
  gender,
  CASE WHEN numofproducts IN (3,4) THEN '3-4' ELSE numofproducts::text END numofproducts,
  CASE WHEN tenure IN (0,1) THEN '0-1'
  WHEN tenure IN (9,10) THEN '9-10'
  ELSE tenure::text
  END as tenure,
  100*ROUND(SUM(CASE WHEN exited = 1 THEN 1 ELSE 0 END)::numeric/COUNT(*),5) AS ratio
FROM churn.raw_data
GROUP BY 1,2,3,4
)
SELECT
  --A.creditscore::double precision as creditscore,
  null AS fold,
  null as rid,
  null AS rf_predict,
  null AS rf_predict_prob,
  null AS gbm_predict,
  null AS gbm_predict_prob,
  null AS glm_predict,
  null AS glm_predict_prob,
  null AS xgboost_predict,
  null AS xgboost_predict_prob,
  null AS kmeans_predict,
  A.age::double precision,
  A.tenure::double precision,
  A.balance::double precision,
  (CASE WHEN A.numofproducts IN (3,4) THEN 3 ELSE A.numofproducts END)::double precision AS numofproducts,
  A.estimatedsalary::double precision,
  B.ratio::double precision,
  A.hascrcard,--CASE WHEN A.hascrcard = 1 THEN True ELSE False END AS hascrcard,
  A.isactivemember,--CASE WHEN A.isactivemember = 1 THEN True ELSE False END AS isactivemember,
  A.geography,  
  A.gender,
  A.exited--CASE WHEN A.exited = 1 THEN True ELSE False END AS exited 
FROM churn.raw_data AS A
  LEFT JOIN base AS B ON
                        A.geography = B.geography
                        AND A.gender = B.gender
                        AND A.numofproducts::text = ANY (string_to_array(B.numofproducts,'-'))
                        AND A.tenure::text = ANY (string_to_array(B.tenure,'-'))
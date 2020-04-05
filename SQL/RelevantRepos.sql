/*Get the list of all repositories that contain at least 8 .java files in
at least 2 packages, that have been starred at least twice and forked at least
once but are not forks themselves.*/

(SELECT original as repo
FROM `YOUR_PROJECT.YOUR_DATASET.forks`

EXCEPT DISTINCT

SELECT fork as repo
FROM `YOUR_PROJECT.YOUR_DATASET.forks`)

INTERSECT DISTINCT

SELECT repo_name AS repo
FROM `YOUR_PROJECT.YOUR_DATASET.repos_all`
WHERE package_count > 1 AND file_count > 7

INTERSECT DISTINCT

SELECT repo
FROM `YOUR_PROJECT.YOUR_DATASET.stars`
GROUP BY repo
HAVING COUNT(time)>1
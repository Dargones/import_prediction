/*Select the contents of all the relevant files*/

SELECT f.repo_name, f.path, f.id, c.content
FROM `bigquery-public-data.github_repos.contents` AS c
JOIN `YOUR_PROJECT.YOUR_DATASET.files_relevant` AS f on f.id=c.id
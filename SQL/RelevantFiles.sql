/*Select all the .java files from the repositories in question*/

SELECT repo_name, path, id
FROM `bigquery-public-data.github_repos.files` AS files
RIGHT JOIN `YOUR_PROJECT.YOUR_DATASET.repos_relevant` AS repos on files.repo_name=repos.repo
WHERE files.path LIKE "%.java"
/*Process the list of GitHub files and return the set of repositories that
contain .java files. Also record the number of files in repository and the
number of different directories containing .java files (which is at least as
big as the number of packages in a repository)*/

SELECT repo_name, COUNT(path) as file_count, COUNT(DISTINCT REGEXP_EXTRACT(path, r"^(.*)/[^/]*?\.java")) AS package_count
FROM `bigquery-public-data.github_repos.files` AS files
WHERE files.path LIKE "%.java"
GROUP BY repo_name
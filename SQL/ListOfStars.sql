/*Return the list of all stars events (time a star was added to a repository)
with corresponding repository name and timestamp*/

SELECT repo.name AS repo, created_at AS time
FROM `githubarchive.year.201*`
WHERE type="WatchEvent"
UNION ALL
SELECT repo.name AS repo, created_at AS time
FROM `githubarchive.month.202*`
WHERE type="WatchEvent"
/*Return a list of fork-original pairs. This might not include all forks,
since the API has changed several times since 2011 and the structure of the way
GitArchive recorded events changed as well. Further filtering is done at a
later step (see ../notebooks/Filtering.ipynb)*/

CREATE TEMPORARY FUNCTION parseJson(record STRING)
RETURNS STRING
LANGUAGE js AS """
  var tree = JSON.parse(record).forkee
  if (!tree) {
    return ""
  }
  if (tree.full_name) {
    return tree.full_name
  }
  if ((tree.owner) && (tree.owner.login)) {
    return tree.owner.login + '/' + tree.name
  }
  if (tree.name) {
    return tree.name
  }
  return ""
""";


SELECT repo.name AS original, parseJson(payload) AS fork
FROM `githubarchive.year.20*`
WHERE type="ForkEvent"
UNION ALL
SELECT repo.name AS original, parseJson(payload) AS fork
FROM `githubarchive.month.2020*`
WHERE type="ForkEvent"
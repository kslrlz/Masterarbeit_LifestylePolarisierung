# Gemeinsame Grundlage aller Auswertungsnotebooks: Pfade, Panel, Lebensstil-Etikett.
#
# Das Etikett folgt der 2016er-Regel. Fehlt das Urteil für 2016, wird es aus den
# Jahren 2017 bis 2024 ergänzt, aber nur wenn es dort eindeutig ausfällt.
# Uneindeutige Fälle bleiben offen und fallen aus allen Populationen heraus.
#
# Das Etikett wird hier EINMAL gebildet, in zwei Zuschnitten:
#   lifestyle_alle = alle Subreddits mit Urteil "nicht politisch" (Grundlage von FF2)
#   ana_subs       = dieselbe Regel, eingeschränkt auf das Panel (Grundlage von FF1)

library(data.table)
library(igraph)
library(ineq)
library(dbscan)
library(tidyverse)

# igraph und purrr definieren beide simplify(). Ausserhalb eines Notebooks
# hängt es von der Ladereihenfolge ab, welche Fassung gilt, deshalb der
# ausgeschriebene Aufruf igraph::simplify() im gesamten Code.

PROJ    <- Sys.getenv("MA_PROJ", unset = normalizePath("..", mustWork = FALSE))
VEK_DIR <- file.path(PROJ, "Data/Vektoren")
ALI_DIR <- file.path(PROJ, "SeNSe-main/SeNSe-main/output")
OUT_DIR <- "Auswertung_CSV"
ABB_DIR <- "Abbildungen"
JAHRE   <- 2016:2024

LLM_CSV     <- file.path(PROJ, "all_cluster_results_political.csv")
SCORE_CSV   <- file.path(PROJ, "politischer_wandel_2016_2024.csv")
CLUSTER_CSV <- file.path(PROJ, "subreddit_clusters_aligned_2016_2024_LABEL2016.csv")

dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)
dir.create(ABB_DIR, showWarnings = FALSE, recursive = TRUE)

vek_datei <- function(jahr) file.path(VEK_DIR, sprintf("vektoren_%d.txt", jahr))
ali_datei <- function(jahr) {
  if (jahr == 2016) vek_datei(2016)
  else file.path(ALI_DIR, sprintf("projected_%02d_onto_16_FULL.txt", jahr %% 100))
}

# Panel: Subreddits, die in jedem der neun Jahre einen Vektor haben.
panel <- Reduce(intersect, lapply(JAHRE, function(j)
  fread(vek_datei(j), select = 1, skip = 1, header = FALSE)[[1]]))

# Etikett je Subreddit, 2016er Urteil mit Ergänzung aus den Folgejahren.
labels <- fread(LLM_CSV, colClasses = "character", header = TRUE)
labels[, subreddit := sub("^r/", "", subreddit)]
labels <- unique(labels, by = "subreddit")

urteil_2016 <- labels[["2016"]]
folgejahre  <- as.matrix(labels[, as.character(2017:2024), with = FALSE])
n_true      <- rowSums(folgejahre == "true",  na.rm = TRUE)
n_false     <- rowSums(folgejahre == "false", na.rm = TRUE)
imputiert   <- ifelse(n_true == 0 & n_false > 0, "false",
              ifelse(n_false == 0 & n_true > 0, "true", NA_character_))
leer        <- is.na(urteil_2016) | urteil_2016 == ""

etikett <- ifelse(leer, imputiert, urteil_2016)
names(etikett) <- labels$subreddit

# "false" = nicht politisch = Lebensstil. Zeitinvariant, gilt für alle neun Jahre.
lifestyle_alle <- names(etikett)[which(etikett == "false")]

# Panel-Zuschnitt: TRUE = Lebensstil, FALSE = politisch, NA = offen
lifestyle <- unname(etikett[panel]) == "false"
names(lifestyle) <- panel

# Analysepanel für FF1: die Lebensstil-Subreddits des vollen Panels
ana_subs <- names(which(lifestyle))

cat(sprintf("Panel %d | Lebensstil %d | politisch %d | offen %d | Lebensstil gesamt %d\n",
            length(panel), sum(lifestyle, na.rm = TRUE),
            sum(!lifestyle, na.rm = TRUE), sum(is.na(lifestyle)),
            length(lifestyle_alle)))

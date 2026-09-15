library(readr)
library(dplyr)

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

set_a_dir <- "data/set-a"
output_dir <- "output"

dir.create(output_dir, showWarnings = FALSE)

# ------------------------------------------------------------
# Find all Set A patient files
# ------------------------------------------------------------

files <- list.files(
  set_a_dir,
  pattern = "\\.txt$",
  full.names = TRUE
)

cat("Number of Set A files:", length(files), "\n")

stopifnot(length(files) == 4000)

# ------------------------------------------------------------
# Function to read one ICU record
# ------------------------------------------------------------

read_one_record <- function(file) {

  record <- read_csv(
    file,
    show_col_types = FALSE
  )

  # RecordID is stored as one of the rows in each raw file.
  record_id <- record |>
    filter(Parameter == "RecordID") |>
    pull(Value)

  # Every row needs an explicit RecordID before all patient
  # files are stacked together.
  record |>
    mutate(
      RecordID = as.integer(record_id)
    ) |>
    select(
      RecordID,
      Time,
      Parameter,
      Value
    )
}

# ------------------------------------------------------------
# Test the function on one record first
# ------------------------------------------------------------

test_record <- read_one_record(files[1])

print(head(test_record))

stopifnot(
  n_distinct(test_record$RecordID) == 1
)

# ------------------------------------------------------------
# Combine all 4000 records
# ------------------------------------------------------------

set_a_long <- lapply(
  files,
  read_one_record
) |>
  bind_rows()

# ------------------------------------------------------------
# Sanity checks
# ------------------------------------------------------------

cat(
  "Rows in long table:",
  nrow(set_a_long),
  "\n"
)

cat(
  "Unique RecordIDs:",
  n_distinct(set_a_long$RecordID),
  "\n"
)

stopifnot(
  n_distinct(set_a_long$RecordID) == 4000
)

stopifnot(
  !any(
    is.na(set_a_long$RecordID)
  )
)

# ------------------------------------------------------------
# Save long-format table
# ------------------------------------------------------------

write_csv(
  set_a_long,
  file.path(
    output_dir,
    "set-a_long.csv"
  )
)

cat(
  "Created output/set-a_long.csv\n"
)
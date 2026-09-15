# BIOSTAT 707 Checkpoint 1
# Build one-row-per-admission wide table from cleaned long data.

library(readr)
library(dplyr)
library(tidyr)
library(here)

N_STAYS <- 4000

# -----------------------------------------------------------------------------
# 1. Load cleaned long data
# -----------------------------------------------------------------------------

long <- read_csv(
  here("output", "set-a_long_clean.csv"),
  show_col_types = FALSE
)

stopifnot(
  n_distinct(long$RecordID) == N_STAYS,
  all(c(
    "RecordID",
    "Time",
    "Parameter",
    "Value_clean",
    "hours_since_admission"
  ) %in% names(long))
)

cat("Long rows:", nrow(long), "\n")
cat("ICU stays:", n_distinct(long$RecordID), "\n")


# -----------------------------------------------------------------------------
# 2. Extract admission-level descriptors
# -----------------------------------------------------------------------------
#
# Age, Gender, Height, and ICUType are admission descriptors.
#
# Weight is different because it can be measured repeatedly during the
# 48-hour period, so Weight will be summarized with the time-series variables.

STATIC_VARS <- c(
  "Age",
  "Gender",
  "Height",
  "ICUType"
)

static <- long |>
  filter(Parameter %in% STATIC_VARS) |>
  select(
    RecordID,
    Parameter,
    Value_clean
  ) |>
  distinct() |>
  pivot_wider(
    names_from = Parameter,
    values_from = Value_clean
  )

stopifnot(
  nrow(static) == N_STAYS,
  n_distinct(static$RecordID) == N_STAYS
)


# -----------------------------------------------------------------------------
# 3. Select longitudinal variables
# -----------------------------------------------------------------------------

longitudinal <- long |>
  filter(
    !Parameter %in%
      c("RecordID", STATIC_VARS)
  ) |>
  arrange(
    RecordID,
    Parameter,
    hours_since_admission
  )


# -----------------------------------------------------------------------------
# 4. Summarize each variable over the full 48 hours
# -----------------------------------------------------------------------------
#
# For every ICU stay and every longitudinal variable:
#
#   count = number of valid measurements
#   first = first valid measurement
#   last  = last valid measurement
#   min   = minimum
#   max   = maximum
#   mean  = mean
#
# If a variable was never validly measured for a stay, it will later appear
# as missing in the wide table.

longitudinal_summary <- longitudinal |>
  filter(
    !is.na(Value_clean)
  ) |>
  summarise(
    count = n(),
    first = first(Value_clean),
    last = last(Value_clean),
    min = min(Value_clean),
    max = max(Value_clean),
    mean = mean(Value_clean),
    .by = c(
      RecordID,
      Parameter
    )
  )


# -----------------------------------------------------------------------------
# 5. Convert summaries to wide format
# -----------------------------------------------------------------------------

summary_wide <- longitudinal_summary |>
  pivot_wider(
    id_cols = RecordID,
    names_from = Parameter,
    values_from = c(
      count,
      first,
      last,
      min,
      max,
      mean
    ),
    names_glue = "{Parameter}_{.value}"
  )


# -----------------------------------------------------------------------------
# 6. Combine static and longitudinal features
# -----------------------------------------------------------------------------

wide <- static |>
  left_join(
    summary_wide,
    by = "RecordID"
  )

stopifnot(
  nrow(wide) == N_STAYS,
  n_distinct(wide$RecordID) == N_STAYS,
  !anyDuplicated(wide$RecordID)
)

cat(
  "Wide table before outcomes:",
  nrow(wide),
  "rows x",
  ncol(wide),
  "columns\n"
)


# -----------------------------------------------------------------------------
# 7. Load Challenge 2012 outcomes
# -----------------------------------------------------------------------------

outcomes_path <- here(
  "data",
  "Outcomes-a.txt"
)

stopifnot(
  file.exists(outcomes_path)
)

outcomes <- read_csv(
  outcomes_path,
  show_col_types = FALSE
)

cat("\nOutcome columns:\n")
print(names(outcomes))

stopifnot(
  "RecordID" %in% names(outcomes),
  nrow(outcomes) == N_STAYS,
  n_distinct(outcomes$RecordID) == N_STAYS
)


# -----------------------------------------------------------------------------
# 8. Join outcomes to wide table
# -----------------------------------------------------------------------------
#
# Use a left join so all 4,000 Set A admissions are retained.

wide <- wide |>
  left_join(
    outcomes,
    by = "RecordID"
  )

stopifnot(
  nrow(wide) == N_STAYS,
  n_distinct(wide$RecordID) == N_STAYS,
  !anyDuplicated(wide$RecordID)
)


# -----------------------------------------------------------------------------
# 9. Check outcome matching
# -----------------------------------------------------------------------------

outcome_columns <- setdiff(
  names(outcomes),
  "RecordID"
)

unmatched_outcomes <- wide |>
  filter(
    if_all(
      all_of(outcome_columns),
      is.na
    )
  )

stopifnot(
  nrow(unmatched_outcomes) == 0
)


# -----------------------------------------------------------------------------
# 10. Add explicit zero counts for never-measured variables
# -----------------------------------------------------------------------------
#
# Important distinction:
#
# If Lactate was never measured:
#
#   Lactate_count = 0
#   Lactate_mean  = NA
#   Lactate_min   = NA
#   ...
#
# Count = 0 is informative and should not itself be missing.

count_columns <- names(wide)[
  grepl("_count$", names(wide))
]

wide <- wide |>
  mutate(
    across(
      all_of(count_columns),
      ~ replace_na(.x, 0)
    )
  )


# -----------------------------------------------------------------------------
# 11. Save wide table
# -----------------------------------------------------------------------------

wide_path <- here(
  "output",
  "set-a_wide.csv"
)

write_csv(
  wide,
  wide_path
)


# -----------------------------------------------------------------------------
# 12. Summarize wide-table missingness
# -----------------------------------------------------------------------------

missingness <- tibble(
  variable = names(wide),
  n_missing = colSums(is.na(wide)),
  proportion_missing =
    colMeans(is.na(wide))
) |>
  arrange(
    desc(proportion_missing)
  )

write_csv(
  missingness,
  here(
    "output",
    "wide_missingness.csv"
  )
)


# -----------------------------------------------------------------------------
# 13. Final validation
# -----------------------------------------------------------------------------

stopifnot(
  nrow(wide) == N_STAYS,
  n_distinct(wide$RecordID) == N_STAYS,
  all(wide[count_columns] >= 0, na.rm = TRUE)
)

cat(
  "\n============================================================\n",
  "Wide-table construction complete.\n",
  "============================================================\n",
  sep = ""
)

cat(
  "Rows:",
  nrow(wide),
  "\n"
)

cat(
  "Columns:",
  ncol(wide),
  "\n"
)

cat(
  "Unique ICU stays:",
  n_distinct(wide$RecordID),
  "\n"
)

cat(
  "Output:",
  wide_path,
  "\n"
)
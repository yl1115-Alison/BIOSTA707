# BIOSTAT 707 Checkpoint 2
# 07_define_resampling.R
#
# Purpose:
#   Define a fixed, reproducible, stratified 5-fold cross-validation
#   assignment for Challenge 2012 Set A.
#
# Important:
#   This script ONLY defines resampling.
#
#   It DOES NOT:
#     - impute missing values
#     - standardize predictors
#     - perform feature selection
#     - fit models
#     - tune hyperparameters
#     - access Set B outcomes
#
# Inputs:
#   output/set-a_features.csv
#
# Outputs:
#   output/set-a_cv_folds.csv
#   output/cv_fold_summary.csv


# =============================================================================
# 1. Packages and constants
# =============================================================================

library(readr)
library(dplyr)
library(here)

N_STAYS <- 4000
N_FOLDS <- 5

OUTCOME_COL <- "In-hospital_death"

# Fixed seed so every model uses exactly the same folds.
CV_SEED <- 20261003


# =============================================================================
# 2. Load Set A feature table
# =============================================================================

input_path <- here(
  "output",
  "set-a_features.csv"
)

stopifnot(
  file.exists(input_path)
)

set_a <- read_csv(
  input_path,
  show_col_types = FALSE
)


# =============================================================================
# 3. Basic validation
# =============================================================================

required_columns <- c(
  "RecordID",
  OUTCOME_COL
)

stopifnot(
  all(
    required_columns %in%
      names(set_a)
  ),

  nrow(set_a) ==
    N_STAYS,

  n_distinct(
    set_a$RecordID
  ) ==
    N_STAYS,

  !anyDuplicated(
    set_a$RecordID
  ),

  !any(
    is.na(
      set_a$RecordID
    )
  ),

  !any(
    is.na(
      set_a[[OUTCOME_COL]]
    )
  ),

  all(
    set_a[[OUTCOME_COL]] %in%
      c(0, 1)
  )
)


n_deaths <- sum(
  set_a[[OUTCOME_COL]] == 1
)

mortality_rate <- mean(
  set_a[[OUTCOME_COL]] == 1
)


cat(
  "Set A ICU stays:",
  nrow(set_a),
  "\n"
)

cat(
  "Deaths:",
  n_deaths,
  "\n"
)

cat(
  "Overall mortality:",
  sprintf(
    "%.1f%%",
    100 * mortality_rate
  ),
  "\n"
)


# Check against the result established in Checkpoint 1.

stopifnot(
  n_deaths == 554
)


# =============================================================================
# 4. Create stratified folds
# =============================================================================
#
# Stratification is performed separately within:
#
#   survivors
#   deaths
#
# Each outcome group is randomly shuffled using the fixed seed, then assigned
# approximately equally across the five folds.
#
# This ensures that each validation fold has approximately the same mortality
# proportion as the full Set A cohort.


set.seed(
  CV_SEED
)


assign_folds_within_stratum <- function(df) {

  n <- nrow(df)

  shuffled_rows <- sample(
    seq_len(n),
    size = n,
    replace = FALSE
  )

  shuffled_df <- df[
    shuffled_rows,
    ,
    drop = FALSE
  ]

  shuffled_df$fold <- rep(
    seq_len(N_FOLDS),
    length.out = n
  )

  shuffled_df
}


fold_assignments <- set_a |>
  select(
    RecordID,
    all_of(
      OUTCOME_COL
    )
  ) |>
  group_split(
    .data[[OUTCOME_COL]]
  ) |>
  lapply(
    assign_folds_within_stratum
  ) |>
  bind_rows() |>
  arrange(
    RecordID
  )


# =============================================================================
# 5. Validate fold assignments
# =============================================================================

stopifnot(
  nrow(fold_assignments) ==
    N_STAYS,

  n_distinct(
    fold_assignments$RecordID
  ) ==
    N_STAYS,

  !anyDuplicated(
    fold_assignments$RecordID
  ),

  !any(
    is.na(
      fold_assignments$fold
    )
  ),

  all(
    fold_assignments$fold %in%
      seq_len(
        N_FOLDS
      )
  )
)


# Every RecordID in Set A must appear exactly once in the fold table.

stopifnot(
  setequal(
    set_a$RecordID,
    fold_assignments$RecordID
  )
)


# =============================================================================
# 6. Summarize fold balance
# =============================================================================

cv_fold_summary <- fold_assignments |>
  summarise(

    n =
      n(),

    deaths =
      sum(
        .data[[OUTCOME_COL]] == 1
      ),

    survivors =
      sum(
        .data[[OUTCOME_COL]] == 0
      ),

    mortality_rate =
      mean(
        .data[[OUTCOME_COL]] == 1
      ),

    .by =
      fold
  ) |>
  arrange(
    fold
  )


cat(
  "\nCross-validation fold summary:\n"
)

print(
  cv_fold_summary,
  n = Inf
)


# =============================================================================
# 7. Validate fold sizes and mortality balance
# =============================================================================
#
# With 4,000 observations and 5 folds, each fold should contain approximately
# 800 ICU stays.
#
# With 554 deaths, each fold should contain approximately 111 deaths.


expected_fold_size <-
  N_STAYS / N_FOLDS


stopifnot(
  max(
    cv_fold_summary$n
  ) -
    min(
      cv_fold_summary$n
    ) <=
    2
)


# Stratification should keep mortality close to the overall cohort rate.

max_mortality_difference <- max(
  abs(
    cv_fold_summary$mortality_rate -
      mortality_rate
  )
)


cat(
  "\nMaximum absolute fold-vs-overall mortality difference:",
  sprintf(
    "%.3f percentage points",
    100 *
      max_mortality_difference
  ),
  "\n"
)


# A one-percentage-point tolerance is intentionally generous relative to the
# balance expected from the stratified assignment.

stopifnot(
  max_mortality_difference <
    0.01
)


# =============================================================================
# 8. Verify training/validation sizes for every fold
# =============================================================================
#
# For fold k:
#
#   validation = fold k
#   training   = all other folds


cv_split_summary <- lapply(
  seq_len(
    N_FOLDS
  ),
  function(k) {

    validation <- fold_assignments |>
      filter(
        fold == k
      )

    training <- fold_assignments |>
      filter(
        fold != k
      )


    stopifnot(
      nrow(training) +
        nrow(validation) ==
        N_STAYS,

      length(
        intersect(
          training$RecordID,
          validation$RecordID
        )
      ) ==
        0,

      length(
        union(
          training$RecordID,
          validation$RecordID
        )
      ) ==
        N_STAYS
    )


    tibble(
      fold =
        k,

      n_train =
        nrow(training),

      n_validation =
        nrow(validation),

      deaths_train =
        sum(
          training[[OUTCOME_COL]] ==
            1
        ),

      deaths_validation =
        sum(
          validation[[OUTCOME_COL]] ==
            1
        ),

      mortality_train =
        mean(
          training[[OUTCOME_COL]] ==
            1
        ),

      mortality_validation =
        mean(
          validation[[OUTCOME_COL]] ==
            1
        )
    )
  }
) |>
  bind_rows()


cat(
  "\nTraining / validation split summary:\n"
)

print(
  cv_split_summary,
  n = Inf
)


# =============================================================================
# 9. Save fold assignments
# =============================================================================
#
# Do NOT save outcome in the permanent fold-assignment file.
#
# The only information needed by later scripts is:
#
#   RecordID
#   fold
#
# Keeping the fold file minimal also makes its role explicit.


fold_output <- fold_assignments |>
  select(
    RecordID,
    fold
  )


write_csv(
  fold_output,
  here(
    "output",
    "set-a_cv_folds.csv"
  )
)


write_csv(
  cv_fold_summary,
  here(
    "output",
    "cv_fold_summary.csv"
  )
)


# =============================================================================
# 10. Re-read and verify reproducible output
# =============================================================================

saved_folds <- read_csv(
  here(
    "output",
    "set-a_cv_folds.csv"
  ),
  show_col_types = FALSE
)


stopifnot(
  nrow(saved_folds) == N_STAYS,

  n_distinct(saved_folds$RecordID) == N_STAYS,

  !anyDuplicated(saved_folds$RecordID),

  all(
    saved_folds$fold %in%
      seq_len(N_FOLDS)
  ),

  isTRUE(
    all.equal(
      fold_output,
      saved_folds,
      check.attributes = FALSE
    )
  )
)


# =============================================================================
# 11. Final report
# =============================================================================

cat(
  "\n============================================================\n"
)

cat(
  "Checkpoint 2 resampling definition complete.\n"
)

cat(
  "============================================================\n"
)

cat(
  "Cross-validation folds:",
  N_FOLDS,
  "\n"
)

cat(
  "Random seed:",
  CV_SEED,
  "\n"
)

cat(
  "Set A ICU stays:",
  N_STAYS,
  "\n"
)

cat(
  "Set A deaths:",
  n_deaths,
  "\n"
)

cat(
  "Overall mortality:",
  sprintf(
    "%.1f%%",
    100 *
      mortality_rate
  ),
  "\n"
)

cat(
  "\nImportant:\n"
)

cat(
  paste(
    "All later model classes and hyperparameter configurations",
    "must use these same fixed folds.\n"
  )
)

cat(
  paste(
    "Imputation, scaling, feature selection, and other learned",
    "preprocessing must be fit separately inside each training fold.\n"
  )
)

cat(
  paste(
    "Set B is not used anywhere in this resampling definition.\n"
  )
)

cat(
  "\nGenerated outputs:\n"
)

cat(
  "  output/set-a_cv_folds.csv\n"
)

cat(
  "  output/cv_fold_summary.csv\n"
)
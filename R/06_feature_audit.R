# BIOSTAT 707 Checkpoint 2
# 06_feature_audit.R
#
# Purpose:
#   Audit the engineered Set A / Set B feature tables created by
#   05_build_features.R before supervised modeling.
#
# This script checks:
#   - dimensions and IDs
#   - predictor schema consistency
#   - forbidden columns
#   - missingness
#   - finite numeric values
#   - number of unique values
#   - zero-variance / near-zero-variance features
#   - logical consistency of explicit missingness indicators
#   - slope / delta availability
#   - descriptive Set A / Set B predictor differences
#
# IMPORTANT:
#   This script performs QA only.
#
#   It DOES NOT:
#     - impute missing values
#     - standardize predictors
#     - perform feature selection
#     - tune models
#     - use Set B outcomes
#
#   Set B comparisons are descriptive diagnostics only and must not be used
#   to tune the modeling pipeline.
#
# Inputs:
#   output/set-a_features.csv
#   output/set-b_features.csv
#   output/feature_dictionary.csv
#
# Outputs:
#   output/feature_audit.csv
#   output/feature_missingness.csv
#   output/feature_flags.csv
#   output/feature_shift_summary.csv
#   output/feature_missingness.png


# =============================================================================
# 1. Packages and constants
# =============================================================================

library(readr)
library(dplyr)
library(tidyr)
library(ggplot2)
library(here)

N_A <- 4000
N_B <- 4000

OUTCOME_COL <- "In-hospital_death"
BENCHMARK_COLS <- c(
  "SAPS-I",
  "SOFA"
)

NON_PREDICTOR_COLS_A <- c(
  "RecordID",
  OUTCOME_COL,
  BENCHMARK_COLS
)

NON_PREDICTOR_COLS_B <- c(
  "RecordID"
)


# =============================================================================
# 2. Load feature tables
# =============================================================================

a_path <- here(
  "output",
  "set-a_features.csv"
)

b_path <- here(
  "output",
  "set-b_features.csv"
)

dictionary_path <- here(
  "output",
  "feature_dictionary.csv"
)

stopifnot(
  file.exists(a_path),
  file.exists(b_path),
  file.exists(dictionary_path)
)

set_a <- read_csv(
  a_path,
  show_col_types = FALSE
)

set_b <- read_csv(
  b_path,
  show_col_types = FALSE
)

feature_dictionary <- read_csv(
  dictionary_path,
  show_col_types = FALSE
)


cat(
  "Set A:",
  nrow(set_a),
  "rows x",
  ncol(set_a),
  "columns\n"
)

cat(
  "Set B:",
  nrow(set_b),
  "rows x",
  ncol(set_b),
  "columns\n"
)


# =============================================================================
# 3. Basic structural validation
# =============================================================================

stopifnot(
  nrow(set_a) == N_A,
  nrow(set_b) == N_B,

  n_distinct(set_a$RecordID) == N_A,
  n_distinct(set_b$RecordID) == N_B,

  !anyDuplicated(set_a$RecordID),
  !anyDuplicated(set_b$RecordID),

  !any(is.na(set_a$RecordID)),
  !any(is.na(set_b$RecordID))
)


# Set A must contain the outcome and clinical benchmarks.

stopifnot(
  all(
    c(
      OUTCOME_COL,
      BENCHMARK_COLS
    ) %in%
      names(set_a)
  )
)


# Set B must remain outcome-blind at this stage.

stopifnot(
  !OUTCOME_COL %in%
    names(set_b)
)


# SAPS-I / SOFA are benchmark variables rather than engineered predictors.
# They should not be present in Set B's engineered predictor table.

stopifnot(
  !any(
    BENCHMARK_COLS %in%
      names(set_b)
  )
)


# Explicitly guard against known future/outcome-derived columns.

forbidden_cols <- c(
  "Length_of_stay",
  "Survival"
)

stopifnot(
  !any(
    forbidden_cols %in%
      names(set_a)
  ),

  !any(
    forbidden_cols %in%
      names(set_b)
  )
)


# =============================================================================
# 4. Identify predictor columns
# =============================================================================

predictors_a <- setdiff(
  names(set_a),
  NON_PREDICTOR_COLS_A
)

predictors_b <- setdiff(
  names(set_b),
  NON_PREDICTOR_COLS_B
)


stopifnot(
  identical(
    predictors_a,
    predictors_b
  )
)


predictor_names <- predictors_a

cat(
  "Engineered predictors:",
  length(predictor_names),
  "\n"
)


# Feature dictionary must describe every engineered predictor.

stopifnot(
  all(
    predictor_names %in%
      feature_dictionary$feature
  ),

  all(
    feature_dictionary$feature %in%
      predictor_names
  )
)


# =============================================================================
# 5. Validate outcome
# =============================================================================

stopifnot(
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
  "Set A deaths:",
  n_deaths,
  "\n"
)

cat(
  "Set A mortality:",
  sprintf(
    "%.1f%%",
    100 * mortality_rate
  ),
  "\n"
)


# These values were established in Checkpoint 1.
# This protects against an accidental outcome join error.

stopifnot(
  n_deaths == 554
)


# =============================================================================
# 6. Helper functions
# =============================================================================

safe_mean <- function(x) {

  x <- x[
    is.finite(x)
  ]

  if (
    length(x) == 0
  ) {
    return(NA_real_)
  }

  mean(x)
}


safe_sd <- function(x) {

  x <- x[
    is.finite(x)
  ]

  if (
    length(x) < 2
  ) {
    return(NA_real_)
  }

  sd(x)
}


safe_median <- function(x) {

  x <- x[
    is.finite(x)
  ]

  if (
    length(x) == 0
  ) {
    return(NA_real_)
  }

  median(x)
}


safe_quantile <- function(
  x,
  probability
) {

  x <- x[
    is.finite(x)
  ]

  if (
    length(x) == 0
  ) {
    return(NA_real_)
  }

  as.numeric(
    quantile(
      x,
      probs = probability,
      names = FALSE
    )
  )
}


safe_min <- function(x) {

  x <- x[
    is.finite(x)
  ]

  if (
    length(x) == 0
  ) {
    return(NA_real_)
  }

  min(x)
}


safe_max <- function(x) {

  x <- x[
    is.finite(x)
  ]

  if (
    length(x) == 0
  ) {
    return(NA_real_)
  }

  max(x)
}


count_nonfinite <- function(x) {

  if (!is.numeric(x)) {
    return(NA_integer_)
  }

  sum(
    !is.na(x) &
      !is.finite(x)
  )
}


count_unique_observed <- function(x) {

  length(
    unique(
      x[
        !is.na(x)
      ]
    )
  )
}


# Near-zero variance diagnostic:
#
# This is intentionally only a QA flag.
# We do NOT remove features here.
#
# A feature is flagged when:
#   - it has at least two observed values, AND
#   - the most common observed value accounts for >= 99% of observations.
#
# This simple rule is transparent and does not use outcomes.

is_near_zero_variance <- function(x) {

  x <- x[
    !is.na(x)
  ]

  if (
    length(x) == 0
  ) {
    return(TRUE)
  }

  unique_values <- unique(x)

  if (
    length(unique_values) <= 1
  ) {
    return(TRUE)
  }

  frequencies <- table(x)

  max(frequencies) /
    sum(frequencies) >= 0.99
}


# =============================================================================
# 7. Audit each predictor in Set A and Set B
# =============================================================================

audit_one_feature <- function(feature) {

  x_a <- set_a[[feature]]
  x_b <- set_b[[feature]]


  tibble(
    feature =
      feature,

    n_missing_a =
      sum(
        is.na(x_a)
      ),

    proportion_missing_a =
      mean(
        is.na(x_a)
      ),

    n_missing_b =
      sum(
        is.na(x_b)
      ),

    proportion_missing_b =
      mean(
        is.na(x_b)
      ),

    n_unique_a =
      count_unique_observed(
        x_a
      ),

    n_unique_b =
      count_unique_observed(
        x_b
      ),

    n_nonfinite_a =
      count_nonfinite(
        x_a
      ),

    n_nonfinite_b =
      count_nonfinite(
        x_b
      ),

    mean_a =
      safe_mean(
        x_a
      ),

    sd_a =
      safe_sd(
        x_a
      ),

    median_a =
      safe_median(
        x_a
      ),

    q01_a =
      safe_quantile(
        x_a,
        0.01
      ),

    q99_a =
      safe_quantile(
        x_a,
        0.99
      ),

    min_a =
      safe_min(
        x_a
      ),

    max_a =
      safe_max(
        x_a
      ),

    mean_b =
      safe_mean(
        x_b
      ),

    sd_b =
      safe_sd(
        x_b
      ),

    median_b =
      safe_median(
        x_b
      ),

    q01_b =
      safe_quantile(
        x_b,
        0.01
      ),

    q99_b =
      safe_quantile(
        x_b,
        0.99
      ),

    min_b =
      safe_min(
        x_b
      ),

    max_b =
      safe_max(
        x_b
      ),

    zero_variance_a =
      count_unique_observed(
        x_a
      ) <= 1,

    zero_variance_b =
      count_unique_observed(
        x_b
      ) <= 1,

    near_zero_variance_a =
      is_near_zero_variance(
        x_a
      ),

    near_zero_variance_b =
      is_near_zero_variance(
        x_b
      )
  )
}


feature_audit <- lapply(
  predictor_names,
  audit_one_feature
) |>
  bind_rows() |>
  left_join(
    feature_dictionary,
    by =
      "feature"
  ) |>
  relocate(
    feature,
    source_parameter,
    feature_type
  )


# =============================================================================
# 8. Fail on non-finite values
# =============================================================================
#
# NA is expected.
# Inf / -Inf / NaN are not acceptable engineered feature values.


nonfinite_problems <- feature_audit |>
  filter(
    coalesce(
      n_nonfinite_a,
      0L
    ) > 0 |
      coalesce(
        n_nonfinite_b,
        0L
      ) > 0
  )


if (
  nrow(nonfinite_problems) > 0
) {

  print(
    nonfinite_problems |>
      select(
        feature,
        n_nonfinite_a,
        n_nonfinite_b
      ),
    n = Inf
  )

  stop(
    "Non-finite engineered feature values detected."
  )
}


# =============================================================================
# 9. Missingness summary
# =============================================================================

feature_missingness <- feature_audit |>
  select(
    feature,
    source_parameter,
    feature_type,
    n_missing_a,
    proportion_missing_a,
    n_missing_b,
    proportion_missing_b
  ) |>
  mutate(
    missingness_difference_b_minus_a =
      proportion_missing_b -
      proportion_missing_a
  ) |>
  arrange(
    desc(
      proportion_missing_a
    ),
    feature
  )


# =============================================================================
# 10. Validate explicit missingness indicators
# =============================================================================

missing_indicator_features <- grep(
  "_missing$",
  predictor_names,
  value = TRUE
)


for (
  feature in missing_indicator_features
) {

  stopifnot(
    all(
      set_a[[feature]] %in%
        c(0, 1)
    ),

    all(
      set_b[[feature]] %in%
        c(0, 1)
    )
  )
}


# For every "<Parameter>_missing" feature, check whether the corresponding
# "<Parameter>_count" column exists. If it does:
#
#   missing == 1  <=>  count == 0


missing_count_checks <- lapply(
  missing_indicator_features,
  function(missing_feature) {

    prefix <- sub(
      "_missing$",
      "",
      missing_feature
    )

    count_feature <- paste0(
      prefix,
      "_count"
    )


    if (
      !count_feature %in%
        predictor_names
    ) {

      return(
        tibble(
          missing_feature =
            missing_feature,

          count_feature =
            NA_character_,

          mismatches_a =
            NA_integer_,

          mismatches_b =
            NA_integer_
        )
      )
    }


    expected_a <- as.integer(
      set_a[[count_feature]] == 0
    )

    expected_b <- as.integer(
      set_b[[count_feature]] == 0
    )


    tibble(
      missing_feature =
        missing_feature,

      count_feature =
        count_feature,

      mismatches_a =
        sum(
          set_a[[missing_feature]] !=
            expected_a,
          na.rm = TRUE
        ),

      mismatches_b =
        sum(
          set_b[[missing_feature]] !=
            expected_b,
          na.rm = TRUE
        )
    )
  }
) |>
  bind_rows()


logic_failures <- missing_count_checks |>
  filter(
    coalesce(
      mismatches_a,
      0L
    ) > 0 |
      coalesce(
        mismatches_b,
        0L
      ) > 0
  )


if (
  nrow(logic_failures) > 0
) {

  print(
    logic_failures,
    n = Inf
  )

  stop(
    "Missingness-indicator / count inconsistency detected."
  )
}


# Weight has no count feature, so check its missingness indicator separately.

if (
  all(
    c(
      "Weight_first",
      "Weight_missing"
    ) %in%
      predictor_names
  )
) {

  stopifnot(
    all(
      set_a$Weight_missing ==
        as.integer(
          is.na(
            set_a$Weight_first
          )
        )
    ),

    all(
      set_b$Weight_missing ==
        as.integer(
          is.na(
            set_b$Weight_first
          )
        )
    )
  )
}


# =============================================================================
# 11. Audit slope and delta availability
# =============================================================================

slope_features <- grep(
  "_slope$",
  predictor_names,
  value = TRUE
)

delta_features <- grep(
  "_delta$",
  predictor_names,
  value = TRUE
)


slope_delta_summary <- bind_rows(

  tibble(
    feature =
      slope_features,
    family =
      "slope"
  ),

  tibble(
    feature =
      delta_features,
    family =
      "delta"
  )
) |>
  left_join(
    feature_audit |>
      select(
        feature,
        proportion_missing_a,
        proportion_missing_b
      ),
    by =
      "feature"
  ) |>
  arrange(
    family,
    desc(
      proportion_missing_a
    )
  )


cat(
  "\nSlope / delta availability:\n"
)

print(
  slope_delta_summary,
  n = Inf
)


# =============================================================================
# 12. Feature QA flags
# =============================================================================
#
# These flags are for REVIEW ONLY.
# They do not automatically remove any predictor.


feature_flags <- feature_audit |>
  transmute(
    feature,
    source_parameter,
    feature_type,

    proportion_missing_a,
    proportion_missing_b,

    zero_variance_a,
    zero_variance_b,

    near_zero_variance_a,
    near_zero_variance_b,

    flag_high_missing_a =
      proportion_missing_a >= 0.80,

    flag_high_missing_b =
      proportion_missing_b >= 0.80,

    flag_missingness_difference =
      abs(
        proportion_missing_b -
          proportion_missing_a
      ) >= 0.10,

    flag_review =
      zero_variance_a |
      near_zero_variance_a |
      proportion_missing_a >= 0.80
  ) |>
  arrange(
    desc(flag_review),
    desc(proportion_missing_a),
    feature
  )


# =============================================================================
# 13. Descriptive A/B shift summary
# =============================================================================
#
# IMPORTANT:
#
# Set B is the future test set. These statistics are descriptive QA only.
# They must NOT be used to tune features, hyperparameters, or model classes.
#
# For continuous features, calculate a simple standardized mean difference
# using Set A SD as the reference scale:
#
#   SMD = (mean_B - mean_A) / SD_A
#
# This is NOT used for model selection.


feature_shift_summary <- feature_audit |>
  mutate(

    standardized_mean_difference =
      case_when(

        is.na(mean_a) |
          is.na(mean_b) |
          is.na(sd_a) |
          sd_a == 0 ~
          NA_real_,

        TRUE ~
          (mean_b - mean_a) /
            sd_a
      ),

    missingness_difference =
      proportion_missing_b -
      proportion_missing_a
  ) |> 
    select(
    feature,
    source_parameter,
    feature_type,

    mean_a,
    mean_b,
    sd_a,

    standardized_mean_difference,

    proportion_missing_a,
    proportion_missing_b,
    missingness_difference
  ) |>
  arrange(
    desc(
      abs(
        standardized_mean_difference
      )
    )
  )


# =============================================================================
# 14. Missingness plot
# =============================================================================
#
# Plot Set A missingness only for modeling-development interpretation.
#
# Set B is intentionally excluded from this figure because Set B is the
# reserved test set and should not guide model-development decisions.


missingness_plot_data <- feature_missingness |>
  arrange(
    desc(
      proportion_missing_a
    )
  ) |>
  slice_head(
    n = 50
  ) |>
  mutate(
    feature =
      factor(
        feature,
        levels =
          rev(feature)
      )
  )


missingness_plot <- ggplot(
  missingness_plot_data,
  aes(
    x = proportion_missing_a,
    y = feature
  )
) +
  geom_col() +
  scale_x_continuous(
    labels =
      scales::percent_format(
        accuracy = 1
      ),
    limits =
      c(0, 1)
  ) +
  labs(
    title =
      "Missingness among engineered Set A features",

    subtitle =
      "Top 50 features by proportion missing before CV-based imputation",

    x =
      "Proportion missing",

    y =
      NULL
  ) +
  theme_minimal(
    base_size = 10
  )


ggsave(
  filename =
    here(
      "output",
      "feature_missingness.png"
    ),

  plot =
    missingness_plot,

  width =
    9,

  height =
    10,

  dpi =
    150
)


# =============================================================================
# 15. Additional validation of feature families
# =============================================================================

# Count features should be non-negative.

count_features <- grep(
  "_count$",
  predictor_names,
  value = TRUE
)


for (
  feature in count_features
) {

  stopifnot(
    all(
      set_a[[feature]] >= 0,
      na.rm = TRUE
    ),

    all(
      set_b[[feature]] >= 0,
      na.rm = TRUE
    )
  )
}


# Missingness indicators should themselves never be missing.

for (
  feature in missing_indicator_features
) {

  stopifnot(
    !any(
      is.na(
        set_a[[feature]]
      )
    ),

    !any(
      is.na(
        set_b[[feature]]
      )
    )
  )
}


# MechVent_ever must remain binary.

if (
  "MechVent_ever" %in%
    predictor_names
) {

  stopifnot(
    all(
      set_a$MechVent_ever %in%
        c(0, 1)
    ),

    all(
      set_b$MechVent_ever %in%
        c(0, 1)
    )
  )
}


# Static categorical variables should retain their documented coding.

if (
  "Gender" %in%
    predictor_names
) {

  stopifnot(
    all(
      set_a$Gender[
        !is.na(
          set_a$Gender
        )
      ] %in%
        c(0, 1)
    ),

    all(
      set_b$Gender[
        !is.na(
          set_b$Gender
        )
      ] %in%
        c(0, 1)
    )
  )
}


if (
  "ICUType" %in%
    predictor_names
) {

  stopifnot(
    all(
      set_a$ICUType[
        !is.na(
          set_a$ICUType
        )
      ] %in%
        1:4
    ),

    all(
      set_b$ICUType[
        !is.na(
          set_b$ICUType
        )
      ] %in%
        1:4
    )
  )
}


# =============================================================================
# 16. Summarize important QA findings
# =============================================================================

n_zero_variance_a <- sum(
  feature_audit$zero_variance_a
)

n_near_zero_variance_a <- sum(
  feature_audit$near_zero_variance_a
)

n_high_missing_a <- sum(
  feature_audit$proportion_missing_a >=
    0.80
)

n_complete_a <- sum(
  feature_audit$proportion_missing_a ==
    0
)

n_missingness_shift <- sum(
  abs(
    feature_audit$proportion_missing_b -
      feature_audit$proportion_missing_a
  ) >=
    0.10
)


cat(
  "\nFeature QA summary:\n"
)

cat(
  "  Zero-variance features in Set A:",
  n_zero_variance_a,
  "\n"
)

cat(
  "  Near-zero-variance features in Set A:",
  n_near_zero_variance_a,
  "\n"
)

cat(
  "  Features >=80% missing in Set A:",
  n_high_missing_a,
  "\n"
)

cat(
  "  Features with no missing values in Set A:",
  n_complete_a,
  "\n"
)

cat(
  "  Features with >=10 percentage-point A/B missingness difference:",
  n_missingness_shift,
  "\n"
)


# Print high-missingness features for review.

high_missing_features <- feature_audit |>
  filter(
    proportion_missing_a >=
      0.80
  ) |>
  select(
    feature,
    source_parameter,
    feature_type,
    proportion_missing_a,
    proportion_missing_b
  ) |>
  arrange(
    desc(
      proportion_missing_a
    )
  )


if (
  nrow(high_missing_features) > 0
) {

  cat(
    "\nFeatures >=80% missing in Set A:\n"
  )

  print(
    high_missing_features,
    n = Inf
  )
}


# Print zero / near-zero variance features.

variance_review <- feature_audit |>
  filter(
    zero_variance_a |
      near_zero_variance_a
  ) |>
  select(
    feature,
    source_parameter,
    feature_type,
    n_unique_a,
    zero_variance_a,
    near_zero_variance_a,
    proportion_missing_a
  ) |>
  arrange(
    desc(
      zero_variance_a
    ),
    desc(
      near_zero_variance_a
    ),
    feature
  )


if (
  nrow(variance_review) > 0
) {

  cat(
    "\nZero / near-zero variance features for review:\n"
  )

  print(
    variance_review,
    n = Inf
  )
}


# =============================================================================
# 17. Save audit outputs
# =============================================================================

write_csv(
  feature_audit,
  here(
    "output",
    "feature_audit.csv"
  )
)


write_csv(
  feature_missingness,
  here(
    "output",
    "feature_missingness.csv"
  )
)


write_csv(
  feature_flags,
  here(
    "output",
    "feature_flags.csv"
  )
)


write_csv(
  feature_shift_summary,
  here(
    "output",
    "feature_shift_summary.csv"
  )
)


# =============================================================================
# 18. Final validation
# =============================================================================

stopifnot(
  nrow(feature_audit) ==
    length(
      predictor_names
    ),

  nrow(feature_dictionary) ==
    length(
      predictor_names
    ),

  nrow(feature_missingness) ==
    length(
      predictor_names
    ),

  nrow(feature_flags) ==
    length(
      predictor_names
    )
)


# =============================================================================
# 19. Final report
# =============================================================================

cat(
  "\n============================================================\n"
)

cat(
  "Checkpoint 2 feature audit complete.\n"
)

cat(
  "============================================================\n"
)

cat(
  "Set A ICU stays:",
  nrow(set_a),
  "\n"
)

cat(
  "Set B ICU stays:",
  nrow(set_b),
  "\n"
)

cat(
  "Engineered predictors audited:",
  length(
    predictor_names
  ),
  "\n"
)

cat(
  "Set A deaths:",
  n_deaths,
  "\n"
)

cat(
  "Set A mortality:",
  sprintf(
    "%.1f%%",
    100 *
      mortality_rate
  ),
  "\n"
)

cat(
  "Non-finite feature problems:",
  nrow(
    nonfinite_problems
  ),
  "\n"
)

cat(
  "Missingness/count logic failures:",
  nrow(
    logic_failures
  ),
  "\n"
)

cat(
  "\nImportant:\n"
)

cat(
  paste(
    "No imputation, scaling, feature selection,",
    "or model tuning was performed.\n"
  )
)

cat(
  paste(
    "Set B predictor comparisons are descriptive QA only",
    "and must not guide model development.\n"
  )
)

cat(
  "\nGenerated outputs:\n"
)

cat(
  "  output/feature_audit.csv\n"
)

cat(
  "  output/feature_missingness.csv\n"
)

cat(
  "  output/feature_flags.csv\n"
)

cat(
  "  output/feature_shift_summary.csv\n"
)

cat(
  "  output/feature_missingness.png\n"
)
# BIOSTAT 707 Checkpoint 2
# 12_test_set_evaluation.R
#
# FINAL SINGLE-LOOK SET-B EVALUATION
#
# Primary model:
#   LASSO logistic regression
#
# The model-development process was completed and frozen using Set A.
#
# This script:
#
#   1. reconstructs the frozen Set-A preprocessing pipeline
#   2. fits the frozen LASSO model on all Set A
#   3. applies Set-A preprocessing unchanged to Set B
#   4. generates Set-B predictions
#   5. reads Set-B outcomes
#   6. evaluates LASSO, SAPS-I, and SOFA exactly once
#
# NO:
#   - feature redesign
#   - hyperparameter tuning
#   - Set-B-derived imputation
#   - Set-B-derived scaling
#   - model switching
#   - threshold optimization
#
# Inputs:
#
#   output/set-a_features.csv
#   output/set-b_features.csv
#   output/feature_dictionary.csv
#   output/regularized_final_configuration_set_a.csv
#   data/Outcomes-b.txt
#
# Outputs:
#
#   output/set-b_predictions_final.csv
#   output/set-b_test_performance.csv
#   output/set-b_calibration_summary.csv
#   output/set-b_calibration_bins.csv
#   output/set-b_calibration_plot.png
#   output/set-b_final_lasso_coefficients.csv
#   output/set-b_final_preprocessing_summary.csv


# =============================================================================
# 1. Packages and constants
# =============================================================================

library(readr)
library(dplyr)
library(ggplot2)
library(glmnet)
library(pROC)
library(here)

N_A <- 4000
N_B <- 4000

OUTCOME <- "In-hospital_death"

MAX_MISSING_PROPORTION <- 0.80

EPS <- 1e-6


# =============================================================================
# 2. Input paths
# =============================================================================

set_a_path <- here(
  "output",
  "set-a_features.csv"
)

set_b_path <- here(
  "output",
  "set-b_features.csv"
)

dictionary_path <- here(
  "output",
  "feature_dictionary.csv"
)

configuration_path <- here(
  "output",
  "regularized_final_configuration_set_a.csv"
)

outcomes_b_path <- here(
  "data",
  "Outcomes-b.txt"
)


required_files <- c(
  set_a_path,
  set_b_path,
  dictionary_path,
  configuration_path,
  outcomes_b_path
)


missing_files <- required_files[
  !file.exists(
    required_files
  )
]


if (
  length(
    missing_files
  ) >
    0
) {

  stop(
    paste(
      "Required file(s) missing:",
      paste(
        missing_files,
        collapse = ", "
      )
    )
  )
}


# =============================================================================
# 3. Load Set A and Set B predictors
# =============================================================================
#
# Set B predictors are loaded before outcomes.
#
# No modeling decision below may depend on Set B outcomes.


set_a <- read_csv(
  set_a_path,
  show_col_types = FALSE
)


set_b <- read_csv(
  set_b_path,
  show_col_types = FALSE
)


feature_dictionary <- read_csv(
  dictionary_path,
  show_col_types = FALSE
)


stopifnot(
  nrow(set_a) == N_A,
  nrow(set_b) == N_B,

  n_distinct(
    set_a$RecordID
  ) ==
    N_A,

  n_distinct(
    set_b$RecordID
  ) ==
    N_B,

  !anyDuplicated(
    set_a$RecordID
  ),

  !anyDuplicated(
    set_b$RecordID
  ),

  OUTCOME %in%
    names(set_a),

  !OUTCOME %in%
    names(set_b)
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


# =============================================================================
# 4. Define frozen engineered predictor space
# =============================================================================

non_predictors_a <- c(
  "RecordID",
  OUTCOME,
  "SAPS-I",
  "SOFA"
)


all_predictors <- setdiff(
  names(set_a),
  non_predictors_a
)


stopifnot(
  length(
    all_predictors
  ) ==
    313,

  identical(
    all_predictors,
    setdiff(
      names(set_b),
      "RecordID"
    )
  ),

  all(
    all_predictors %in%
      feature_dictionary$feature
  )
)


CATEGORICAL_FEATURES <- c(
  "ICUType"
)


# =============================================================================
# 5. Load frozen LASSO hyperparameters
# =============================================================================
#
# These were selected using Set A only in 09_regularized_models.R.
#
# DO NOT recompute lambda using Set B.


configuration <- read_csv(
  configuration_path,
  show_col_types = FALSE
)


lasso_configuration <- configuration |>
  filter(
    model ==
      "LASSO"
  )


stopifnot(
  nrow(
    lasso_configuration
  ) ==
    1
)


FROZEN_ALPHA <- lasso_configuration$
  alpha[[1]]

FROZEN_LAMBDA <- lasso_configuration$
  lambda[[1]]


stopifnot(
  FROZEN_ALPHA ==
    1,

  is.finite(
    FROZEN_LAMBDA
  ),

  FROZEN_LAMBDA >
    0
)


cat(
  "\nFrozen primary model:\n"
)

cat(
  "  Model: LASSO\n"
)

cat(
  "  Alpha:",
  FROZEN_ALPHA,
  "\n"
)

cat(
  "  Lambda:",
  FROZEN_LAMBDA,
  "\n"
)


# =============================================================================
# 6. Frozen preprocessing helpers
# =============================================================================
#
# These reproduce the preprocessing used in 09.
#
# CRITICAL:
#   every fitted preprocessing quantity comes from Set A only.


fit_mode <- function(x) {

  x <- x[
    !is.na(x)
  ]


  if (
    length(x) ==
      0
  ) {

    stop(
      "Cannot estimate mode from an entirely missing variable."
    )
  }


  tab <- table(
    x
  )


  names(
    tab
  )[
    which.max(
      tab
    )
  ]
}


# -----------------------------------------------------------------------------
# 6a. Feature eligibility
# -----------------------------------------------------------------------------

fit_feature_filter <- function(
  training_data
) {

  missing_proportion <- vapply(
    all_predictors,
    function(feature) {

      mean(
        is.na(
          training_data[[feature]]
        )
      )
    },
    numeric(1)
  )


  feature_types <- feature_dictionary$
    feature_type[
      match(
        all_predictors,
        feature_dictionary$feature
      )
    ]


  protected <-
    grepl(
      "^missingness:",
      feature_types
    ) |
    grepl(
      "^measurement:count",
      feature_types
    )


  eligible <-
    missing_proportion <
      MAX_MISSING_PROPORTION |
      protected


  candidates <- all_predictors[
    eligible
  ]


  has_variation <- vapply(
    candidates,
    function(feature) {

      x <- training_data[[feature]]

      x <- x[
        !is.na(x)
      ]


      length(
        unique(
          x
        )
      ) >
        1
    },
    logical(1)
  )


  list(
    selected =
      candidates[
        has_variation
      ],

    excluded_high_missing =
      all_predictors[
        !eligible
      ],

    excluded_zero_variance =
      candidates[
        !has_variation
      ],

    missing_proportion =
      missing_proportion
  )
}


# -----------------------------------------------------------------------------
# 6b. Set-A imputation parameters
# -----------------------------------------------------------------------------

fit_imputer <- function(
  training_data,
  features
) {

  categorical <- intersect(
    CATEGORICAL_FEATURES,
    features
  )


  numeric_features <- setdiff(
    features,
    categorical
  )


  medians <- vapply(
    numeric_features,
    function(feature) {

      x <- training_data[[feature]]

      x <- x[
        !is.na(x) &
          is.finite(x)
      ]


      if (
        length(x) ==
          0
      ) {

        return(
          NA_real_
        )
      }


      median(
        x
      )
    },
    numeric(1)
  )


  if (
    any(
      is.na(
        medians
      )
    )
  ) {

    bad <- names(
      medians
    )[
      is.na(
        medians
      )
    ]


    stop(
      paste(
        "Cannot estimate Set-A median for:",
        paste(
          bad,
          collapse = ", "
        )
      )
    )
  }


  modes <- list()


  for (
    feature in categorical
  ) {

    modes[[feature]] <- fit_mode(
      training_data[[feature]]
    )
  }


  list(
    medians =
      medians,

    modes =
      modes
  )
}


apply_imputer <- function(
  data,
  imputer
) {

  output <- data


  for (
    feature in names(
      imputer$medians
    )
  ) {

    missing <- is.na(
      output[[feature]]
    )


    output[[feature]][missing] <-
      imputer$medians[[feature]]
  }


  for (
    feature in names(
      imputer$modes
    )
  ) {

    missing <- is.na(
      output[[feature]]
    )


    output[[feature]][missing] <-
      as.numeric(
        imputer$modes[[feature]]
      )
  }


  output
}


# -----------------------------------------------------------------------------
# 6c. Set-A design-matrix schema
# -----------------------------------------------------------------------------

fit_design_schema <- function(
  data,
  features
) {

  working <- data |>
    select(
      all_of(
        features
      )
    )


  if (
    "ICUType" %in%
      features
  ) {

    working$ICUType <- factor(
      working$ICUType,
      levels =
        1:4
    )
  }


  x <- model.matrix(
    ~ .,
    data =
      working
  )


  x <- x[
    ,
    colnames(x) !=
      "(Intercept)",
    drop = FALSE
  ]


  list(
    columns =
      colnames(
        x
      )
  )
}


make_design_matrix <- function(
  data,
  features,
  schema
) {

  working <- data |>
    select(
      all_of(
        features
      )
    )


  if (
    "ICUType" %in%
      features
  ) {

    working$ICUType <- factor(
      working$ICUType,
      levels =
        1:4
    )
  }


  x <- model.matrix(
    ~ .,
    data =
      working
  )


  x <- x[
    ,
    colnames(x) !=
      "(Intercept)",
    drop = FALSE
  ]


  missing_columns <- setdiff(
    schema$columns,
    colnames(
      x
    )
  )


  if (
    length(
      missing_columns
    ) >
      0
  ) {

    zeros <- matrix(
      0,
      nrow =
        nrow(x),
      ncol =
        length(
          missing_columns
        )
    )


    colnames(
      zeros
    ) <-
      missing_columns


    x <- cbind(
      x,
      zeros
    )
  }


  extra_columns <- setdiff(
    colnames(
      x
    ),
    schema$columns
  )


  if (
    length(
      extra_columns
    ) >
      0
  ) {

    x <- x[
      ,
      setdiff(
        colnames(x),
        extra_columns
      ),
      drop = FALSE
    ]
  }


  x <- x[
    ,
    schema$columns,
    drop = FALSE
  ]


  x
}


# -----------------------------------------------------------------------------
# 6d. Set-A scaling
# -----------------------------------------------------------------------------

fit_scaler <- function(x) {

  center <- colMeans(
    x
  )


  scale <- apply(
    x,
    2,
    sd
  )


  bad <- !is.finite(
    scale
  ) |
    scale ==
      0


  if (
    any(
      bad
    )
  ) {

    stop(
      paste(
        "Invalid Set-A scaling parameter for:",
        paste(
          names(scale)[bad],
          collapse = ", "
        )
      )
    )
  }


  list(
    center =
      center,

    scale =
      scale
  )
}


apply_scaler <- function(
  x,
  scaler
) {

  x <- sweep(
    x,
    2,
    scaler$center,
    FUN = "-"
  )


  x <- sweep(
    x,
    2,
    scaler$scale,
    FUN = "/"
  )


  x
}


# =============================================================================
# 7. Fit frozen preprocessing using Set A ONLY
# =============================================================================

feature_filter <- fit_feature_filter(
  set_a
)


selected_features <- feature_filter$
  selected


imputer <- fit_imputer(
  training_data =
    set_a,

  features =
    selected_features
)


set_a_imputed <- apply_imputer(
  set_a,
  imputer
)


design_schema <- fit_design_schema(
  set_a_imputed,
  selected_features
)


x_a_unscaled <- make_design_matrix(
  set_a_imputed,
  selected_features,
  design_schema
)


scaler <- fit_scaler(
  x_a_unscaled
)


x_a <- apply_scaler(
  x_a_unscaled,
  scaler
)


y_a <- set_a[[OUTCOME]]


stopifnot(
  all(
    is.finite(
      x_a
    )
  )
)


cat(
  "\nFrozen Set-A preprocessing:\n"
)

cat(
  "  Raw engineered predictors:",
  length(
    all_predictors
  ),
  "\n"
)

cat(
  "  Retained features:",
  length(
    selected_features
  ),
  "\n"
)

cat(
  "  Design columns:",
  ncol(
    x_a
  ),
  "\n"
)

cat(
  "  Excluded >=80% missing:",
  length(
    feature_filter$
      excluded_high_missing
  ),
  "\n"
)

cat(
  "  Excluded zero variance:",
  length(
    feature_filter$
      excluded_zero_variance
  ),
  "\n"
)


# =============================================================================
# 8. Fit frozen final LASSO on all Set A
# =============================================================================

final_lasso <- glmnet(
  x =
    x_a,

  y =
    y_a,

  family =
    "binomial",

  alpha =
    FROZEN_ALPHA,

  lambda =
    FROZEN_LAMBDA,

  standardize =
    FALSE,

  intercept =
    TRUE
)


coefficient_matrix <- as.matrix(
  coef(
    final_lasso,
    s =
      FROZEN_LAMBDA
  )
)


lasso_coefficients <- tibble(
  term =
    rownames(
      coefficient_matrix
    ),

  coefficient =
    as.numeric(
      coefficient_matrix[
        ,
        1
      ]
    ),

  selected =
    as.numeric(
      coefficient_matrix[
        ,
        1
      ]
    ) !=
      0
)


cat(
  "\nFinal LASSO nonzero coefficients:",
  sum(
    lasso_coefficients$selected[
      lasso_coefficients$term !=
        "(Intercept)"
    ]
  ),
  "\n"
)


# =============================================================================
# 9. Apply frozen Set-A preprocessing to Set B
# =============================================================================
#
# CRITICAL:
#
# Nothing below is fit using Set B.


set_b_imputed <- apply_imputer(
  set_b,
  imputer
)


x_b_unscaled <- make_design_matrix(
  set_b_imputed,
  selected_features,
  design_schema
)


x_b <- apply_scaler(
  x_b_unscaled,
  scaler
)


stopifnot(
  nrow(
    x_b
  ) ==
    N_B,

  identical(
    colnames(
      x_a
    ),
    colnames(
      x_b
    )
  ),

  all(
    is.finite(
      x_b
    )
  )
)


# =============================================================================
# 10. Generate frozen LASSO Set-B predictions
# =============================================================================
#
# At this point Set-B outcomes have still not been read.


lasso_probability_b <- as.numeric(
  predict(
    final_lasso,
    newx =
      x_b,
    type =
      "response",
    s =
      FROZEN_LAMBDA
  )
)


stopifnot(
  length(
    lasso_probability_b
  ) ==
    N_B,

  all(
    is.finite(
      lasso_probability_b
    )
  ),

  all(
    lasso_probability_b >=
      0 &
      lasso_probability_b <=
      1
  )
)


set_b_lasso_predictions <- tibble(
  RecordID =
    set_b$RecordID,

  model =
    "LASSO",

  probability =
    lasso_probability_b
)


cat(
  "\nFrozen Set-B predictions generated BEFORE outcome loading.\n"
)


# =============================================================================
# 11. NOW load Set B outcomes
# =============================================================================
#
# This is the single-look point.


cat(
  "\n============================================================\n"
)

cat(
  "Loading Set B outcomes: SINGLE-LOOK TEST EVALUATION\n"
)

cat(
  "============================================================\n"
)


outcomes_b <- read_csv(
  outcomes_b_path,
  show_col_types = FALSE
)


# =============================================================================
# 12. Validate Set B outcomes
# =============================================================================

required_outcome_columns <- c(
  "RecordID",
  "SAPS-I",
  "SOFA",
  "Length_of_stay",
  "Survival",
  OUTCOME
)


stopifnot(
  all(
    required_outcome_columns %in%
      names(outcomes_b)
  ),

  nrow(outcomes_b) ==
    N_B,

  n_distinct(
    outcomes_b$RecordID
  ) ==
    N_B,

  !anyDuplicated(
    outcomes_b$RecordID
  ),

  !any(
    is.na(
      outcomes_b$RecordID
    )
  ),

  all(
    outcomes_b[[OUTCOME]] %in%
      c(0, 1)
  )
)


# Predictor and outcome files must refer to exactly the same patients.

stopifnot(
  setequal(
    set_b$RecordID,
    outcomes_b$RecordID
  )
)


cat(
  "Set B deaths:",
  sum(
    outcomes_b[[OUTCOME]] ==
      1
  ),
  "\n"
)

cat(
  "Set B mortality:",
  sprintf(
    "%.1f%%",
    100 *
      mean(
        outcomes_b[[OUTCOME]] ==
          1
      )
  ),
  "\n"
)


# =============================================================================
# 13. Join frozen LASSO predictions to Set B outcomes
# =============================================================================

lasso_test <- set_b_lasso_predictions |>
  left_join(
    outcomes_b |>
      select(
        RecordID,
        all_of(
          OUTCOME
        )
      ),
    by =
      "RecordID"
  ) |>
  transmute(
    RecordID,
    outcome =
      .data[[OUTCOME]],
    model =
      "LASSO",
    probability
  )


stopifnot(
  nrow(
    lasso_test
  ) ==
    N_B,

  !any(
    is.na(
      lasso_test$outcome
    )
  ),

  !any(
    is.na(
      lasso_test$probability
    )
  )
)


# =============================================================================
# 14. Benchmark prediction helper
# =============================================================================
#
# SAPS-I and SOFA must also be evaluated without fitting anything on Set B.
#
# For each benchmark:
#
#   1. Fit the benchmark-to-mortality logistic mapping on ALL Set A.
#   2. Learn any missing-value imputation from Set A only.
#   3. Apply that mapping unchanged to Set B.
#
# This reproduces the benchmark philosophy used in 08.


quote_name <- function(x) {
  paste0(
    "`",
    x,
    "`"
  )
}


make_formula <- function(
  response,
  predictor
) {

  as.formula(
    paste(
      quote_name(
        response
      ),
      "~",
      quote_name(
        predictor
      )
    )
  )
}


fit_benchmark_and_predict <- function(
  benchmark_name
) {

  # ---------------------------------------------------------------------------
  # Set-A benchmark imputation
  # ---------------------------------------------------------------------------

  benchmark_a <- set_a[[benchmark_name]]

  observed_a <- benchmark_a[
    !is.na(benchmark_a) &
      is.finite(benchmark_a)
  ]


  if (
    length(
      observed_a
    ) ==
      0
  ) {

    stop(
      paste(
        "No valid Set-A values for benchmark:",
        benchmark_name
      )
    )
  }


  benchmark_median_a <- median(
    observed_a
  )


  benchmark_a_imputed <- benchmark_a

  benchmark_a_imputed[
    is.na(
      benchmark_a_imputed
    )
  ] <- benchmark_median_a


  benchmark_train <- tibble(
    outcome =
      set_a[[OUTCOME]],

    benchmark =
      benchmark_a_imputed
  )


  benchmark_fit <- glm(
    outcome ~ benchmark,
    data =
      benchmark_train,
    family =
      binomial()
  )


  stopifnot(
    isTRUE(
      benchmark_fit$converged
    )
  )


  # ---------------------------------------------------------------------------
  # Apply Set-A imputation and model to Set B
  # ---------------------------------------------------------------------------

  benchmark_b <- outcomes_b[[benchmark_name]]


  benchmark_b_imputed <- benchmark_b

  benchmark_b_imputed[
    is.na(
      benchmark_b_imputed
    )
  ] <- benchmark_median_a


  benchmark_test_data <- tibble(
    benchmark =
      benchmark_b_imputed
  )


  probability <- as.numeric(
    predict(
      benchmark_fit,
      newdata =
        benchmark_test_data,
      type =
        "response"
    )
  )


  stopifnot(
    length(
      probability
    ) ==
      N_B,

    all(
      is.finite(
        probability
      )
    ),

    all(
      probability >=
        0 &
        probability <=
        1
    )
  )


  tibble(
    RecordID =
      outcomes_b$RecordID,

    outcome =
      outcomes_b[[OUTCOME]],

    model =
      benchmark_name,

    probability =
      probability
  )
}


# =============================================================================
# 15. Generate frozen SAPS-I / SOFA benchmark predictions
# =============================================================================

saps_test <- fit_benchmark_and_predict(
  "SAPS-I"
)


sofa_test <- fit_benchmark_and_predict(
  "SOFA"
)


# =============================================================================
# 16. Combine final Set B predictions
# =============================================================================

all_test_predictions <- bind_rows(
  lasso_test,
  saps_test,
  sofa_test
)


expected_models <- c(
  "LASSO",
  "SAPS-I",
  "SOFA"
)


for (
  model_name in expected_models
) {

  model_predictions <- all_test_predictions |>
    filter(
      model ==
        model_name
    )


  stopifnot(
    nrow(
      model_predictions
    ) ==
      N_B,

    n_distinct(
      model_predictions$RecordID
    ) ==
      N_B,

    !anyDuplicated(
      model_predictions$RecordID
    ),

    all(
      is.finite(
        model_predictions$probability
      )
    )
  )
}


# =============================================================================
# 17. Test-set metric helpers
# =============================================================================

clip_probability <- function(p) {

  pmin(
    pmax(
      p,
      EPS
    ),
    1 - EPS
  )
}


calc_auc <- function(
  y,
  p
) {

  roc_object <- pROC::roc(
    response =
      y,

    predictor =
      p,

    levels =
      c(0, 1),

    direction =
      "<",

    quiet =
      TRUE
  )


  as.numeric(
    pROC::auc(
      roc_object
    )
  )
}


calc_brier <- function(
  y,
  p
) {

  mean(
    (y - p)^2
  )
}


calc_log_loss <- function(
  y,
  p
) {

  p <- clip_probability(
    p
  )


  -mean(
    y *
      log(p) +
      (1 - y) *
        log(1 - p)
  )
}


# =============================================================================
# 18. Calibration helper
# =============================================================================

calculate_calibration <- function(
  predictions,
  model_name
) {

  p <- clip_probability(
    predictions$probability
  )


  calibration_data <- tibble(
    outcome =
      predictions$outcome,

    linear_predictor =
      qlogis(
        p
      )
  )


  # Calibration intercept:
  #
  #   logit(Y) = intercept + offset(logit(p))
  #
  # Ideal = 0.

  intercept_fit <- glm(
    outcome ~ 1,
    data =
      calibration_data,
    family =
      binomial(),
    offset =
      linear_predictor
  )


  # Calibration slope:
  #
  #   logit(Y) = beta0 + beta1 * logit(p)
  #
  # Ideal beta1 = 1.

  slope_fit <- glm(
    outcome ~ linear_predictor,
    data =
      calibration_data,
    family =
      binomial()
  )


  tibble(
    model =
      model_name,

    calibration_intercept =
      unname(
        coef(
          intercept_fit
        )[["(Intercept)"]]
      ),

    calibration_slope =
      unname(
        coef(
          slope_fit
        )[["linear_predictor"]]
      )
  )
}


# =============================================================================
# 19. Evaluate Set B exactly once
# =============================================================================

test_performance <- lapply(
  expected_models,
  function(
    model_name
  ) {

    predictions <- all_test_predictions |>
      filter(
        model ==
          model_name
      )


    tibble(
      model =
        model_name,

      n =
        nrow(
          predictions
        ),

      deaths =
        sum(
          predictions$outcome ==
            1
        ),

      mortality =
        mean(
          predictions$outcome ==
            1
        ),

      auc =
        calc_auc(
          predictions$outcome,
          predictions$probability
        ),

      brier =
        calc_brier(
          predictions$outcome,
          predictions$probability
        ),

      log_loss =
        calc_log_loss(
          predictions$outcome,
          predictions$probability
        )
    )
  }
) |>
  bind_rows() |>
  arrange(
    desc(
      auc
    )
  )


test_calibration <- lapply(
  expected_models,
  function(
    model_name
  ) {

    predictions <- all_test_predictions |>
      filter(
        model ==
          model_name
      )


    calculate_calibration(
      predictions =
        predictions,

      model_name =
        model_name
    )
  }
) |>
  bind_rows()


test_summary <- test_performance |>
  left_join(
    test_calibration,
    by =
      "model"
  ) |>
  arrange(
    desc(
      auc
    )
  )


# =============================================================================
# 20. Calibration bins
# =============================================================================

make_calibration_bins <- function(
  predictions,
  model_name,
  n_bins = 10
) {

  predictions |>
    mutate(
      calibration_bin =
        ntile(
          probability,
          n_bins
        )
    ) |>
    summarise(
      n =
        n(),

      mean_predicted =
        mean(
          probability
        ),

      observed_mortality =
        mean(
          outcome
        ),

      .by =
        calibration_bin
    ) |>
    mutate(
      model =
        model_name
    ) |>
    select(
      model,
      calibration_bin,
      n,
      mean_predicted,
      observed_mortality
    )
}


test_calibration_bins <- lapply(
  expected_models,
  function(
    model_name
  ) {

    predictions <- all_test_predictions |>
      filter(
        model ==
          model_name
      )


    make_calibration_bins(
      predictions =
        predictions,

      model_name =
        model_name
    )
  }
) |>
  bind_rows()


# =============================================================================
# 21. Set B calibration plot
# =============================================================================

calibration_plot <- ggplot(
  test_calibration_bins,
  aes(
    x =
      mean_predicted,

    y =
      observed_mortality,

    group =
      model
  )
) +

  geom_abline(
    intercept =
      0,

    slope =
      1,

    linetype =
      "dashed"
  ) +

  geom_line() +

  geom_point() +

  facet_wrap(
    ~ model,
    ncol =
      3
  ) +

  coord_equal(
    xlim =
      c(0, 1),

    ylim =
      c(0, 1)
  ) +

  labs(
    title =
      "Set B calibration",

    subtitle =
      paste(
        "Single-look held-out test evaluation;",
        "observed versus predicted mortality by risk decile"
      ),

    x =
      "Mean predicted mortality probability",

    y =
      "Observed mortality"
  ) +

  theme_minimal(
    base_size =
      11
  )


ggsave(
  filename =
    here(
      "output",
      "set-b_calibration_plot.png"
    ),

  plot =
    calibration_plot,

  width =
    10,

  height =
    4.5,

  dpi =
    150
)


# =============================================================================
# 22. Compare frozen LASSO development and test performance
# =============================================================================
#
# This comparison is descriptive.
#
# We do NOT use any Set-B discrepancy to revise the model.


development_path <- here(
  "output",
  "model_comparison_final_set_a.csv"
)


if (
  file.exists(
    development_path
  )
) {

  development_results <- read_csv(
    development_path,
    show_col_types =
      FALSE
  ) |>
    filter(
      model ==
        "LASSO"
    ) |>
    transmute(
      dataset =
        "Set A nested CV",

      model,

      auc,

      brier,

      log_loss,

      calibration_intercept,

      calibration_slope
    )


  test_results <- test_summary |>
    filter(
      model ==
        "LASSO"
    ) |>
    transmute(
      dataset =
        "Set B held-out test",

      model,

      auc,

      brier,

      log_loss,

      calibration_intercept,

      calibration_slope
    )


  development_test_comparison <- bind_rows(
    development_results,
    test_results
  )


  write_csv(
    development_test_comparison,
    here(
      "output",
      "lasso_set_a_vs_set_b.csv"
    )
  )


  cat(
    "\nLASSO development vs held-out test:\n"
  )


  print(
    development_test_comparison,
    n = Inf
  )
}


# =============================================================================
# 23. Save frozen preprocessing audit
# =============================================================================

preprocessing_summary <- tibble(
  item =
    c(
      "Raw engineered predictors",
      "Retained raw features",
      "Design-matrix columns",
      "Excluded for >=80% Set-A missingness",
      "Excluded for zero variance in Set A",
      "Frozen alpha",
      "Frozen lambda"
    ),

  value =
    c(
      as.character(
        length(
          all_predictors
        )
      ),

      as.character(
        length(
          selected_features
        )
      ),

      as.character(
        ncol(
          x_a
        )
      ),

      as.character(
        length(
          feature_filter$
            excluded_high_missing
        )
      ),

      as.character(
        length(
          feature_filter$
            excluded_zero_variance
        )
      ),

      as.character(
        FROZEN_ALPHA
      ),

      as.character(
        FROZEN_LAMBDA
      )
    )
)


# =============================================================================
# 24. Save all outputs
# =============================================================================

write_csv(
  all_test_predictions,
  here(
    "output",
    "set-b_predictions_final.csv"
  )
)


write_csv(
  test_summary,
  here(
    "output",
    "set-b_test_performance.csv"
  )
)


write_csv(
  test_calibration,
  here(
    "output",
    "set-b_calibration_summary.csv"
  )
)


write_csv(
  test_calibration_bins,
  here(
    "output",
    "set-b_calibration_bins.csv"
  )
)


write_csv(
  lasso_coefficients,
  here(
    "output",
    "set-b_final_lasso_coefficients.csv"
  )
)


write_csv(
  preprocessing_summary,
  here(
    "output",
    "set-b_final_preprocessing_summary.csv"
  )
)


# =============================================================================
# 25. Final validation
# =============================================================================

stopifnot(
  nrow(
    all_test_predictions
  ) ==
    N_B *
      length(
        expected_models
      ),

  nrow(
    test_summary
  ) ==
    length(
      expected_models
    ),

  all(
    is.finite(
      test_summary$auc
    )
  ),

  all(
    is.finite(
      test_summary$brier
    )
  ),

  all(
    is.finite(
      test_summary$log_loss
    )
  ),

  all(
    is.finite(
      test_summary$
        calibration_intercept
    )
  ),

  all(
    is.finite(
      test_summary$
        calibration_slope
    )
  )
)


# =============================================================================
# 26. Final single-look report
# =============================================================================

cat(
  "\n============================================================\n"
)

cat(
  "FINAL SET B SINGLE-LOOK EVALUATION COMPLETE\n"
)

cat(
  "============================================================\n"
)


cat(
  "\nSet B cohort:\n"
)

cat(
  "  ICU stays:",
  N_B,
  "\n"
)

cat(
  "  Deaths:",
  sum(
    outcomes_b[[OUTCOME]] ==
      1
  ),
  "\n"
)

cat(
  "  Mortality:",
  sprintf(
    "%.1f%%",
    100 *
      mean(
        outcomes_b[[OUTCOME]] ==
          1
      )
  ),
  "\n"
)


cat(
  "\nFrozen primary model:\n"
)

cat(
  "  LASSO logistic regression\n"
)

cat(
  "  alpha =",
  FROZEN_ALPHA,
  "\n"
)

cat(
  "  lambda =",
  FROZEN_LAMBDA,
  "\n"
)


cat(
  "\nHeld-out Set B performance:\n"
)

print(
  test_summary,
  n = Inf
)


cat(
  "\n============================================================\n"
)

cat(
  "MODEL DEVELOPMENT IS CLOSED\n"
)

cat(
  "============================================================\n"
)

cat(
  paste(
    "Set B outcomes have now been examined.",
    "No model, feature, preprocessing, or hyperparameter",
    "changes should be made in response to these test results.\n"
  )
)


cat(
  "\nGenerated outputs:\n"
)

cat(
  "  output/set-b_predictions_final.csv\n"
)

cat(
  "  output/set-b_test_performance.csv\n"
)

cat(
  "  output/set-b_calibration_summary.csv\n"
)

cat(
  "  output/set-b_calibration_bins.csv\n"
)

cat(
  "  output/set-b_calibration_plot.png\n"
)

cat(
  "  output/set-b_final_lasso_coefficients.csv\n"
)

cat(
  "  output/set-b_final_preprocessing_summary.csv\n"
)

if (
  exists(
    "development_test_comparison"
  )
) {

  cat(
    "  output/lasso_set_a_vs_set_b.csv
    \n"
  )
}


cat(
  "\nSingle-look Set B evaluation finished successfully.\n"
)
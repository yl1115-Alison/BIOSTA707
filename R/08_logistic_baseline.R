# BIOSTAT 707 Checkpoint 2
# 08_logistic_baseline.R
#
# Purpose:
#   Cross-validate:
#     1. SAPS-I benchmark
#     2. SOFA benchmark
#     3. Clinically motivated logistic-regression baseline
#
# Leakage control:
#   All learned preprocessing parameters are estimated using only
#   the training portion of each CV fold and then applied unchanged
#   to the corresponding validation fold.
#
# Inputs:
#   output/set-a_features.csv
#   output/set-a_cv_folds.csv
#
# Outputs:
#   output/logistic_baseline_features.csv
#   output/cv_predictions_benchmarks_logistic.csv
#   output/cv_performance_benchmarks_logistic.csv
#   output/cv_performance_by_fold.csv
#   output/cv_performance_fold_summary.csv
#   output/logistic_baseline_coefficients.csv
#   output/logistic_baseline_imputation_values.csv
#   output/logistic_baseline_categorical_imputation.csv


# =============================================================================
# 1. Packages and constants
# =============================================================================

library(readr)
library(dplyr)
library(here)
library(pROC)

N_STAYS <- 4000
N_FOLDS <- 5

OUTCOME <- "In-hospital_death"


# =============================================================================
# 2. Formula helpers
# =============================================================================
#
# Some Challenge 2012 variables contain hyphens:
#
#   In-hospital_death
#   SAPS-I
#
# These are non-syntactic R variable names and therefore must be quoted
# with backticks when used inside model formulas.


quote_name <- function(x) {

  paste0(
    "`",
    x,
    "`"
  )
}


make_formula <- function(
  response,
  predictors
) {

  as.formula(
    paste(
      quote_name(
        response
      ),
      "~",
      paste(
        quote_name(
          predictors
        ),
        collapse = " + "
      )
    )
  )
}


# =============================================================================
# 3. Load Set A and fixed folds
# =============================================================================

set_a_path <- here(
  "output",
  "set-a_features.csv"
)

fold_path <- here(
  "output",
  "set-a_cv_folds.csv"
)

stopifnot(
  file.exists(set_a_path),
  file.exists(fold_path)
)


dat <- read_csv(
  set_a_path,
  show_col_types = FALSE
)

folds <- read_csv(
  fold_path,
  show_col_types = FALSE
)


stopifnot(
  nrow(dat) == N_STAYS,
  nrow(folds) == N_STAYS,

  n_distinct(dat$RecordID) == N_STAYS,
  n_distinct(folds$RecordID) == N_STAYS,

  !anyDuplicated(dat$RecordID),
  !anyDuplicated(folds$RecordID),

  all(
    folds$fold %in%
      seq_len(N_FOLDS)
  ),

  OUTCOME %in%
    names(dat),

  all(
    dat[[OUTCOME]] %in%
      c(0, 1)
  )
)


dat <- dat |>
  left_join(
    folds,
    by = "RecordID"
  )


stopifnot(
  !any(
    is.na(
      dat$fold
    )
  )
)


cat(
  "Set A rows:",
  nrow(dat),
  "\n"
)

cat(
  "Deaths:",
  sum(
    dat[[OUTCOME]] == 1
  ),
  "\n"
)

cat(
  "Mortality:",
  sprintf(
    "%.1f%%",
    100 *
      mean(
        dat[[OUTCOME]] == 1
      )
  ),
  "\n"
)


stopifnot(
  sum(
    dat[[OUTCOME]] == 1
  ) ==
    554
)


# =============================================================================
# 4. Define compact logistic baseline
# =============================================================================
#
# This feature set is specified before inspecting model performance.
#
# The ordinary logistic baseline intentionally uses a compact set of
# clinically interpretable predictors rather than all 313 engineered
# features.
#
# The broader engineered feature space will be handled by regularized
# models in the next stage.


baseline_continuous <- c(

  # Baseline characteristics
  "Age",
  "Height",
  "Weight_first",

  # Neurologic status
  "GCS_min",
  "GCS_delta",

  # Heart rate and temperature
  "HR_mean",
  "HR_max",
  "HR_slope",
  "Temp_mean",
  "Temp_max",

  # Blood pressure
  "MAP_mean",
  "MAP_min",
  "NIMAP_mean",
  "NIMAP_min",

  # Respiratory status
  "FiO2_max",
  "SaO2_min",

  # Renal / metabolic
  "Creatinine_mean",
  "Creatinine_delta",
  "BUN_mean",
  "Glucose_mean",

  # Acid-base / perfusion
  "pH_min",
  "Lactate_max",
  "Lactate_delta",

  # Hematologic
  "WBC_mean",
  "Platelets_min",

  # Fluid balance
  "Urine_total"
)


baseline_binary <- c(
  "Gender",
  "MechVent_ever",

  # Explicit missingness indicators
  "Lactate_missing",
  "SaO2_missing",
  "FiO2_missing"
)


baseline_categorical <- c(
  "ICUType"
)


baseline_features <- c(
  baseline_continuous,
  baseline_binary,
  baseline_categorical
)


missing_baseline_features <- setdiff(
  baseline_features,
  names(dat)
)


if (
  length(
    missing_baseline_features
  ) >
    0
) {

  stop(
    paste(
      "Baseline features absent from feature table:",
      paste(
        missing_baseline_features,
        collapse = ", "
      )
    )
  )
}


baseline_dictionary <- tibble(
  feature =
    baseline_features,

  role =
    case_when(

      feature %in%
        baseline_continuous ~
        "continuous",

      feature %in%
        baseline_binary ~
        "binary",

      feature %in%
        baseline_categorical ~
        "categorical",

      TRUE ~
        "other"
    )
)


write_csv(
  baseline_dictionary,
  here(
    "output",
    "logistic_baseline_features.csv"
  )
)


cat(
  "Compact logistic baseline predictors:",
  length(
    baseline_features
  ),
  "\n"
)


# =============================================================================
# 5. Metric helpers
# =============================================================================

clip_probability <- function(p) {

  pmin(
    pmax(
      p,
      1e-6
    ),
    1 - 1e-6
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


calculate_metrics <- function(
  predictions,
  model_name,
  fold_value = NA_integer_
) {

  tibble(
    model =
      model_name,

    fold =
      fold_value,

    n =
      nrow(
        predictions
      ),

    deaths =
      sum(
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


# =============================================================================
# 6. Fold-specific median imputation
# =============================================================================
#
# Medians are learned from the training fold only.
#
# The validation fold never contributes to its own imputation parameters.


fit_median_imputer <- function(
  training_data,
  variables
) {

  medians <- vapply(
    variables,
    function(variable) {

      x <- training_data[[variable]]

      x <- x[
        !is.na(x) &
          is.finite(x)
      ]

      if (length(x) == 0) {
        return(NA_real_)
      }

      median(x)
    },
    numeric(1)
  )

  if (any(is.na(medians))) {

    bad <- names(medians)[
      is.na(medians)
    ]

    stop(
      paste(
        "Cannot estimate training-fold median for:",
        paste(
          bad,
          collapse = ", "
        )
      )
    )
  }

  medians
}


apply_median_imputer <- function(
  data,
  medians
) {

  output <- data

  for (variable in names(medians)) {

    missing <- is.na(
      output[[variable]]
    )

    output[[variable]][missing] <-
      medians[[variable]]
  }

  output
}


# =============================================================================
# 7. Fold-specific mode imputation
# =============================================================================

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


apply_mode_imputation <- function(
  data,
  variable,
  mode_value
) {

  output <- data

  missing <- is.na(
    output[[variable]]
  )

  output[[variable]][missing] <-
    as.numeric(mode_value)

  output
}


# =============================================================================
# 8. Categorical handling
# =============================================================================

prepare_categorical <- function(
  data
) {

  data |>
    mutate(
      ICUType =
        factor(
          ICUType,
          levels =
            1:4
        )
    )
}


# =============================================================================
# 9. Model formulas
# =============================================================================

baseline_formula <- make_formula(
  response =
    OUTCOME,

  predictors =
    baseline_features
)


cat(
  "\nBaseline logistic formula:\n"
)

print(
  baseline_formula
)


# =============================================================================
# 10. Generic held-out-fold prediction function
# =============================================================================

predict_one_fold <- function(
  fold_number,
  model_name,
  model_type
) {

  train <- dat |>
    filter(
      fold !=
        fold_number
    )


  validation <- dat |>
    filter(
      fold ==
        fold_number
    )


  stopifnot(
    nrow(train) +
      nrow(validation) ==
      N_STAYS,

    length(
      intersect(
        train$RecordID,
        validation$RecordID
      )
    ) ==
      0
  )


  # ---------------------------------------------------------------------------
  # SAPS-I benchmark
  # ---------------------------------------------------------------------------

  if (
    model_type ==
      "saps"
  ) {

    variable <-
      "SAPS-I"


    medians <- fit_median_imputer(
      training_data =
        train,

      variables =
        variable
    )


    train <- apply_median_imputer(
      data =
        train,

      medians =
        medians
    )


    validation <- apply_median_imputer(
      data =
        validation,

      medians =
        medians
    )


    benchmark_formula <- make_formula(
      response =
        OUTCOME,

      predictors =
        variable
    )


    fit <- glm(
      benchmark_formula,
      data =
        train,
      family =
        binomial()
    )
  }


  # ---------------------------------------------------------------------------
  # SOFA benchmark
  # ---------------------------------------------------------------------------

  if (
    model_type ==
      "sofa"
  ) {

    variable <-
      "SOFA"


    medians <- fit_median_imputer(
      training_data =
        train,

      variables =
        variable
    )


    train <- apply_median_imputer(
      data =
        train,

      medians =
        medians
    )


    validation <- apply_median_imputer(
      data =
        validation,

      medians =
        medians
    )


    benchmark_formula <- make_formula(
      response =
        OUTCOME,

      predictors =
        variable
    )


    fit <- glm(
      benchmark_formula,
      data =
        train,
      family =
        binomial()
    )
  }


  # ---------------------------------------------------------------------------
  # Compact logistic baseline
  # ---------------------------------------------------------------------------

  if (
    model_type ==
      "baseline"
  ) {

    # -------------------------------------------------------------------------
    # Continuous variables:
    # training-fold median imputation
    # -------------------------------------------------------------------------

    medians <- fit_median_imputer(
      training_data =
        train,

      variables =
        baseline_continuous
    )


    train <- apply_median_imputer(
      data =
        train,

      medians =
        medians
    )


    validation <- apply_median_imputer(
      data =
        validation,

      medians =
        medians
    )


    # -------------------------------------------------------------------------
    # Gender:
    # training-fold mode imputation
    # -------------------------------------------------------------------------

    gender_mode <- fit_mode(
      train$Gender
    )


    train <- apply_mode_imputation(
      data =
        train,

      variable =
        "Gender",

      mode_value =
        gender_mode
    )


    validation <- apply_mode_imputation(
      data =
        validation,

      variable =
        "Gender",

      mode_value =
        gender_mode
    )


    # -------------------------------------------------------------------------
    # ICUType:
    # training-fold mode imputation
    # -------------------------------------------------------------------------

    icu_mode <- fit_mode(
      train$ICUType
    )


    train <- apply_mode_imputation(
      data =
        train,

      variable =
        "ICUType",

      mode_value =
        icu_mode
    )


    validation <- apply_mode_imputation(
      data =
        validation,

      variable =
        "ICUType",

      mode_value =
        icu_mode
    )


    # -------------------------------------------------------------------------
    # Convert ICUType to a categorical factor
    # -------------------------------------------------------------------------

    train <- prepare_categorical(
      train
    )


    validation <- prepare_categorical(
      validation
    )


    # -------------------------------------------------------------------------
    # Fit model
    # -------------------------------------------------------------------------

    fit <- glm(
      baseline_formula,
      data =
        train,
      family =
        binomial()
    )
  }


  # ---------------------------------------------------------------------------
  # Validate model fit
  # ---------------------------------------------------------------------------

  if (
    !isTRUE(
      fit$converged
    )
  ) {

    warning(
      paste(
        model_name,
        "did not converge in fold",
        fold_number
      )
    )
  }


  # ---------------------------------------------------------------------------
  # Predict held-out fold
  # ---------------------------------------------------------------------------

  probability <- predict(
    fit,
    newdata =
      validation,
    type =
      "response"
  )


  stopifnot(
    length(
      probability
    ) ==
      nrow(
        validation
      ),

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
      validation$RecordID,

    fold =
      fold_number,

    outcome =
      validation[[OUTCOME]],

    model =
      model_name,

    probability =
      as.numeric(
        probability
      )
  )
}


# =============================================================================
# 11. Cross-validated SAPS-I benchmark
# =============================================================================

cat(
  "\nRunning SAPS-I benchmark CV...\n"
)


pred_saps <- lapply(
  seq_len(
    N_FOLDS
  ),
  function(k) {

    cat(
      "  SAPS-I fold",
      k,
      "of",
      N_FOLDS,
      "\n"
    )


    predict_one_fold(
      fold_number =
        k,

      model_name =
        "SAPS-I",

      model_type =
        "saps"
    )
  }
) |>
  bind_rows()


# =============================================================================
# 12. Cross-validated SOFA benchmark
# =============================================================================

cat(
  "\nRunning SOFA benchmark CV...\n"
)


pred_sofa <- lapply(
  seq_len(
    N_FOLDS
  ),
  function(k) {

    cat(
      "  SOFA fold",
      k,
      "of",
      N_FOLDS,
      "\n"
    )


    predict_one_fold(
      fold_number =
        k,

      model_name =
        "SOFA",

      model_type =
        "sofa"
    )
  }
) |>
  bind_rows()


# =============================================================================
# 13. Cross-validated compact logistic baseline
# =============================================================================

cat(
  "\nRunning compact logistic baseline CV...\n"
)


pred_logistic <- lapply(
  seq_len(
    N_FOLDS
  ),
  function(k) {

    cat(
      "  Logistic fold",
      k,
      "of",
      N_FOLDS,
      "\n"
    )


    predict_one_fold(
      fold_number =
        k,

      model_name =
        "Logistic baseline",

      model_type =
        "baseline"
    )
  }
) |>
  bind_rows()


# =============================================================================
# 14. Combine and validate out-of-fold predictions
# =============================================================================

all_predictions <- bind_rows(
  pred_saps,
  pred_sofa,
  pred_logistic
)


expected_models <- c(
  "SAPS-I",
  "SOFA",
  "Logistic baseline"
)


for (
  model_name in expected_models
) {

  model_predictions <-

    model_predictions <- all_predictions |>
    filter(
      model ==
        model_name
    )


  stopifnot(
    nrow(
      model_predictions
    ) ==
      N_STAYS,

    n_distinct(
      model_predictions$RecordID
    ) ==
      N_STAYS,

    !anyDuplicated(
      model_predictions$RecordID
    ),

    !any(
      is.na(
        model_predictions$probability
      )
    )
  )
}


# =============================================================================
# 15. Overall out-of-fold performance
# =============================================================================
#
# Each patient contributes exactly one prediction from a model fitted without
# that patient. These pooled held-out predictions provide the main Set A
# development-performance summary.


overall_performance <- lapply(
  expected_models,
  function(
    model_name
  ) {

    predictions <- all_predictions |>
      filter(
        model ==
          model_name
      )


    calculate_metrics(
      predictions =
        predictions,

      model_name =
        model_name
    )
  }
) |>
  bind_rows() |>
  arrange(
    desc(
      auc
    )
  )


cat(
  "\nOverall out-of-fold performance:\n"
)

print(
  overall_performance,
  n = Inf
)


# =============================================================================
# 16. Performance within each validation fold
# =============================================================================

performance_by_fold <- lapply(
  expected_models,
  function(
    model_name
  ) {

    lapply(
      seq_len(
        N_FOLDS
      ),
      function(k) {

        predictions <- all_predictions |>
          filter(
            model ==
              model_name,
            fold ==
              k
          )


        calculate_metrics(
          predictions =
            predictions,

          model_name =
            model_name,

          fold_value =
            k
        )
      }
    ) |>
      bind_rows()
  }
) |>
  bind_rows() |>
  arrange(
    model,
    fold
  )


cat(
  "\nPerformance by fold:\n"
)

print(
  performance_by_fold,
  n = Inf
)


# =============================================================================
# 17. Summarize fold-to-fold variability
# =============================================================================

performance_fold_summary <- performance_by_fold |>
  summarise(

    mean_auc =
      mean(
        auc,
        na.rm = TRUE
      ),

    sd_auc =
      sd(
        auc,
        na.rm = TRUE
      ),

    mean_brier =
      mean(
        brier,
        na.rm = TRUE
      ),

    sd_brier =
      sd(
        brier,
        na.rm = TRUE
      ),

    mean_log_loss =
      mean(
        log_loss,
        na.rm = TRUE
      ),

    sd_log_loss =
      sd(
        log_loss,
        na.rm = TRUE
      ),

    .by =
      model
  ) |>
  arrange(
    desc(
      mean_auc
    )
  )


cat(
  "\nMean fold performance:\n"
)

print(
  performance_fold_summary,
  n = Inf
)


# =============================================================================
# 18. Fit final compact logistic baseline on all Set A
# =============================================================================
#
# This final Set-A fit is used only to:
#
#   - document the baseline model
#   - inspect coefficients
#   - save the preprocessing parameters that would later be applied to Set B
#
# It is NOT used to estimate honest predictive performance.
#
# Predictive performance above comes from held-out out-of-fold predictions.


final_train <- dat


# =============================================================================
# 19. Final Set-A continuous-variable imputation
# =============================================================================
#
# Once model development is complete, fitting preprocessing on all Set A is
# appropriate for the final development-set model.
#
# These Set-A-derived values would later be applied unchanged to Set B.


final_medians <- fit_median_imputer(
  training_data =
    final_train,

  variables =
    baseline_continuous
)


final_train <- apply_median_imputer(
  data =
    final_train,

  medians =
    final_medians
)


# =============================================================================
# 20. Final Set-A categorical imputation
# =============================================================================

gender_mode <- fit_mode(
  final_train$Gender
)


icu_mode <- fit_mode(
  final_train$ICUType
)


final_train <- apply_mode_imputation(
  data =
    final_train,

  variable =
    "Gender",

  mode_value =
    gender_mode
)


final_train <- apply_mode_imputation(
  data =
    final_train,

  variable =
    "ICUType",

  mode_value =
    icu_mode
)


final_train <- prepare_categorical(
  final_train
)


# =============================================================================
# 21. Fit final Set-A logistic baseline
# =============================================================================

final_logistic_fit <- glm(
  baseline_formula,
  data =
    final_train,
  family =
    binomial()
)


if (
  !isTRUE(
    final_logistic_fit$converged
  )
) {

  warning(
    "Final compact logistic model did not converge."
  )
}


cat(
  "\nFinal compact logistic model converged:",
  final_logistic_fit$converged,
  "\n"
)


# =============================================================================
# 22. Extract final logistic coefficients
# =============================================================================
#
# These coefficient summaries are descriptive for the prespecified baseline.
#
# They should not be used to perform data-driven feature selection for the
# subsequent models.


coefficient_matrix <- summary(
  final_logistic_fit
)$coefficients


logistic_coefficients <- tibble(

  term =
    rownames(
      coefficient_matrix
    ),

  estimate =
    coefficient_matrix[
      ,
      "Estimate"
    ],

  standard_error =
    coefficient_matrix[
      ,
      "Std. Error"
    ],

  z_value =
    coefficient_matrix[
      ,
      "z value"
    ],

  p_value =
    coefficient_matrix[
      ,
      "Pr(>|z|)"
    ]
) |>
  mutate(

    odds_ratio =
      exp(
        estimate
      ),

    conf_low_95 =
      exp(
        estimate -
          1.96 *
            standard_error
      ),

    conf_high_95 =
      exp(
        estimate +
          1.96 *
            standard_error
      )
  )


# =============================================================================
# 23. Save final Set-A preprocessing parameters
# =============================================================================

final_imputation_values <- tibble(

  feature =
    names(
      final_medians
    ),

  median_set_a =
    as.numeric(
      final_medians
    )
)


final_categorical_imputation <- tibble(

  feature =
    c(
      "Gender",
      "ICUType"
    ),

  mode_set_a =
    c(
      as.numeric(
        gender_mode
      ),

      as.numeric(
        icu_mode
      )
    )
)


# =============================================================================
# 24. Save outputs
# =============================================================================

write_csv(
  all_predictions,
  here(
    "output",
    "cv_predictions_benchmarks_logistic.csv"
  )
)


write_csv(
  overall_performance,
  here(
    "output",
    "cv_performance_benchmarks_logistic.csv"
  )
)


write_csv(
  performance_by_fold,
  here(
    "output",
    "cv_performance_by_fold.csv"
  )
)


write_csv(
  performance_fold_summary,
  here(
    "output",
    "cv_performance_fold_summary.csv"
  )
)


write_csv(
  logistic_coefficients,
  here(
    "output",
    "logistic_baseline_coefficients.csv"
  )
)


write_csv(
  final_imputation_values,
  here(
    "output",
    "logistic_baseline_imputation_values.csv"
  )
)


write_csv(
  final_categorical_imputation,
  here(
    "output",
    "logistic_baseline_categorical_imputation.csv"
  )
)


# =============================================================================
# 25. Re-read and validate saved predictions
# =============================================================================

saved_predictions <- read_csv(
  here(
    "output",
    "cv_predictions_benchmarks_logistic.csv"
  ),
  show_col_types = FALSE
)


stopifnot(
  nrow(
    saved_predictions
  ) ==
    N_STAYS *
      length(
        expected_models
      ),

  all(
    expected_models %in%
      unique(
        saved_predictions$model
      )
  )
)


prediction_count_check <- saved_predictions |>
  count(
    RecordID,
    model,
    name =
      "n_predictions"
  )


stopifnot(
  all(
    prediction_count_check$n_predictions ==
      1
  ),

  nrow(
    prediction_count_check
  ) ==
    N_STAYS *
      length(
        expected_models
      )
)


# =============================================================================
# 26. Additional probability validation
# =============================================================================

stopifnot(
  all(
    saved_predictions$probability >=
      0 &
      saved_predictions$probability <=
      1
  ),

  !any(
    is.na(
      saved_predictions$probability
    )
  ),

  !any(
    !is.finite(
      saved_predictions$probability
    )
  )
)


# =============================================================================
# 27. Final report
# =============================================================================

cat(
  "\n============================================================\n"
)

cat(
  "Checkpoint 2 benchmark / logistic baseline complete.\n"
)

cat(
  "============================================================\n"
)

cat(
  "Set A ICU stays:",
  N_STAYS,
  "\n"
)

cat(
  "Cross-validation folds:",
  N_FOLDS,
  "\n"
)

cat(
  "Compact logistic predictors:",
  length(
    baseline_features
  ),
  "\n"
)


cat(
  "\nOverall out-of-fold performance:\n"
)

print(
  overall_performance |>
    select(
      model,
      auc,
      brier,
      log_loss
    ),
  n = Inf
)


cat(
  "\nFold-averaged performance:\n"
)

print(
  performance_fold_summary,
  n = Inf
)


cat(
  "\nImportant leakage safeguards:\n"
)

cat(
  paste(
    "  1. Every model used the fixed folds from",
    "07_define_resampling.R.\n"
  )
)

cat(
  paste(
    "  2. Continuous-variable imputation was estimated",
    "using the training portion of each fold only.\n"
  )
)

cat(
  paste(
    "  3. Gender and ICUType imputation used the",
    "training-fold mode only.\n"
  )
)

cat(
  paste(
    "  4. Validation observations never contributed",
    "to their own preprocessing parameters.\n"
  )
)

cat(
  "  5. Set B was not accessed.\n"
)

cat(
  paste(
    "  6. SAPS-I and SOFA were treated as separate",
    "clinical benchmarks and were not included in the",
    "engineered-feature logistic baseline.\n"
  )
)


cat(
  "\nGenerated outputs:\n"
)

cat(
  "  output/logistic_baseline_features.csv\n"
)

cat(
  "  output/cv_predictions_benchmarks_logistic.csv\n"
)

cat(
  "  output/cv_performance_benchmarks_logistic.csv\n"
)

cat(
  "  output/cv_performance_by_fold.csv\n"
)

cat(
  "  output/cv_performance_fold_summary.csv\n"
)

cat(
  "  output/logistic_baseline_coefficients.csv\n"
)

cat(
  "  output/logistic_baseline_imputation_values.csv\n"
)

cat(
  "  output/logistic_baseline_categorical_imputation.csv\n"
)
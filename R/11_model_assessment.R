# BIOSTAT 707 Checkpoint 2
# 11_model_assessment.R
#
# Purpose:
#   Final Set-A model assessment before freezing the Checkpoint 2 pipeline.
#
# Tasks:
#   1. Examine clinically plausible nonlinear effects using a prespecified
#      natural-spline logistic regression.
#   2. Generate out-of-fold predictions for that nonlinear logistic model.
#   3. Combine OOF predictions from all primary models.
#   4. Evaluate discrimination, probability accuracy, and calibration.
#   5. Produce calibration plots and a unified model-comparison table.
#
# IMPORTANT:
#   - Set B is NOT accessed.
#   - XGBoost 1000-round analysis remains a sensitivity analysis and is
#     not treated as a primary model.
#   - No new model family or open-ended hyperparameter search is introduced.
#
# Inputs:
#   output/set-a_features.csv
#   output/set-a_cv_folds.csv
#   output/cv_predictions_benchmarks_logistic.csv
#   output/cv_predictions_regularized.csv
#   output/cv_predictions_tree_boosting.csv
#   output/xgb_round_sensitivity_predictions.csv   [optional sensitivity]
#
# Outputs:
#   output/cv_predictions_nonlinear_logistic.csv
#   output/cv_performance_nonlinear_logistic.csv
#   output/model_assessment_all_predictions.csv
#   output/model_assessment_metrics.csv
#   output/model_calibration_summary.csv
#   output/model_calibration_bins.csv
#   output/model_calibration_plot.png
#   output/model_comparison_final_set_a.csv
#   output/nonlinear_logistic_coefficients_set_a.csv


# =============================================================================
# 1. Packages and constants
# =============================================================================

library(readr)
library(dplyr)
library(tidyr)
library(ggplot2)
library(splines)
library(pROC)
library(here)

N_STAYS <- 4000
N_FOLDS <- 5

OUTCOME <- "In-hospital_death"

EPS <- 1e-6


# =============================================================================
# 2. Helper for non-syntactic R names
# =============================================================================

quote_name <- function(x) {
  paste0("`", x, "`")
}


# =============================================================================
# 3. Load Set A and fixed folds
# =============================================================================

dat <- read_csv(
  here(
    "output",
    "set-a_features.csv"
  ),
  show_col_types = FALSE
)

folds <- read_csv(
  here(
    "output",
    "set-a_cv_folds.csv"
  ),
  show_col_types = FALSE
)


stopifnot(
  nrow(dat) == N_STAYS,
  nrow(folds) == N_STAYS,
  !anyDuplicated(dat$RecordID),
  !anyDuplicated(folds$RecordID),
  OUTCOME %in% names(dat),
  all(dat[[OUTCOME]] %in% c(0, 1))
)


dat <- dat |>
  left_join(
    folds,
    by = "RecordID"
  )


stopifnot(
  !any(is.na(dat$fold))
)


cat(
  "Set A ICU stays:",
  nrow(dat),
  "\n"
)

cat(
  "Deaths:",
  sum(dat[[OUTCOME]] == 1),
  "\n"
)


# =============================================================================
# 4. Prespecified compact logistic feature set
# =============================================================================
#
# This matches the compact logistic baseline used in 08.


baseline_continuous <- c(
  "Age",
  "Height",
  "Weight_first",

  "GCS_min",
  "GCS_delta",

  "HR_mean",
  "HR_max",
  "HR_slope",

  "Temp_mean",
  "Temp_max",

  "MAP_mean",
  "MAP_min",
  "NIMAP_mean",
  "NIMAP_min",

  "FiO2_max",
  "SaO2_min",

  "Creatinine_mean",
  "Creatinine_delta",
  "BUN_mean",
  "Glucose_mean",

  "pH_min",
  "Lactate_max",
  "Lactate_delta",

  "WBC_mean",
  "Platelets_min",

  "Urine_total"
)


baseline_binary <- c(
  "Gender",
  "MechVent_ever",
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


stopifnot(
  all(
    baseline_features %in%
      names(dat)
  )
)


# =============================================================================
# 5. Prespecified nonlinear terms
# =============================================================================
#
# These variables have clinically plausible nonlinear relationships with
# mortality and are examined using natural cubic splines with df = 3.
#
# This specification is fixed before evaluating the nonlinear model.


nonlinear_features <- c(
  "Age",
  "HR_mean",
  "MAP_min",
  "Creatinine_mean",
  "Lactate_max"
)


SPLINE_DF <- 3


stopifnot(
  all(
    nonlinear_features %in%
      baseline_continuous
  )
)


linear_features <- setdiff(
  baseline_features,
  nonlinear_features
)


# =============================================================================
# 6. Build nonlinear logistic formula
# =============================================================================

spline_terms <- paste0(
  "splines::ns(",
  quote_name(nonlinear_features),
  ", df = ",
  SPLINE_DF,
  ")"
)


linear_terms <- quote_name(
  linear_features
)


nonlinear_formula <- as.formula(
  paste(
    quote_name(OUTCOME),
    "~",
    paste(
      c(
        spline_terms,
        linear_terms
      ),
      collapse = " + "
    )
  )
)


cat(
  "\nNonlinear logistic formula:\n"
)

print(
  nonlinear_formula
)


# =============================================================================
# 7. Leakage-safe imputation helpers
# =============================================================================

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
        "Cannot estimate median for:",
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


fit_mode <- function(x) {

  x <- x[
    !is.na(x)
  ]

  if (length(x) == 0) {
    stop(
      "Cannot estimate mode from an entirely missing variable."
    )
  }

  tab <- table(x)

  names(tab)[
    which.max(tab)
  ]
}


apply_mode <- function(
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


prepare_categorical <- function(data) {

  data |>
    mutate(
      ICUType = factor(
        ICUType,
        levels = 1:4
      )
    )
}


# =============================================================================
# 8. Metrics
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
    response = y,
    predictor = p,
    levels = c(0, 1),
    direction = "<",
    quiet = TRUE
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

  p <- clip_probability(p)

  -mean(
    y * log(p) +
      (1 - y) *
        log(1 - p)
  )
}


# =============================================================================
# 9. Fit nonlinear logistic model within one held-out fold
# =============================================================================

predict_nonlinear_fold <- function(
  fold_number
) {

  train <- dat |>
    filter(
      fold != fold_number
    )

  validation <- dat |>
    filter(
      fold == fold_number
    )


  stopifnot(
    length(
      intersect(
        train$RecordID,
        validation$RecordID
      )
    ) == 0
  )


  # ---------------------------------------------------------------------------
  # Continuous-variable imputation from training fold only
  # ---------------------------------------------------------------------------

  medians <- fit_median_imputer(
    training_data = train,
    variables = baseline_continuous
  )


  train <- apply_median_imputer(
    train,
    medians
  )


  validation <- apply_median_imputer(
    validation,
    medians
  )


  # ---------------------------------------------------------------------------
  # Gender / ICUType mode imputation from training fold only
  # ---------------------------------------------------------------------------

  gender_mode <- fit_mode(
    train$Gender
  )

  icu_mode <- fit_mode(
    train$ICUType
  )


  train <- apply_mode(
    train,
    "Gender",
    gender_mode
  )

  validation <- apply_mode(
    validation,
    "Gender",
    gender_mode
  )


  train <- apply_mode(
    train,
    "ICUType",
    icu_mode
  )

  validation <- apply_mode(
    validation,
    "ICUType",
    icu_mode
  )


  train <- prepare_categorical(
    train
  )

  validation <- prepare_categorical(
    validation
  )


  # ---------------------------------------------------------------------------
  # Fit nonlinear logistic model
  # ---------------------------------------------------------------------------

  fit <- glm(
    nonlinear_formula,
    data = train,
    family = binomial()
  )


  if (!isTRUE(fit$converged)) {

    warning(
      paste(
        "Nonlinear logistic model did not converge in fold",
        fold_number
      )
    )
  }


  probability <- predict(
    fit,
    newdata = validation,
    type = "response"
  )


  stopifnot(
    length(probability) ==
      nrow(validation),

    all(
      is.finite(probability)
    ),

    all(
      probability >= 0 &
        probability <= 1
    )
  )


  tibble(
    RecordID = validation$RecordID,
    fold = fold_number,
    outcome = validation[[OUTCOME]],
    model = "Nonlinear logistic",
    probability = as.numeric(probability)
  )
}


# =============================================================================
# 10. Generate nonlinear-logistic OOF predictions
# =============================================================================

cat(
  "\nRunning nonlinear logistic cross-validation...\n"
)


nonlinear_predictions <- lapply(
  seq_len(N_FOLDS),
  function(k) {

    cat(
      "  Fold",
      k,
      "of",
      N_FOLDS,
      "\n"
    )

    predict_nonlinear_fold(
      k
    )
  }
) |>
  bind_rows()


stopifnot(
  nrow(nonlinear_predictions) ==
    N_STAYS,

  n_distinct(
    nonlinear_predictions$RecordID
  ) ==
    N_STAYS,

  !anyDuplicated(
    nonlinear_predictions$RecordID
  )
)


# =============================================================================
# 11. Nonlinear-logistic performance
# =============================================================================

nonlinear_performance <- tibble(
  model = "Nonlinear logistic",
  fold = NA_integer_,
  n = N_STAYS,
  deaths = sum(
    nonlinear_predictions$outcome == 1
  ),
  auc = calc_auc(
    nonlinear_predictions$outcome,
    nonlinear_predictions$probability
  ),
  brier = calc_brier(
    nonlinear_predictions$outcome,
    nonlinear_predictions$probability
  ),
  log_loss = calc_log_loss(
    nonlinear_predictions$outcome,
    nonlinear_predictions$probability
  )
)


nonlinear_fold_performance <- lapply(
  seq_len(N_FOLDS),
  function(k) {

    p <- nonlinear_predictions |>
      filter(
        fold == k
      )

    tibble(
      model = "Nonlinear logistic",
      fold = k,
      n = nrow(p),
      deaths = sum(
        p$outcome == 1
      ),
      auc = calc_auc(
        p$outcome,
        p$probability
      ),
      brier = calc_brier(
        p$outcome,
        p$probability
      ),
      log_loss = calc_log_loss(
        p$outcome,
        p$probability
      )
    )
  }
) |>
  bind_rows()


cat(
  "\nNonlinear logistic OOF performance:\n"
)

print(
  nonlinear_performance,
  n = Inf
)


# =============================================================================
# 12. Load OOF predictions from previous primary models
# =============================================================================

benchmark_predictions <- read_csv(
  here(
    "output",
    "cv_predictions_benchmarks_logistic.csv"
  ),
  show_col_types = FALSE
)


regularized_predictions <- read_csv(
  here(
    "output",
    "cv_predictions_regularized.csv"
  ),
  show_col_types = FALSE
)


tree_predictions <- read_csv(
  here(
    "output",
    "cv_predictions_tree_boosting.csv"
  ),
  show_col_types = FALSE
)


required_prediction_columns <- c(
  "RecordID",
  "fold",
  "outcome",
  "model",
  "probability"
)


stopifnot(
  all(
    required_prediction_columns %in%
      names(benchmark_predictions)
  ),

  all(
    required_prediction_columns %in%
      names(regularized_predictions)
  ),

  all(
    required_prediction_columns %in%
      names(tree_predictions)
  )
)


# =============================================================================
# 13. Combine PRIMARY models
# =============================================================================
#
# The 1000-round XGBoost sensitivity analysis is deliberately NOT included
# in the primary model ranking.


primary_predictions <- bind_rows(

  benchmark_predictions |>
    select(
      all_of(
        required_prediction_columns
      )
    ),

  nonlinear_predictions |>
    select(
      all_of(
        required_prediction_columns
      )
    ),

  regularized_predictions |>
    select(
      all_of(
        required_prediction_columns
      )
    ),

  tree_predictions |>
    select(
      all_of(
        required_prediction_columns
      )
    )
)


primary_models <- unique(
  primary_predictions$model
)


cat(
  "\nPrimary models included in final assessment:\n"
)

print(
  primary_models
)


# =============================================================================
# 14. Validate OOF prediction structure
# =============================================================================

prediction_counts <- primary_predictions |>
  count(
    model,
    RecordID,
    name = "n_predictions"
  )


stopifnot(
  all(
    prediction_counts$n_predictions == 1
  )
)


model_counts <- primary_predictions |>
  summarise(
    n = n(),
    n_patients = n_distinct(
      RecordID
    ),
    .by = model
  )


stopifnot(
  all(
    model_counts$n_patients ==
      N_STAYS
  ),

  all(
    model_counts$n ==
      N_STAYS
  )
)


# Confirm outcomes agree across model files.

outcome_check <- primary_predictions |>
  summarise(
    n_outcomes =
      n_distinct(
        outcome
      ),
    .by = RecordID
  )


stopifnot(
  all(
    outcome_check$n_outcomes == 1
  )
)


# =============================================================================
# 15. Unified performance table
# =============================================================================

assessment_metrics <- lapply(
  primary_models,
  function(model_name) {

    p <- primary_predictions |>
      filter(
        model == model_name
      )

    tibble(
      model = model_name,

      auc =
        calc_auc(
          p$outcome,
          p$probability
        ),

      brier =
        calc_brier(
          p$outcome,
          p$probability
        ),

      log_loss =
        calc_log_loss(
          p$outcome,
          p$probability
        )
    )
  }
) |>
  bind_rows() |>
  arrange(
    desc(auc)
  )


# =============================================================================
# 16. Calibration intercept and slope
# =============================================================================
#
# For OOF predictions p:
#
#   logit(p) = log(p / (1-p))
#
# Calibration intercept:
#   Fit outcome ~ offset(logit(p))
#
# Ideal intercept = 0.
#
# Calibration slope:
#   Fit outcome ~ logit(p)
#
# Ideal slope = 1.
#
# slope < 1:
#   predictions tend to be too extreme / overfit
#
# slope > 1:
#   predictions tend to be insufficiently extreme.


calculate_calibration <- function(
  predictions,
  model_name
) {

  p <- clip_probability(
    predictions$probability
  )

  lp <- qlogis(
    p
  )

  calibration_data <- tibble(
    outcome =
      predictions$outcome,

    linear_predictor =
      lp
  )


  # ---------------------------------------------------------------------------
  # Calibration intercept
  # ---------------------------------------------------------------------------
  #
  # Hold the prediction linear predictor fixed as an offset:
  #
  #   logit(P(Y=1)) = intercept + offset(lp)
  #
  # Ideal intercept = 0.

  intercept_fit <- glm(
    outcome ~ 1,
    data =
      calibration_data,
    family =
      binomial(),
    offset =
      linear_predictor
  )


  # ---------------------------------------------------------------------------
  # Calibration slope
  # ---------------------------------------------------------------------------
  #
  #   logit(P(Y=1)) = beta0 + beta1 * lp
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


calibration_summary <- lapply(
  primary_models,
  function(model_name) {

    predictions <- primary_predictions |>
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


# =============================================================================
# 17. Calibration bins
# =============================================================================
#
# Use deciles of predicted risk separately within each model.
#
# For each bin:
#
#   mean predicted probability
#   observed mortality
#
# are compared.
#
# This is descriptive calibration assessment based on OOF predictions.


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


calibration_bins <- lapply(
  primary_models,
  function(model_name) {

    predictions <- primary_predictions |>
      filter(
        model ==
          model_name
      )

    make_calibration_bins(
      predictions =
        predictions,

      model_name =
        model_name,

      n_bins =
        10
    )
  }
) |>
  bind_rows()


# =============================================================================
# 18. Calibration plot
# =============================================================================

calibration_plot <- ggplot(
  calibration_bins,
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
      "Out-of-fold calibration by model",

    subtitle =
      paste(
        "Observed mortality versus mean predicted risk",
        "within deciles of predicted probability"
      ),

    x =
      "Mean predicted mortality probability",

    y =
      "Observed mortality"
  ) +

  theme_minimal(
    base_size =
      10
  )


ggsave(
  filename =
    here(
      "output",
      "model_calibration_plot.png"
    ),

  plot =
    calibration_plot,

  width =
    11,

  height =
    9,

  dpi =
    150
)


# =============================================================================
# 19. Combine discrimination / probability accuracy / calibration
# =============================================================================

final_comparison <- assessment_metrics |>
  left_join(
    calibration_summary,
    by =
      "model"
  )


# =============================================================================
# 20. Add model-family descriptions
# =============================================================================

model_descriptions <- tibble(

  model =
    c(
      "SAPS-I",
      "SOFA",
      "Logistic baseline",
      "Nonlinear logistic",
      "Ridge",
      "Elastic Net",
      "LASSO",
      "Random Forest",
      "Gradient Boosting"
    ),

  model_family =
    c(
      "Clinical benchmark",
      "Clinical benchmark",
      "Logistic regression",
      "Nonlinear logistic regression",
      "Regularized logistic regression",
      "Regularized logistic regression",
      "Regularized logistic regression",
      "Tree ensemble",
      "Gradient boosting"
    ),

  complexity =
    c(
      "Low",
      "Low",
      "Low",
      "Moderate",
      "Moderate",
      "Moderate",
      "Moderate",
      "High",
      "High"
    )
)


final_comparison <- final_comparison |>
  left_join(
    model_descriptions,
    by =
      "model"
  ) |>
  relocate(
    model,
    model_family,
    complexity
  ) |>
  arrange(
    desc(
      auc
    )
  )


# =============================================================================
# 21. Compare linear vs nonlinear compact logistic
# =============================================================================

linear_vs_nonlinear <- final_comparison |>
  filter(
    model %in%
      c(
        "Logistic baseline",
        "Nonlinear logistic"
      )
  ) |>
  select(
    model,
    auc,
    brier,
    log_loss,
    calibration_intercept,
    calibration_slope
  )


cat(
  "\nLinear vs nonlinear logistic comparison:\n"
)

print(
  linear_vs_nonlinear,
  n = Inf
)


# =============================================================================
# 22. Load XGBoost round sensitivity as a secondary analysis
# =============================================================================
#
# IMPORTANT:
#
# The sensitivity model is NOT inserted into the primary ranking.
#
# It is reported separately because:
#
#   - it was motivated by the 400-round boundary finding
#   - it therefore represents a secondary sensitivity analysis
#   - treating it as another primary candidate would expand the model-search
#     process after observing the original Set-A results.


sensitivity_path <- here(
  "output",
  "xgb_round_sensitivity_predictions.csv"
)


if (
  file.exists(
    sensitivity_path
  )
) {

  sensitivity_predictions <- read_csv(
    sensitivity_path,
    show_col_types =
      FALSE
  )


  stopifnot(
    nrow(
      sensitivity_predictions
    ) ==
      N_STAYS,

    n_distinct(
      sensitivity_predictions$RecordID
    ) ==
      N_STAYS
  )


  sensitivity_metrics <- tibble(

    analysis =
      "Secondary sensitivity",

    model =
      "Gradient Boosting: max 1000 rounds",

    auc =
      calc_auc(
        sensitivity_predictions$outcome,
        sensitivity_predictions$probability
      ),

    brier =
      calc_brier(
        sensitivity_predictions$outcome,
        sensitivity_predictions$probability
      ),

    log_loss =
      calc_log_loss(
        sensitivity_predictions$outcome,
        sensitivity_predictions$probability
      )
  )


  sensitivity_calibration <- calculate_calibration(
    predictions =
      sensitivity_predictions,

    model_name =
      "Gradient Boosting: max 1000 rounds"
  )


  sensitivity_summary <- sensitivity_metrics |>
    left_join(
      sensitivity_calibration,
      by =
        "model"
    )


  cat(
    "\nSecondary XGBoost sensitivity analysis:\n"
  )

  print(
    sensitivity_summary,
    n = Inf
  )


  write_csv(
    sensitivity_summary,
    here(
      "output",
      "model_assessment_xgb_sensitivity.csv"
    )
  )
}


# =============================================================================
# 23. Fit nonlinear logistic model on all Set A
# =============================================================================
#
# This full-Set-A fit is NOT used to estimate predictive performance.
#
# OOF predictions above provide the development-performance estimate.
#
# This fit documents the final nonlinear model specification and coefficients.


final_train <- dat


final_medians <- fit_median_imputer(
  training_data =
    final_train,

  variables =
    baseline_continuous
)


final_train <- apply_median_imputer(
  final_train,
  final_medians
)


gender_mode <- fit_mode(
  final_train$Gender
)


icu_mode <- fit_mode(
  final_train$ICUType
)


final_train <- apply_mode(
  final_train,
  "Gender",
  gender_mode
)


final_train <- apply_mode(
  final_train,
  "ICUType",
  icu_mode
)


final_train <- prepare_categorical(
  final_train
)


final_nonlinear_fit <- glm(
  nonlinear_formula,
  data =
    final_train,
  family =
    binomial()
)


if (
  !isTRUE(
    final_nonlinear_fit$converged
  )
) {

  warning(
    "Final nonlinear logistic model did not converge."
  )
}


cat(
  "\nFinal nonlinear logistic model converged:",
  final_nonlinear_fit$converged,
  "\n"
)


# =============================================================================
# 24. Save nonlinear-model coefficients
# =============================================================================
#
# Natural-spline coefficients are basis coefficients.
#
# They should NOT be interpreted individually as simple one-unit odds ratios.
# Their purpose here is reproducibility/documentation of the fitted model.


coefficient_matrix <- summary(
  final_nonlinear_fit
)$coefficients


nonlinear_coefficients <- tibble(

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
)


# =============================================================================
# 25. Save nonlinear preprocessing specification
# =============================================================================

nonlinear_imputation <- tibble(
  feature =
    names(
      final_medians
    ),

  median_set_a =
    as.numeric(
      final_medians
    )
)


nonlinear_categorical_imputation <- tibble(
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
# 26. Save all outputs
# =============================================================================

write_csv(
  nonlinear_predictions,
  here(
    "output",
    "cv_predictions_nonlinear_logistic.csv"
  )
)


write_csv(
  nonlinear_performance,
  here(
    "output",
    "cv_performance_nonlinear_logistic.csv"
  )
)


write_csv(
  nonlinear_fold_performance,
  here(
    "output",
    "cv_performance_nonlinear_logistic_by_fold.csv"
  )
)


write_csv(
  primary_predictions,
  here(
    "output",
    "model_assessment_all_predictions.csv"
  )
)


write_csv(
  assessment_metrics,
  here(
    "output",
    "model_assessment_metrics.csv"
  )
)


write_csv(
  calibration_summary,
  here(
    "output",
    "model_calibration_summary.csv"
  )
)


write_csv(
  calibration_bins,
  here(
    "output",
    "model_calibration_bins.csv"
  )
)


write_csv(
  final_comparison,
  here(
    "output",
    "model_comparison_final_set_a.csv"
  )
)


write_csv(
  linear_vs_nonlinear,
  here(
    "output",
    "linear_vs_nonlinear_logistic.csv"
  )
)


write_csv(
  nonlinear_coefficients,
  here(
    "output",
    "nonlinear_logistic_coefficients_set_a.csv"
  )
)


write_csv(
  nonlinear_imputation,
  here(
    "output",
    "nonlinear_logistic_imputation_set_a.csv"
  )
)


write_csv(
  nonlinear_categorical_imputation,
  here(
    "output",
    "nonlinear_logistic_categorical_imputation_set_a.csv"
  )
)


# =============================================================================
# 27. Final validation
# =============================================================================

stopifnot(
  nrow(
    final_comparison
  ) ==
    length(
      primary_models
    ),

  !any(
    is.na(
      final_comparison$auc
    )
  ),

  !any(
    is.na(
      final_comparison$brier
    )
  ),

  !any(
    is.na(
      final_comparison$log_loss
    )
  ),

  all(
    is.finite(
      final_comparison$calibration_intercept
    )
  ),

  all(
    is.finite(
      final_comparison$calibration_slope
    )
  )
)


# =============================================================================
# 28. Identify performance leaders without automatically declaring a winner
# =============================================================================
#
# Different metrics answer different questions:
#
#   AUROC:
#     discrimination / ranking
#
#   Brier:
#     squared probability accuracy
#
#   Log loss:
#     probability accuracy with strong penalty for confident errors
#
# Therefore we report metric-specific leaders before making the final
# scientific model-selection decision.


auc_leader <- final_comparison |>
  slice_max(
    order_by =
      auc,
    n =
      1,
    with_ties =
      FALSE
  )


brier_leader <- final_comparison |>
  slice_min(
    order_by =
      brier,
    n =
      1,
    with_ties =
      FALSE
  )


logloss_leader <- final_comparison |>
  slice_min(
    order_by =
      log_loss,
    n =
      1,
    with_ties =
      FALSE
  )


cat(
  "\nMetric-specific leaders:\n"
)

cat(
  "  Highest AUROC:",
  auc_leader$model,
  sprintf(
    "(%.4f)",
    auc_leader$auc
  ),
  "\n"
)

cat(
  "  Lowest Brier score:",
  brier_leader$model,
  sprintf(
    "(%.4f)",
    brier_leader$brier
  ),
  "\n"
)

cat(
  "  Lowest log loss:",
  logloss_leader$model,
  sprintf(
    "(%.4f)",
    logloss_leader$log_loss
  ),
  "\n"
)


# =============================================================================
# 29. Final report
# =============================================================================

cat(
  "\n============================================================\n"
)

cat(
  "Checkpoint 2 final Set-A model assessment complete.\n"
)

cat(
  "============================================================\n"
)


cat(
  "\nPrimary model comparison:\n"
)

print(
  final_comparison,
  n = Inf
)


cat(
  "\nCalibration summary:\n"
)

print(
  calibration_summary |>
    arrange(
      model
    ),
  n = Inf
)


cat(
  "\nInterpretation guide:\n"
)

cat(
  paste(
    "  AUROC: higher values indicate better discrimination.\n"
  )
)

cat(
  paste(
    "  Brier score: lower values indicate better squared",
    "probability accuracy.\n"
  )
)

cat(
  paste(
    "  Log loss: lower values indicate better probability",
    "accuracy and strongly penalize confident errors.\n"
  )
)

cat(
  paste(
    "  Calibration intercept: ideal value is 0.\n"
  )
)

cat(
  paste(
    "  Calibration slope: ideal value is 1;",
    "values below 1 indicate predictions that are",
    "too extreme on average.\n"
  )
)


cat(
  "\nImportant modeling rule:\n"
)

cat(
  paste(
    "This script completes Set-A model development and assessment.",
    "No Set-B outcome information has been used.\n"
  )
)

cat(
  paste(
    "After reviewing these results, the final model specification",
    "should be frozen before the single-look Set-B evaluation.\n"
  )
)


cat(
  "\nGenerated outputs:\n"
)

cat(
  "  output/cv_predictions_nonlinear_logistic.csv\n"
)

cat(
  "  output/cv_performance_nonlinear_logistic.csv\n"
)

cat(
  "  output/cv_performance_nonlinear_logistic_by_fold.csv\n"
)

cat(
  "  output/model_assessment_all_predictions.csv\n"
)

cat(
  "  output/model_assessment_metrics.csv\n"
)

cat(
  "  output/model_calibration_summary.csv\n"
)

cat(
  "  output/model_calibration_bins.csv\n"
)

cat(
  "  output/model_calibration_plot.png\n"
)

cat(
  "  output/model_comparison_final_set_a.csv\n"
)

cat(
  "  output/linear_vs_nonlinear_logistic.csv\n"
)

cat(
  "  output/nonlinear_logistic_coefficients_set_a.csv\n"
)

cat(
  "  output/nonlinear_logistic_imputation_set_a.csv\n"
)

cat(
  "  output/nonlinear_logistic_categorical_imputation_set_a.csv\n"
)

if (
  exists(
    "sensitivity_summary"
  )
) {

  cat(
    "  output/model_assessment_xgb_sensitivity.csv\n"
  )
}
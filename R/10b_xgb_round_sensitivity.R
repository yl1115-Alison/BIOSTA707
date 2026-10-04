# BIOSTAT 707 Checkpoint 2
# 10b_xgb_round_sensitivity.R
#
# Purpose:
#   Sensitivity analysis for the XGBoost training-round boundary used in
#   10_tree_boosting_models.R.
#
# Motivation:
#   In the primary analysis, all selected XGBoost configurations reached the
#   prespecified maximum of 400 boosting rounds. This sensitivity analysis
#   checks whether that boundary truncated the preferred training duration.
#
# IMPORTANT:
#   This is a ONE-TIME sensitivity analysis.
#
#   The structural hyperparameter grid is unchanged:
#
#     max_depth        = 2, 4, 6
#     eta              = 0.03, 0.10
#     min_child_weight = 1, 5
#
#   Only:
#
#     maximum boosting rounds: 400 -> 1000
#     early stopping patience: 30 -> 50
#
#   are changed.
#
# Resampling:
#   Outer loop:
#     fixed 5 folds from 07_define_resampling.R
#
#   Inner loop:
#     stratified 5-fold CV inside each outer-training partition
#
# Leakage control:
#   - feature filtering is fit on training data only
#   - imputation is fit on training data only
#   - dummy encoding schema is fit on training data only
#   - early stopping uses inner-validation data only
#   - outer-validation data are never used for tuning
#   - Set B is not accessed
#
# Inputs:
#   output/set-a_features.csv
#   output/set-a_cv_folds.csv
#   output/feature_dictionary.csv
#   output/cv_performance_tree_boosting.csv
#
# Outputs:
#   output/xgb_round_sensitivity_predictions.csv
#   output/xgb_round_sensitivity_performance.csv
#   output/xgb_round_sensitivity_by_fold.csv
#   output/xgb_round_sensitivity_tuning.csv
#   output/xgb_round_sensitivity_iteration_summary.csv
#   output/xgb_round_sensitivity_comparison.csv
#   output/xgb_round_sensitivity_final_configuration.csv


# =============================================================================
# 1. Packages and constants
# =============================================================================

library(readr)
library(dplyr)
library(here)
library(pROC)
library(xgboost)

N_STAYS <- 4000
N_OUTER_FOLDS <- 5
N_INNER_FOLDS <- 5

OUTCOME <- "In-hospital_death"

INNER_SEED <- 20261005

MAX_MISSING_PROPORTION <- 0.80

N_THREADS <- 1

# Sensitivity-analysis changes.

XGB_NROUNDS_SENSITIVITY <- 1000
XGB_EARLY_STOPPING_SENSITIVITY <- 50


# =============================================================================
# 2. Load data
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

feature_dictionary <- read_csv(
  here(
    "output",
    "feature_dictionary.csv"
  ),
  show_col_types = FALSE
)


stopifnot(
  nrow(dat) == N_STAYS,
  nrow(folds) == N_STAYS,

  n_distinct(dat$RecordID) == N_STAYS,
  n_distinct(folds$RecordID) == N_STAYS,

  !anyDuplicated(dat$RecordID),
  !anyDuplicated(folds$RecordID),

  OUTCOME %in%
    names(dat),

  all(
    dat[[OUTCOME]] %in%
      c(0, 1)
  ),

  all(
    folds$fold %in%
      seq_len(
        N_OUTER_FOLDS
      )
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


# =============================================================================
# 3. Define predictor space
# =============================================================================

non_predictors <- c(
  "RecordID",
  "fold",
  OUTCOME,
  "SAPS-I",
  "SOFA"
)


all_predictors <- setdiff(
  names(dat),
  non_predictors
)


stopifnot(
  length(all_predictors) == 313,

  all(
    all_predictors %in%
      feature_dictionary$feature
  )
)


CATEGORICAL_FEATURES <- c(
  "ICUType"
)


cat(
  "Set A ICU stays:",
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
  "Engineered predictors:",
  length(all_predictors),
  "\n"
)


# =============================================================================
# 4. Metrics
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

  p <- clip_probability(
    p
  )


  -mean(
    y * log(p) +
      (1 - y) *
        log(1 - p)
  )
}


metric_row <- function(
  predictions,
  fold_value = NA_integer_
) {

  tibble(
    model =
      "Gradient Boosting sensitivity",

    fold =
      fold_value,

    n =
      nrow(
        predictions
      ),

    deaths =
      sum(
        predictions$outcome == 1
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
# 5. Stratified inner folds
# =============================================================================

make_stratified_folds <- function(
  y,
  k,
  seed
) {

  set.seed(
    seed
  )


  fold_id <- integer(
    length(y)
  )


  for (value in c(0, 1)) {

    index <- which(
      y == value
    )


    index <- sample(
      index,
      length(index),
      replace = FALSE
    )


    fold_id[index] <- rep(
      seq_len(k),
      length.out =
        length(index)
    )
  }


  stopifnot(
    all(
      fold_id %in%
        seq_len(k)
    )
  )


  fold_id
}


# =============================================================================
# 6. Leakage-safe preprocessing
# =============================================================================

fit_mode <- function(x) {

  x <- x[
    !is.na(x)
  ]


  if (
    length(x) == 0
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
# 6a. Feature filter
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


  feature_types <- feature_dictionary$feature_type[
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
        unique(x)
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
# 6b. Imputation
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
        length(x) == 0
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
        "Unable to estimate median for:",
        paste(
          bad,
          collapse = ", "
        )
      )
    )
  }


  modes <- list()


  for (feature in categorical) {

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


  for (feature in names(imputer$medians)) {

    missing <- is.na(
      output[[feature]]
    )


    output[[feature]][missing] <-
      imputer$medians[[feature]]
  }


  for (feature in names(imputer$modes)) {

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
# 6c. Design matrix
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
    ) <- missing_columns


    x <- cbind(
      x,
      zeros
    )
  }


  extra_columns <- setdiff(
    colnames(x),
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


  stopifnot(
    all(
      is.finite(
        x
      )
    )
  )


  x
}


# -----------------------------------------------------------------------------
# 6d. Complete preprocessor
# -----------------------------------------------------------------------------

fit_preprocessor <- function(
  training_data
) {

  filter_object <- fit_feature_filter(
    training_data
  )


  features <- filter_object$selected


  imputer <- fit_imputer(
    training_data,
    features
  )


  imputed_training <- apply_imputer(
    training_data,
    imputer
  )


  schema <- fit_design_schema(
    imputed_training,
    features
  )


  list(
    features =
      features,

    filter =
      filter_object,

    imputer =
      imputer,

    schema =
      schema
  )
}


apply_preprocessor <- function(
  data,
  preprocessor
) {

  imputed <- apply_imputer(
    data,
    preprocessor$imputer
  )


  make_design_matrix(
    imputed,
    preprocessor$features,
    preprocessor$schema
  )
}


# =============================================================================
# 7. XGBoost structural grid
# =============================================================================
#
# IDENTICAL to the primary analysis.


XGB_GRID <- expand.grid(
  max_depth = c(
    2,
    4,
    6
  ),

  eta = c(
    0.03,
    0.10
  ),

  min_child_weight = c(
    1,
    5
  ),

  stringsAsFactors =
    FALSE
)


# =============================================================================
# 8. Tune XGBoost within one outer-training set
# =============================================================================

tune_xgb_sensitivity <- function(
  outer_training,
  inner_fold_id
) {

  results <- list()


  for (grid_id in seq_len(nrow(XGB_GRID))) {

    config <- XGB_GRID[
      grid_id,
      ,
      drop = FALSE
    ]


    fold_results <- list()


    for (
      inner_fold in seq_len(
        N_INNER_FOLDS
      )
    ) {

      inner_training <- outer_training[
        inner_fold_id !=
          inner_fold,
        ,
        drop = FALSE
      ]


      inner_validation <- outer_training[
        inner_fold_id ==
          inner_fold,
        ,
        drop = FALSE
      ]


      # -----------------------------------------------------------------------
      # Fit preprocessing using inner-training only
      # -----------------------------------------------------------------------

      preprocessor <- fit_preprocessor(
        inner_training
      )


      x_train <- apply_preprocessor(
        inner_training,
        preprocessor
      )


      x_validation <- apply_preprocessor(
        inner_validation,
        preprocessor
      )


      y_train <- inner_training[[OUTCOME]]

      y_validation <- inner_validation[[OUTCOME]]


      dtrain <- xgb.DMatrix(
        data =
          x_train,

        label =
          y_train
      )


      dvalidation <- xgb.DMatrix(
        data =
          x_validation,

        label =
          y_validation
      )


      # -----------------------------------------------------------------------
      # Inner-fold XGBoost with expanded round boundary
      # -----------------------------------------------------------------------

      fit <- xgb.train(
        params = list(
          objective =
            "binary:logistic",

          eval_metric =
            "auc",

          max_depth =
            config$max_depth,

          eta =
            config$eta,

          min_child_weight =
            config$min_child_weight,

          subsample =
            0.8,

          colsample_bytree =
            0.8,

          nthread =
            N_THREADS
        ),

        data =
          dtrain,

        nrounds =
          XGB_NROUNDS_SENSITIVITY,

        watchlist =
          list(
            validation =
              dvalidation
          ),

        early_stopping_rounds =
          XGB_EARLY_STOPPING_SENSITIVITY,

        verbose =
          0
      )


      probability <- predict(
        fit,
        dvalidation
      )


      best_iteration <- fit$best_iteration


      if (
        is.null(
          best_iteration
        ) ||
          length(
            best_iteration
          ) ==
          0
      ) {

        best_iteration <-
          XGB_NROUNDS_SENSITIVITY
      }


      fold_results[[inner_fold]] <- tibble(
        grid_id =
          grid_id,

        inner_fold =
          inner_fold,

        max_depth =
          config$max_depth,

        eta =
          config$eta,

        min_child_weight =
          config$min_child_weight,

        best_iteration =
          as.integer(
            best_iteration
          ),

        hit_round_boundary =
          as.integer(
            best_iteration >=
              XGB_NROUNDS_SENSITIVITY
          ),

        auc =          calc_auc(
            y_validation,
            probability
          ),

        brier =
          calc_brier(
            y_validation,
            probability
          ),

        log_loss =
          calc_log_loss(
            y_validation,
            probability
          )
      )
    }


    results[[grid_id]] <- bind_rows(
      fold_results
    )
  }


  # ---------------------------------------------------------------------------
  # Summarize each structural configuration across inner folds
  # ---------------------------------------------------------------------------

  fold_results_all <- bind_rows(
    results
  )


  tuning_summary <- fold_results_all |>
    summarise(
      mean_auc =
        mean(
          auc
        ),

      sd_auc =
        sd(
          auc
        ),

      mean_brier =
        mean(
          brier
        ),

      mean_log_loss =
        mean(
          log_loss
        ),

      median_best_iteration =
        as.integer(
          round(
            median(
              best_iteration
            )
          )
        ),

      min_best_iteration =
        min(
          best_iteration
        ),

      max_best_iteration =
        max(
          best_iteration
        ),

      proportion_hitting_boundary =
        mean(
          hit_round_boundary
        ),

      .by =
        c(
          grid_id,
          max_depth,
          eta,
          min_child_weight
        )
    ) |>
    arrange(
      desc(
        mean_auc
      ),
      mean_log_loss,
      mean_brier
    )


  best <- tuning_summary |>
    slice_head(
      n = 1
    )


  list(
    best =
      best,

    tuning_summary =
      tuning_summary,

    fold_results =
      fold_results_all
  )
}


# =============================================================================
# 9. Fit selected XGBoost configuration on one outer fold
# =============================================================================

fit_outer_xgb_sensitivity <- function(
  outer_training,
  outer_validation,
  tuning,
  outer_fold
) {

  # Fit preprocessing using the complete outer-training partition only.

  preprocessor <- fit_preprocessor(
    outer_training
  )


  x_train <- apply_preprocessor(
    outer_training,
    preprocessor
  )


  x_validation <- apply_preprocessor(
    outer_validation,
    preprocessor
  )


  y_train <- outer_training[[OUTCOME]]


  dtrain <- xgb.DMatrix(
    data =
      x_train,

    label =
      y_train
  )


  dvalidation <- xgb.DMatrix(
    data =
      x_validation
  )


  # The number of rounds comes only from the inner-CV tuning result.

  nrounds_final <- max(
    1L,
    as.integer(
      tuning$median_best_iteration[[1]]
    )
  )


  fit <- xgb.train(
    params = list(
      objective =
        "binary:logistic",

      eval_metric =
        "auc",

      max_depth =
        tuning$max_depth[[1]],

      eta =
        tuning$eta[[1]],

      min_child_weight =
        tuning$min_child_weight[[1]],

      subsample =
        0.8,

      colsample_bytree =
        0.8,

      nthread =
        N_THREADS
    ),

    data =
      dtrain,

    nrounds =
      nrounds_final,

    verbose =
      0
  )


  probability <- predict(
    fit,
    dvalidation
  )


  stopifnot(
    length(
      probability
    ) ==
      nrow(
        outer_validation
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
      outer_validation$RecordID,

    fold =
      outer_fold,

    outcome =
      outer_validation[[OUTCOME]],

    model =
      "Gradient Boosting sensitivity",

    max_depth =
      tuning$max_depth[[1]],

    eta =
      tuning$eta[[1]],

    min_child_weight =
      tuning$min_child_weight[[1]],

    nrounds =
      nrounds_final,

    probability =
      as.numeric(
        probability
      )
  )
}


# =============================================================================
# 10. Outer nested-CV sensitivity loop
# =============================================================================

outer_predictions <- list()
outer_tuning <- list()
outer_inner_iterations <- list()


for (outer_fold in seq_len(N_OUTER_FOLDS)) {

  cat(
    "\n============================================================\n"
  )

  cat(
    "XGBoost sensitivity: outer fold",
    outer_fold,
    "of",
    N_OUTER_FOLDS,
    "\n"
  )

  cat(
    "============================================================\n"
  )


  outer_training <- dat |>
    filter(
      fold !=
        outer_fold
    )


  outer_validation <- dat |>
    filter(
      fold ==
        outer_fold
    )


  stopifnot(
    nrow(outer_training) +
      nrow(outer_validation) ==
      N_STAYS,

    length(
      intersect(
        outer_training$RecordID,
        outer_validation$RecordID
      )
    ) ==
      0
  )


  cat(
    "Outer training:",
    nrow(
      outer_training
    ),
    "\n"
  )

  cat(
    "Outer validation:",
    nrow(
      outer_validation
    ),
    "\n"
  )


  # ---------------------------------------------------------------------------
  # Create inner folds from outer-training observations only
  # ---------------------------------------------------------------------------

  inner_fold_id <- make_stratified_folds(
    y =
      outer_training[[OUTCOME]],

    k =
      N_INNER_FOLDS,

    seed =
      INNER_SEED +
        outer_fold
  )


  # ---------------------------------------------------------------------------
  # Tune using expanded training-round boundary
  # ---------------------------------------------------------------------------

  cat(
    "Running inner-CV XGBoost sensitivity tuning...\n"
  )


  tuning <- tune_xgb_sensitivity(
    outer_training =
      outer_training,

    inner_fold_id =
      inner_fold_id
  )


  best <- tuning$best |>
    mutate(
      outer_fold =
        outer_fold
    )


  outer_tuning[[outer_fold]] <-
    best


  # Preserve all inner-fold best-iteration results so that we can explicitly
  # inspect whether 1000 rounds is still a binding boundary.

  outer_inner_iterations[[outer_fold]] <-
    tuning$fold_results |>
    mutate(
      outer_fold =
        outer_fold
    )


  cat(
    "\nSelected sensitivity configuration:\n"
  )

  print(
    best,
    n = Inf
  )


  # ---------------------------------------------------------------------------
  # Fit on outer-training data and predict untouched outer-validation
  # ---------------------------------------------------------------------------

  outer_predictions[[outer_fold]] <-
    fit_outer_xgb_sensitivity(
      outer_training =
        outer_training,

      outer_validation =
        outer_validation,

      tuning =
        best,

      outer_fold =
        outer_fold
    )
}


# =============================================================================
# 11. Combine outer-fold results
# =============================================================================

all_predictions <- bind_rows(
  outer_predictions
)


all_tuning <- bind_rows(
  outer_tuning
)


all_inner_iterations <- bind_rows(
  outer_inner_iterations
)


# =============================================================================
# 12. Validate out-of-fold predictions
# =============================================================================

stopifnot(
  nrow(
    all_predictions
  ) ==
    N_STAYS,

  n_distinct(
    all_predictions$RecordID
  ) ==
    N_STAYS,

  !anyDuplicated(
    all_predictions$RecordID
  ),

  !any(
    is.na(
      all_predictions$probability
    )
  ),

  all(
    is.finite(
      all_predictions$probability
    )
  ),

  all(
    all_predictions$probability >=
      0 &
      all_predictions$probability <=
      1
  )
)


# =============================================================================
# 13. Overall sensitivity performance
# =============================================================================

overall_performance <- metric_row(
  predictions =
    all_predictions
)


cat(
  "\n============================================================\n"
)

cat(
  "XGBoost round-sensitivity nested-CV performance\n"
)

cat(
  "============================================================\n"
)

print(
  overall_performance,
  n = Inf
)


# =============================================================================
# 14. Fold-specific performance
# =============================================================================

performance_by_fold <- lapply(
  seq_len(
    N_OUTER_FOLDS
  ),
  function(k) {

    predictions <- all_predictions |>
      filter(
        fold ==
          k
      )


    metric_row(
      predictions =
        predictions,

      fold_value =
        k
    )
  }
) |>
  bind_rows()


performance_fold_summary <- performance_by_fold |>
  summarise(
    mean_auc =
      mean(
        auc
      ),

    sd_auc =
      sd(
        auc
      ),

    mean_brier =
      mean(
        brier
      ),

    sd_brier =
      sd(
        brier
      ),

    mean_log_loss =
      mean(
        log_loss
      ),

    sd_log_loss =
      sd(
        log_loss
      )
  )


# =============================================================================
# 15. Diagnose whether 1000 rounds remains a binding boundary
# =============================================================================
#
# We examine the INNER-CV runs corresponding to the structural configuration
# selected in each outer fold.


selected_iteration_details <- lapply(
  seq_len(
    N_OUTER_FOLDS
  ),
  function(k) {

    selected <- all_tuning |>
      filter(
        outer_fold ==
          k
      )


    all_inner_iterations |>
      filter(
        outer_fold ==
          k,

        max_depth ==
          selected$max_depth[[1]],

        eta ==
          selected$eta[[1]],

        min_child_weight ==
          selected$min_child_weight[[1]]
      )
  }
) |>
  bind_rows()


iteration_summary <- selected_iteration_details |>
  summarise(
    n_inner_fits =
      n(),

    median_best_iteration =
      median(
        best_iteration
      ),

    min_best_iteration =
      min(
        best_iteration
      ),

    max_best_iteration =
      max(
        best_iteration
      ),

    n_hitting_1000 =
      sum(
        best_iteration >=
          XGB_NROUNDS_SENSITIVITY
      ),

    proportion_hitting_1000 =
      mean(
        best_iteration >=
          XGB_NROUNDS_SENSITIVITY
      ),

    .by =
      outer_fold
  ) |>
  arrange(
    outer_fold
  )


cat(
  "\nSelected-configuration iteration diagnostics:\n"
)

print(
  iteration_summary,
  n = Inf
)


# =============================================================================
# 16. Compare sensitivity analysis with primary 400-round XGBoost
# =============================================================================

primary_path <- here(
  "output",
  "cv_performance_tree_boosting.csv"
)


if (
  file.exists(
    primary_path
  )
) {

  primary_performance <- read_csv(
    primary_path,
    show_col_types = FALSE
  ) |>
    filter(
      model ==
        "Gradient Boosting"
    ) |>
    select(
      model,
      auc,
      brier,
      log_loss
    ) |>
    mutate(
      analysis =
        "Primary: max 400 rounds"
    )


  sensitivity_performance <- overall_performance |>
    select(
      model,
      auc,
      brier,
      log_loss
    ) |>
    mutate(
      analysis =
        "Sensitivity: max 1000 rounds"
    )


  comparison <- bind_rows(
    primary_performance,
    sensitivity_performance
  ) |>
    select(
      analysis,
      model,
      auc,
      brier,
      log_loss
    )


  cat(
    "\nPrimary vs round-sensitivity comparison:\n"
  )

  print(
    comparison,
    n = Inf
  )


  write_csv(
    comparison,
    here(
      "output",
      "xgb_round_sensitivity_comparison.csv"
    )
  )
}


# =============================================================================
# 17. Final full-Set-A sensitivity tuning
# =============================================================================
#
# This provides the candidate configuration that would be used if the
# sensitivity analysis demonstrates that the original 400-round boundary
# materially truncated training.
#
# It does NOT access Set B.


cat(
  "\n============================================================\n"
)

cat(
  "Final full-Set-A XGBoost sensitivity tuning\n"
)

cat(
  "============================================================\n"
)


final_inner_fold_id <- make_stratified_folds(
  y =
    dat[[OUTCOME]],

  k =
    N_INNER_FOLDS,

  seed =
    INNER_SEED +
      1000
)


final_tuning <- tune_xgb_sensitivity(
  outer_training =
    dat,

  inner_fold_id =
    final_inner_fold_id
)


final_configuration <- final_tuning$best |>
  mutate(
    model =
      "Gradient Boosting sensitivity"
  ) |>
  select(
    model,
    max_depth,
    eta,
    min_child_weight,
    median_best_iteration,
    min_best_iteration,
    max_best_iteration,
    proportion_hitting_boundary,
    mean_auc,
    sd_auc,
    mean_brier,
    mean_log_loss
  )


cat(
  "\nFinal full-Set-A sensitivity configuration:\n"
)

print(
  final_configuration,
  n = Inf
)


# =============================================================================
# 18. Save outputs
# =============================================================================

write_csv(
  all_predictions,
  here(
    "output",
    "xgb_round_sensitivity_predictions.csv"
  )
)


write_csv(
  overall_performance,
  here(
    "output",
    "xgb_round_sensitivity_performance.csv"
  )
)


write_csv(
  performance_by_fold,
  here(
    "output",
    "xgb_round_sensitivity_by_fold.csv"
  )
)


write_csv(
  all_tuning,
  here(
    "output",
    "xgb_round_sensitivity_tuning.csv"
  )
)


write_csv(
  selected_iteration_details,
  here(
    "output",
    "xgb_round_sensitivity_selected_iterations.csv"
  )
)


write_csv(
  iteration_summary,
  here(
    "output",
    "xgb_round_sensitivity_iteration_summary.csv"
  )
)


write_csv(
  final_configuration,
  here(
    "output",
    "xgb_round_sensitivity_final_configuration.csv"
  )
)


# =============================================================================
# 19. Boundary interpretation
# =============================================================================

overall_boundary_rate <- mean(
  selected_iteration_details$
    best_iteration >=
    XGB_NROUNDS_SENSITIVITY
)


cat(
  "\n============================================================\n"
)

cat(
  "Round-boundary diagnostic\n"
)

cat(
  "============================================================\n"
)

cat(
  "Selected inner fits hitting 1000 rounds:",
  sum(
    selected_iteration_details$
      best_iteration >=
      XGB_NROUNDS_SENSITIVITY
  ),
  "of",
  nrow(
    selected_iteration_details
  ),
  "\n"
)

cat(
  "Overall boundary-hit proportion:",
  sprintf(
    "%.1f%%",
    100 *
      overall_boundary_rate
  ),
  "\n"
)


if (
  overall_boundary_rate ==
    0
) {

  cat(
    paste(
      "Interpretation: the expanded 1000-round limit was not binding",
      "for any selected inner-CV fit.\n"
    )
  )

} else if (
  overall_boundary_rate <=
    0.20
) {

  cat(
    paste(
      "Interpretation: the 1000-round boundary was reached only",
      "occasionally; the expanded search largely resolves the",
      "original 400-round boundary concern.\n"
    )
  )

} else {

  cat(
    paste(
      "Interpretation: the expanded round limit remains binding",
      "for a substantial fraction of selected inner-CV fits.",
      "Report this limitation rather than continuing indefinite",
      "hyperparameter expansion.\n"
    )
  )
}


# =============================================================================
# 20. Final report
# =============================================================================

cat(
  "\n============================================================\n"
)

cat(
  "XGBoost round-sensitivity analysis complete.\n"
)

cat(
  "============================================================\n"
)


cat(
  "Maximum boosting rounds:",
  XGB_NROUNDS_SENSITIVITY,
  "\n"
)

cat(
  "Early-stopping patience:",
  XGB_EARLY_STOPPING_SENSITIVITY,
  "\n"
)


cat(
  "\nNested-CV sensitivity performance:\n"
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
  "\nFold-averaged sensitivity performance:\n"
)

print(
  performance_fold_summary,
  n = Inf
)


cat(
  "\nLeakage safeguards:\n"
)

cat(
  paste(
    "  1. The same fixed outer folds from",
    "07_define_resampling.R were used.\n"
  )
)

cat(
  paste(
    "  2. Expanded-round tuning occurred only",
    "inside each outer-training partition.\n"
  )
)

cat(
  paste(
    "  3. Early stopping used only inner-validation",
    "observations, never outer-validation observations.\n"
  )
)

cat(
  paste(
    "  4. Feature filtering, imputation, and dummy",
    "encoding were fit separately within each",
    "training partition.\n"
  )
)

cat(
  paste(
    "  5. The structural XGBoost hyperparameter grid",
    "was unchanged from the primary analysis.\n"
  )
)

cat(
  "  6. Set B was not accessed.\n"
)


cat(
  "\nGenerated outputs:\n"
)

cat(
  "  output/xgb_round_sensitivity_predictions.csv\n"
)

cat(
  "  output/xgb_round_sensitivity_performance.csv\n"
)

cat(
  "  output/xgb_round_sensitivity_by_fold.csv\n"
)

cat(
  "  output/xgb_round_sensitivity_tuning.csv\n"
)

cat(
  "  output/xgb_round_sensitivity_selected_iterations.csv\n"
)

cat(
  "  output/xgb_round_sensitivity_iteration_summary.csv\n"
)

cat(
  "  output/xgb_round_sensitivity_comparison.csv\n"
)

cat(
  "  output/xgb_round_sensitivity_final_configuration.csv\n"
)
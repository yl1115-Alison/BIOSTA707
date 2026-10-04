# BIOSTAT 707 Checkpoint 2
# 10_tree_boosting_models.R
#
# Nested-CV comparison of:
#   1. Random Forest
#   2. Gradient Boosting (XGBoost)
#
# Outer folds:
#   fixed folds from 07_define_resampling.R
#
# Inner folds:
#   stratified 5-fold CV inside each outer-training partition
#
# Leakage control:
#   - feature filtering: training data only
#   - imputation: training data only
#   - dummy encoding: training-data schema
#   - hyperparameter tuning: inner CV only
#   - no Set B access
#
# Tree models do not require standardization.

library(readr)
library(dplyr)
library(here)
library(pROC)
library(ranger)
library(xgboost)

N_STAYS <- 4000
N_OUTER_FOLDS <- 5
N_INNER_FOLDS <- 5
OUTCOME <- "In-hospital_death"
INNER_SEED <- 20261005
MAX_MISSING_PROPORTION <- 0.80

# Keep computation reproducible.
N_THREADS <- 1

# =============================================================================
# 1. Load data
# =============================================================================

dat <- read_csv(
  here("output", "set-a_features.csv"),
  show_col_types = FALSE
)

folds <- read_csv(
  here("output", "set-a_cv_folds.csv"),
  show_col_types = FALSE
)

feature_dictionary <- read_csv(
  here("output", "feature_dictionary.csv"),
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
  left_join(folds, by = "RecordID")

stopifnot(!any(is.na(dat$fold)))

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
  all(all_predictors %in% feature_dictionary$feature)
)

CATEGORICAL_FEATURES <- "ICUType"

cat("Set A ICU stays:", nrow(dat), "\n")
cat("Deaths:", sum(dat[[OUTCOME]]), "\n")
cat("Engineered predictors:", length(all_predictors), "\n")


# =============================================================================
# 2. Metrics
# =============================================================================

clip_probability <- function(p) {
  pmin(pmax(p, 1e-6), 1 - 1e-6)
}

calc_auc <- function(y, p) {
  r <- pROC::roc(
    response = y,
    predictor = p,
    levels = c(0, 1),
    direction = "<",
    quiet = TRUE
  )
  as.numeric(pROC::auc(r))
}

calc_brier <- function(y, p) {
  mean((y - p)^2)
}

calc_log_loss <- function(y, p) {
  p <- clip_probability(p)
  -mean(
    y * log(p) +
      (1 - y) * log(1 - p)
  )
}

metric_row <- function(pred, model, fold = NA_integer_) {
  tibble(
    model = model,
    fold = fold,
    n = nrow(pred),
    deaths = sum(pred$outcome == 1),
    auc = calc_auc(pred$outcome, pred$probability),
    brier = calc_brier(pred$outcome, pred$probability),
    log_loss = calc_log_loss(pred$outcome, pred$probability)
  )
}


# =============================================================================
# 3. Stratified inner folds
# =============================================================================

make_stratified_folds <- function(y, k, seed) {

  set.seed(seed)

  fold_id <- integer(length(y))

  for (value in c(0, 1)) {

    idx <- which(y == value)
    idx <- sample(idx)

    fold_id[idx] <- rep(
      seq_len(k),
      length.out = length(idx)
    )
  }

  stopifnot(all(fold_id %in% seq_len(k)))
  fold_id
}


# =============================================================================
# 4. Leakage-safe preprocessing
# =============================================================================

fit_mode <- function(x) {

  x <- x[!is.na(x)]

  if (length(x) == 0) {
    stop("Cannot estimate mode from an entirely missing variable.")
  }

  tab <- table(x)
  names(tab)[which.max(tab)]
}


fit_feature_filter <- function(training_data) {

  missing_prop <- vapply(
    all_predictors,
    function(feature) {
      mean(is.na(training_data[[feature]]))
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
    grepl("^missingness:", feature_types) |
    grepl("^measurement:count", feature_types)

  eligible <-
    missing_prop < MAX_MISSING_PROPORTION |
    protected

  candidates <- all_predictors[eligible]

  has_variation <- vapply(
    candidates,
    function(feature) {
      x <- training_data[[feature]]
      x <- x[!is.na(x)]
      length(unique(x)) > 1
    },
    logical(1)
  )

  list(
    selected = candidates[has_variation],
    excluded_high_missing = all_predictors[!eligible],
    excluded_zero_variance = candidates[!has_variation],
    missing_proportion = missing_prop
  )
}


fit_imputer <- function(training_data, features) {

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
      x <- x[!is.na(x) & is.finite(x)]

      if (length(x) == 0) {
        return(NA_real_)
      }

      median(x)
    },
    numeric(1)
  )

  if (any(is.na(medians))) {
    stop(
      paste(
        "Unable to estimate median for:",
        paste(names(medians)[is.na(medians)], collapse = ", ")
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
    medians = medians,
    modes = modes
  )
}


apply_imputer <- function(data, imputer) {

  out <- data

  for (feature in names(imputer$medians)) {

    idx <- is.na(out[[feature]])

    out[[feature]][idx] <-
      imputer$medians[[feature]]
  }

  for (feature in names(imputer$modes)) {

    idx <- is.na(out[[feature]])

    out[[feature]][idx] <-
      as.numeric(imputer$modes[[feature]])
  }

  out
}


fit_design_schema <- function(data, features) {

  working <- data |>
    select(all_of(features))

  if ("ICUType" %in% features) {
    working$ICUType <- factor(
      working$ICUType,
      levels = 1:4
    )
  }

  x <- model.matrix(
    ~ .,
    data = working
  )

  x <- x[
    ,
    colnames(x) != "(Intercept)",
    drop = FALSE
  ]

  list(columns = colnames(x))
}


make_design_matrix <- function(
  data,
  features,
  schema
) {

  working <- data |>
    select(all_of(features))

  if ("ICUType" %in% features) {
    working$ICUType <- factor(
      working$ICUType,
      levels = 1:4
    )
  }

  x <- model.matrix(
    ~ .,
    data = working
  )

  x <- x[
    ,
    colnames(x) != "(Intercept)",
    drop = FALSE
  ]

  missing_columns <- setdiff(
    schema$columns,
    colnames(x)
  )

  if (length(missing_columns) > 0) {

    z <- matrix(
      0,
      nrow = nrow(x),
      ncol = length(missing_columns)
    )

    colnames(z) <- missing_columns

    x <- cbind(x, z)
  }

  extra_columns <- setdiff(
    colnames(x),
    schema$columns
  )

  if (length(extra_columns) > 0) {
    x <- x[
      ,
      setdiff(colnames(x), extra_columns),
      drop = FALSE
    ]
  }

  x <- x[
    ,
    schema$columns,
    drop = FALSE
  ]

  stopifnot(all(is.finite(x)))

  x
}


fit_preprocessor <- function(training_data) {

  filter_obj <- fit_feature_filter(
    training_data
  )

  features <- filter_obj$selected

  imputer <- fit_imputer(
    training_data,
    features
  )

  imputed <- apply_imputer(
    training_data,
    imputer
  )

  schema <- fit_design_schema(
    imputed,
    features
  )

  list(
    features = features,
    filter = filter_obj,
    imputer = imputer,
    schema = schema
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
# 5. Compact tuning grids
# =============================================================================
#
# We intentionally use modest prespecified grids.
# The objective is statistically valid comparison, not exhaustive search.

RF_GRID <- expand.grid(
  mtry_fraction = c(
    0.10,
    0.25,
    0.50
  ),
  min_node_size = c(
    5,
    15,
    30
  ),
  stringsAsFactors = FALSE
)

RF_NUM_TREES <- 500


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
  stringsAsFactors = FALSE
)

XGB_NROUNDS <- 400
XGB_EARLY_STOPPING <- 30


# =============================================================================
# 6. Random Forest inner-CV tuning
# =============================================================================

tune_rf <- function(
  outer_training,
  inner_fold_id
) {

  results <- list()
  result_index <- 1L

  for (grid_id in seq_len(nrow(RF_GRID))) {

    config <- RF_GRID[
      grid_id,
      ,
      drop = FALSE
    ]

    fold_scores <- list()

    for (inner_fold in seq_len(N_INNER_FOLDS)) {

      train <- outer_training[
        inner_fold_id != inner_fold,
        ,
        drop = FALSE
      ]

      validation <- outer_training[
        inner_fold_id == inner_fold,
        ,
        drop = FALSE
      ]

      prep <- fit_preprocessor(train)

      x_train <- apply_preprocessor(
        train,
        prep
      )

      x_validation <- apply_preprocessor(
        validation,
        prep
      )

      y_train <- train[[OUTCOME]]
      y_validation <- validation[[OUTCOME]]

      p <- ncol(x_train)

      mtry_value <- max(
        1L,
        min(
          p,
          as.integer(
            round(
              config$mtry_fraction *
                p
            )
          )
        )
      )

      train_df <- as.data.frame(x_train)
      train_df$outcome_factor <- factor(
        y_train,
        levels = c(0, 1)
      )

      validation_df <- as.data.frame(
        x_validation
      )

      fit <- ranger(
        outcome_factor ~ .,
        data = train_df,
        probability = TRUE,
        num.trees = RF_NUM_TREES,
        mtry = mtry_value,
        min.node.size = config$min_node_size,
        seed = INNER_SEED +
          grid_id * 100 +
          inner_fold,
        num.threads = N_THREADS
      )

      pred_matrix <- predict(
        fit,
        data = validation_df
      )$predictions

      probability <- pred_matrix[, "1"]

      fold_scores[[inner_fold]] <- tibble(
        grid_id = grid_id,
        inner_fold = inner_fold,
        mtry_fraction = config$mtry_fraction,
        mtry = mtry_value,
        min_node_size = config$min_node_size,
        auc = calc_auc(
          y_validation,
          probability
        ),
        brier = calc_brier(
          y_validation,
          probability
        ),
        log_loss = calc_log_loss(
          y_validation,
          probability
        )
      )
    }

    results[[result_index]] <- bind_rows(
      fold_scores
    )

    result_index <- result_index + 1L
  }

  fold_results <- bind_rows(
    results
  )

  summary <- fold_results |>
    summarise(
      mean_auc = mean(auc),
      sd_auc = sd(auc),
      mean_brier = mean(brier),
      mean_log_loss = mean(log_loss),
      .by = c(
        grid_id,
        mtry_fraction,
        min_node_size
      )
    ) |>
    arrange(
      desc(mean_auc),
      mean_log_loss,
      mean_brier
    )

  list(
    best = summary |>
      slice_head(n = 1),
    summary = summary,
    fold_results = fold_results
  )
}


# =============================================================================
# 7. XGBoost inner-CV tuning
# =============================================================================

tune_xgb <- function(
  outer_training,
  inner_fold_id
) {

  results <- list()
  result_index <- 1L

  for (grid_id in seq_len(nrow(XGB_GRID))) {

    config <- XGB_GRID[
      grid_id,
      ,
      drop = FALSE
    ]

    fold_scores <- list()

    for (inner_fold in seq_len(N_INNER_FOLDS)) {

      train <- outer_training[
        inner_fold_id != inner_fold,
        ,
        drop = FALSE
      ]

      validation <- outer_training[
        inner_fold_id == inner_fold,
        ,
        drop = FALSE
      ]

      prep <- fit_preprocessor(train)

      x_train <- apply_preprocessor(
        train,
        prep
      )

      x_validation <- apply_preprocessor(
        validation,
        prep
      )

      y_train <- train[[OUTCOME]]
      y_validation <- validation[[OUTCOME]]

      dtrain <- xgb.DMatrix(
        data = x_train,
        label = y_train
      )

      dvalidation <- xgb.DMatrix(
        data = x_validation,
        label = y_validation
      )

      fit <- xgb.train(
        params = list(
          objective = "binary:logistic",
          eval_metric = "auc",
          max_depth = config$max_depth,
          eta = config$eta,
          min_child_weight =
            config$min_child_weight,
          subsample = 0.8,
          colsample_bytree = 0.8,
          nthread = N_THREADS
        ),
        data = dtrain,
        nrounds = XGB_NROUNDS,
        watchlist = list(
          validation = dvalidation
        ),
        early_stopping_rounds =
          XGB_EARLY_STOPPING,
        verbose = 0
      )

      probability <- predict(
        fit,
        dvalidation
      )

      best_iteration <- fit$best_iteration

      if (
        is.null(best_iteration) ||
          length(best_iteration) == 0
      ) {
        best_iteration <- XGB_NROUNDS
      }

      fold_scores[[inner_fold]] <- tibble(
        grid_id = grid_id,
        inner_fold = inner_fold,
        max_depth = config$max_depth,
        eta = config$eta,
        min_child_weight =
          config$min_child_weight,
        best_iteration =
          best_iteration,
        auc = calc_auc(
          y_validation,
          probability
        ),
        brier = calc_brier(
          y_validation,
          probability
        ),
        log_loss = calc_log_loss(
          y_validation,
          probability
        )
      )
    }

    results[[result_index]] <- bind_rows(
      fold_scores
    )

    result_index <- result_index + 1L
  }

  fold_results <- bind_rows(
    results
  )

  summary <- fold_results |>
    summarise(
      mean_auc = mean(auc),
      sd_auc = sd(auc),
      mean_brier = mean(brier),
      mean_log_loss = mean(log_loss),

      median_best_iteration =
        as.integer(
          round(
            median(
              best_iteration
            )
          )
        ),

      .by = c(
        grid_id,
        max_depth,
        eta,
        min_child_weight
      )
    ) |>
    arrange(
      desc(mean_auc),
      mean_log_loss,
      mean_brier
    )

  list(
    best = summary |>
      slice_head(n = 1),
    summary = summary,
    fold_results = fold_results
  )
}


# =============================================================================
# 8. Fit selected RF on one outer fold
# =============================================================================

fit_outer_rf <- function(
  outer_training,
  outer_validation,
  tuning,
  outer_fold
) {

  prep <- fit_preprocessor(
    outer_training
  )

  x_train <- apply_preprocessor(
    outer_training,
        prep
  )

  x_validation <- apply_preprocessor(
    outer_validation,
    prep
  )

  y_train <- outer_training[[OUTCOME]]

  p <- ncol(x_train)

  mtry_value <- max(
    1L,
    min(
      p,
      as.integer(
        round(
          tuning$mtry_fraction[[1]] *
            p
        )
      )
    )
  )

  train_df <- as.data.frame(
    x_train
  )

  train_df$outcome_factor <- factor(
    y_train,
    levels = c(0, 1)
  )

  validation_df <- as.data.frame(
    x_validation
  )

  fit <- ranger(
    outcome_factor ~ .,
    data = train_df,
    probability = TRUE,
    num.trees = RF_NUM_TREES,
    mtry = mtry_value,
    min.node.size =
      tuning$min_node_size[[1]],
    seed =
      INNER_SEED +
        10000 +
        outer_fold,
    num.threads = N_THREADS,
    importance = "permutation"
  )

  pred_matrix <- predict(
    fit,
    data = validation_df
  )$predictions

  probability <- pred_matrix[, "1"]

  stopifnot(
    length(probability) ==
      nrow(outer_validation),

    all(
      is.finite(
        probability
      )
    ),

    all(
      probability >= 0 &
        probability <= 1
    )
  )

  importance <- tibble(
    feature =
      names(
        fit$variable.importance
      ),

    importance =
      as.numeric(
        fit$variable.importance
      ),

    outer_fold =
      outer_fold,

    model =
      "Random Forest"
  ) |>
    arrange(
      desc(
        importance
      )
    )

  predictions <- tibble(
    RecordID =
      outer_validation$RecordID,

    fold =
      outer_fold,

    outcome =
      outer_validation[[OUTCOME]],

    model =
      "Random Forest",

    probability =
      as.numeric(
        probability
      )
  )

  preprocessing <- tibble(
    outer_fold =
      outer_fold,

    model =
      "Random Forest",

    n_raw_predictors =
      length(
        all_predictors
      ),

    n_features_after_filter =
      length(
        prep$features
      ),

    n_design_columns =
      ncol(
        x_train
      ),

    n_excluded_high_missing =
      length(
        prep$filter$excluded_high_missing
      ),

    n_excluded_zero_variance =
      length(
        prep$filter$excluded_zero_variance
      ),

    mtry =
      mtry_value,

    min_node_size =
      tuning$min_node_size[[1]]
  )

  list(
    predictions =
      predictions,

    importance =
      importance,

    preprocessing =
      preprocessing
  )
}


# =============================================================================
# 9. Fit selected XGBoost model on one outer fold
# =============================================================================

fit_outer_xgb <- function(
  outer_training,
  outer_validation,
  tuning,
  outer_fold
) {

  prep <- fit_preprocessor(
    outer_training
  )

  x_train <- apply_preprocessor(
    outer_training,
    prep
  )

  x_validation <- apply_preprocessor(
    outer_validation,
    prep
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
    length(probability) ==
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

  importance_raw <- xgb.importance(
    model =
      fit
  )

  if (
    nrow(
      importance_raw
    ) >
      0
  ) {

    importance <- importance_raw |>
      as_tibble() |>
      transmute(
        feature =
          Feature,

        importance =
          Gain,

        outer_fold =
          outer_fold,

        model =
          "Gradient Boosting"
      ) |>
      arrange(
        desc(
          importance
        )
      )

  } else {

    importance <- tibble(
      feature =
        character(),

      importance =
        numeric(),

      outer_fold =
        integer(),

      model =
        character()
    )
  }

  predictions <- tibble(
    RecordID =
      outer_validation$RecordID,

    fold =
      outer_fold,

    outcome =
      outer_validation[[OUTCOME]],

    model =
      "Gradient Boosting",

    probability =
      as.numeric(
        probability
      )
  )

  preprocessing <- tibble(
    outer_fold =
      outer_fold,

    model =
      "Gradient Boosting",

    n_raw_predictors =
      length(
        all_predictors
      ),

    n_features_after_filter =
      length(
        prep$features
      ),

    n_design_columns =
      ncol(
        x_train
      ),

    n_excluded_high_missing =
      length(
        prep$filter$excluded_high_missing
      ),

    n_excluded_zero_variance =
      length(
        prep$filter$excluded_zero_variance
      ),

    max_depth =
      tuning$max_depth[[1]],

    eta =
      tuning$eta[[1]],

    min_child_weight =
      tuning$min_child_weight[[1]],

    nrounds =
      nrounds_final
  )

  list(
    predictions =
      predictions,

    importance =
      importance,

    preprocessing =
      preprocessing
  )
}


# =============================================================================
# 10. Outer nested-CV loop
# =============================================================================

outer_predictions <- list()
outer_tuning <- list()
outer_importance <- list()
outer_preprocessing <- list()


for (outer_fold in seq_len(N_OUTER_FOLDS)) {

  cat(
    "\n============================================================\n"
  )

  cat(
    "Outer fold",
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
      fold != outer_fold
    )

  outer_validation <- dat |>
    filter(
      fold == outer_fold
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
  # Define inner folds using outer-training data only
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
  # Random Forest tuning
  # ---------------------------------------------------------------------------

  cat(
    "\n  Tuning Random Forest...\n"
  )

  rf_tuning <- tune_rf(
    outer_training =
      outer_training,

    inner_fold_id =
      inner_fold_id
  )

  rf_best <- rf_tuning$best |>
    mutate(
      outer_fold =
        outer_fold,

      model =
        "Random Forest"
    )


  cat(
    "  Selected Random Forest configuration:\n"
  )

  print(
    rf_best,
    n = Inf
  )


  # ---------------------------------------------------------------------------
  # Gradient Boosting tuning
  # ---------------------------------------------------------------------------

  cat(
    "\n  Tuning Gradient Boosting...\n"
  )

  xgb_tuning <- tune_xgb(
    outer_training =
      outer_training,

    inner_fold_id =
      inner_fold_id
  )

  xgb_best <- xgb_tuning$best |>
    mutate(
      outer_fold =
        outer_fold,

      model =
        "Gradient Boosting"
    )


  cat(
    "  Selected Gradient Boosting configuration:\n"
  )

  print(
    xgb_best,
    n = Inf
  )


  # ---------------------------------------------------------------------------
  # Save tuning choices
  # ---------------------------------------------------------------------------

  rf_tuning_row <- rf_best |>
    transmute(
      outer_fold,
      model,
      mtry_fraction,
      min_node_size,
      max_depth =
        NA_real_,
      eta =
        NA_real_,
      min_child_weight =
        NA_real_,
      nrounds =
        NA_real_,
      mean_auc,
      sd_auc,
      mean_brier,
      mean_log_loss
    )


  xgb_tuning_row <- xgb_best |>
    transmute(
      outer_fold,
      model,
      mtry_fraction =
        NA_real_,
      min_node_size =
        NA_real_,
      max_depth =
        as.numeric(
          max_depth
        ),
      eta =
        as.numeric(
          eta
        ),
      min_child_weight =
        as.numeric(
          min_child_weight
        ),
      nrounds =
        as.numeric(
          median_best_iteration
        ),
      mean_auc,
      sd_auc,
      mean_brier,
      mean_log_loss
    )


  outer_tuning[[outer_fold]] <- bind_rows(
    rf_tuning_row,
    xgb_tuning_row
  )


  # ---------------------------------------------------------------------------
  # Fit selected RF on complete outer-training set
  # ---------------------------------------------------------------------------

  cat(
    "\n  Fitting selected Random Forest...\n"
  )

  rf_result <- fit_outer_rf(
    outer_training =
      outer_training,

    outer_validation =
      outer_validation,

    tuning =
      rf_best,

    outer_fold =
      outer_fold
  )


  # ---------------------------------------------------------------------------
  # Fit selected XGBoost on complete outer-training set
  # ---------------------------------------------------------------------------

  cat(
    "  Fitting selected Gradient Boosting model...\n"
  )

  xgb_result <- fit_outer_xgb(
    outer_training =
      outer_training,

    outer_validation =
      outer_validation,

    tuning =
      xgb_best,

    outer_fold =
      outer_fold
  )


  # ---------------------------------------------------------------------------
  # Store outer-fold results
  # ---------------------------------------------------------------------------

  outer_predictions[[outer_fold]] <- bind_rows(
    rf_result$predictions,
    xgb_result$predictions
  )

  outer_importance[[outer_fold]] <- bind_rows(
    rf_result$importance,
    xgb_result$importance
  )

  outer_preprocessing[[outer_fold]] <- bind_rows(
    rf_result$preprocessing,
    xgb_result$preprocessing
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

all_importance <- bind_rows(
  outer_importance
)

all_preprocessing <- bind_rows(
  outer_preprocessing
)


expected_models <- c(
  "Random Forest",
  "Gradient Boosting"
)


# =============================================================================
# 12. Validate out-of-fold predictions
# =============================================================================

for (model_name in expected_models) {

  model_predictions <- all_predictions |>
    filter(
      model == model_name
    )

  stopifnot(
    nrow(model_predictions) ==
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
    ),

    all(
      is.finite(
        model_predictions$probability
      )
    ),

    all(
      model_predictions$probability >= 0 &
        model_predictions$probability <= 1
    )
  )
}


# =============================================================================
# 13. Overall nested-CV performance
# =============================================================================

overall_performance <- lapply(
  expected_models,
  function(model_name) {

    predictions <- all_predictions |>
      filter(
        model == model_name
      )

    metric_row(
      pred =
        predictions,

      model =
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
  "\n============================================================\n"
)

cat(
  "Tree / boosting nested-CV performance\n"
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
  expected_models,
  function(model_name) {

    lapply(
      seq_len(
        N_OUTER_FOLDS
      ),
      function(k) {

        predictions <- all_predictions |>
          filter(
            model ==
              model_name,
            fold ==
              k
          )

        metric_row(
          pred =
            predictions,

          model =
            model_name,

          fold =
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
      ),

    .by =
      model
  ) |>
  arrange(
    desc(
      mean_auc
    )
  )


# =============================================================================
# 15. Aggregate feature importance
# =============================================================================
#
# Importance scales differ between RF and XGBoost, so normalize within each
# model/fold before averaging.
#
# These importance results are descriptive, not inferential.


importance_normalized <- all_importance |>
  group_by(
    model,
    outer_fold
  ) |>
  mutate(
    normalized_importance =
      if (
        sum(
          abs(
            importance
          ),
          na.rm = TRUE
        ) >
          0
      ) {

        abs(
          importance
        ) /
          sum(
            abs(
              importance
            ),
            na.rm = TRUE
          )

      } else {

        0
      }
  ) |>
  ungroup()


importance_summary <- importance_normalized |>
  summarise(
    folds_present =
      n_distinct(
        outer_fold
      ),

    mean_normalized_importance =
      mean(
        normalized_importance,
        na.rm = TRUE
      ),

    sd_normalized_importance =
      sd(
        normalized_importance,
        na.rm = TRUE
      ),

    .by =
      c(
        model,
        feature
      )
  ) |>
  arrange(
    model,
    desc(
      mean_normalized_importance
    )
  )


# =============================================================================
# 16. Save tree / boosting outputs
# =============================================================================

write_csv(
  all_predictions,
  here(
    "output",
    "cv_predictions_tree_boosting.csv"
  )
)

write_csv(
  overall_performance,
  here(
    "output",
    "cv_performance_tree_boosting.csv"
  )
)

write_csv(
  performance_by_fold,
  here(
    "output",
    "cv_performance_tree_boosting_by_fold.csv"
  )
)

write_csv(
  performance_fold_summary,
  here(
    "output",
    "cv_performance_tree_boosting_fold_summary.csv"
  )
)

write_csv(
  all_tuning,
  here(
    "output",
    "tree_boosting_tuning_by_fold.csv"
  )
)

write_csv(
  all_preprocessing,
  here(
    "output",
    "tree_boosting_preprocessing_by_fold.csv"
  )
)

write_csv(
  importance_normalized,
  here(
    "output",
    "tree_boosting_importance_by_fold.csv"
  )
)

write_csv(
  importance_summary,
  here(
    "output",
    "tree_boosting_importance_summary.csv"
  )
)


# =============================================================================
# 17. Combine with previous model results
# =============================================================================

previous_path <- here(
  "output",
  "cv_model_comparison_through_regularized.csv"
)

if (
  file.exists(
    previous_path
  )
) {

  previous <- read_csv(
    previous_path,
    show_col_types = FALSE
  )

  complete_comparison <- bind_rows(
    previous |>
      select(
        model,
        auc,
        brier,
        log_loss
      ),

    overall_performance |>
      select(
        model,
        auc,
        brier,
        log_loss
      )
  ) |>
    arrange(
      desc(
        auc
      )
    )

  write_csv(
    complete_comparison,
    here(
      "output",
      "cv_model_comparison_all.csv"
    )
  )

  cat(
    "\nComplete Set A model comparison:\n"
  )

  print(
    complete_comparison,
    n = Inf
  )
}


# =============================================================================
# 18. Final Set-A tuning for Random Forest
# =============================================================================
#
# Nested CV above estimates development performance.
#
# Now use all Set A for final hyperparameter selection. These configurations
# can later be frozen before the single-look Set B evaluation.


cat(
  "\n============================================================\n"
)

cat(
  "Final full-Set-A tuning\n"
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


cat(
  "Tuning final Random Forest...\n"
)

final_rf_tuning <- tune_rf(
      outer_training =
    dat,

  inner_fold_id =
    final_inner_fold_id
)


final_rf <- final_rf_tuning$best |>
  mutate(
    model =
      "Random Forest"
  )


cat(
  "\nFinal Random Forest configuration:\n"
)

print(
  final_rf,
  n = Inf
)


# =============================================================================
# 19. Final Set-A tuning for Gradient Boosting
# =============================================================================

cat(
  "\nTuning final Gradient Boosting model...\n"
)


final_xgb_tuning <- tune_xgb(
  outer_training =
    dat,

  inner_fold_id =
    final_inner_fold_id
)


final_xgb <- final_xgb_tuning$best |>
  mutate(
    model =
      "Gradient Boosting"
  )


cat(
  "\nFinal Gradient Boosting configuration:\n"
)

print(
  final_xgb,
  n = Inf
)


# =============================================================================
# 20. Save final Set-A configurations
# =============================================================================

final_rf_configuration <- final_rf |>
  transmute(
    model,
    mtry_fraction,
    min_node_size,
    max_depth =
      NA_real_,
    eta =
      NA_real_,
    min_child_weight =
      NA_real_,
    nrounds =
      NA_real_,
    mean_auc,
    sd_auc,
    mean_brier,
    mean_log_loss
  )


final_xgb_configuration <- final_xgb |>
  transmute(
    model,
    mtry_fraction =
      NA_real_,
    min_node_size =
      NA_real_,
    max_depth =
      as.numeric(
        max_depth
      ),
    eta =
      as.numeric(
        eta
      ),
    min_child_weight =
      as.numeric(
        min_child_weight
      ),
    nrounds =
      as.numeric(
        median_best_iteration
      ),
    mean_auc,
    sd_auc,
    mean_brier,
    mean_log_loss
  )


final_configuration <- bind_rows(
  final_rf_configuration,
  final_xgb_configuration
)


write_csv(
  final_configuration,
  here(
    "output",
    "tree_boosting_final_configuration_set_a.csv"
  )
)


# =============================================================================
# 21. Fit final preprocessing on all Set A
# =============================================================================
#
# This preprocessing is learned only from Set A.
#
# Nothing is estimated from Set B.


final_preprocessor <- fit_preprocessor(
  dat
)


x_final <- apply_preprocessor(
  dat,
  final_preprocessor
)


y_final <- dat[[OUTCOME]]


cat(
  "\nFinal Set-A preprocessing:\n"
)

cat(
  "  Raw engineered predictors:",
  length(
    all_predictors
  ),
  "\n"
)

cat(
  "  Features retained:",
  length(
    final_preprocessor$features
  ),
  "\n"
)

cat(
  "  Design-matrix columns:",
  ncol(
    x_final
  ),
  "\n"
)

cat(
  "  Excluded for >=80% missingness:",
  length(
    final_preprocessor$filter$
      excluded_high_missing
  ),
  "\n"
)

cat(
  "  Excluded for zero variance:",
  length(
    final_preprocessor$filter$
      excluded_zero_variance
  ),
  "\n"
)


# =============================================================================
# 22. Fit final Random Forest on all Set A
# =============================================================================

final_rf_mtry <- max(
  1L,
  min(
    ncol(
      x_final
    ),
    as.integer(
      round(
        final_rf$mtry_fraction[[1]] *
          ncol(
            x_final
          )
      )
    )
  )
)


rf_final_data <- as.data.frame(
  x_final
)


rf_final_data$outcome_factor <- factor(
  y_final,
  levels =
    c(0, 1)
)


cat(
  "\nFitting final Random Forest on all Set A...\n"
)


final_rf_fit <- ranger(
  outcome_factor ~ .,
  data =
    rf_final_data,
  probability =
    TRUE,
  num.trees =
    RF_NUM_TREES,
  mtry =
    final_rf_mtry,
  min.node.size =
    final_rf$min_node_size[[1]],
  seed =
    INNER_SEED +
      20000,
  num.threads =
    N_THREADS,
  importance =
    "permutation"
)


final_rf_importance <- tibble(
  feature =
    names(
      final_rf_fit$
        variable.importance
    ),
  importance =
    as.numeric(
      final_rf_fit$
        variable.importance
    ),
  model =
    "Random Forest"
) |>
  arrange(
    desc(
      importance
    )
  )


# =============================================================================
# 23. Fit final Gradient Boosting model on all Set A
# =============================================================================

final_xgb_nrounds <- max(
  1L,
  as.integer(
    final_xgb$
      median_best_iteration[[1]]
  )
)


dtrain_final <- xgb.DMatrix(
  data =
    x_final,
  label =
    y_final
)


cat(
  "Fitting final Gradient Boosting model on all Set A...\n"
)


final_xgb_fit <- xgb.train(
  params = list(
    objective =
      "binary:logistic",
    eval_metric =
      "auc",
    max_depth =
      final_xgb$max_depth[[1]],
    eta =
      final_xgb$eta[[1]],
    min_child_weight =
      final_xgb$
        min_child_weight[[1]],
    subsample =
      0.8,
    colsample_bytree =
      0.8,
    nthread =
      N_THREADS
  ),
  data =
    dtrain_final,
  nrounds =
    final_xgb_nrounds,
  verbose =
    0
)


final_xgb_importance_raw <- xgb.importance(
  model =
    final_xgb_fit
)


if (
  nrow(
    final_xgb_importance_raw
  ) >
    0
) {

  final_xgb_importance <- final_xgb_importance_raw |>
    as_tibble() |>
    transmute(
      feature =
        Feature,
      importance =
        Gain,
      model =
        "Gradient Boosting"
    ) |>
    arrange(
      desc(
        importance
      )
    )

} else {

  final_xgb_importance <- tibble(
    feature =
      character(),
    importance =
      numeric(),
    model =
      character()
  )
}


# =============================================================================
# 24. Save final feature importance
# =============================================================================

final_importance <- bind_rows(
  final_rf_importance,
  final_xgb_importance
)


write_csv(
  final_importance,
  here(
    "output",
    "tree_boosting_final_importance_set_a.csv"
  )
)


# =============================================================================
# 25. Save final preprocessing specification
# =============================================================================

final_feature_status <- tibble(
  feature =
    all_predictors,

  retained =
    all_predictors %in%
      final_preprocessor$features,

  excluded_high_missing =
    all_predictors %in%
      final_preprocessor$filter$
        excluded_high_missing,

  excluded_zero_variance =
    all_predictors %in%
      final_preprocessor$filter$
        excluded_zero_variance,

  set_a_missing_proportion =
    as.numeric(
      final_preprocessor$filter$
        missing_proportion[
          all_predictors
        ]
    )
)


write_csv(
  final_feature_status,
  here(
    "output",
    "tree_boosting_final_feature_status.csv"
  )
)


final_numeric_imputation <- tibble(
  feature =
    names(
      final_preprocessor$imputer$
        medians
    ),

  median_set_a =
    as.numeric(
      final_preprocessor$imputer$
        medians
    )
)


write_csv(
  final_numeric_imputation,
  here(
    "output",
    "tree_boosting_final_imputation.csv"
  )
)


if (
  length(
    final_preprocessor$imputer$
      modes
  ) >
    0
) {

  final_categorical_imputation <- tibble(
    feature =
      names(
        final_preprocessor$imputer$
          modes
      ),

    mode_set_a =
      as.numeric(
        unlist(
          final_preprocessor$imputer$
            modes
        )
      )
  )

} else {

  final_categorical_imputation <- tibble(
    feature =
      character(),
    mode_set_a =
      numeric()
  )
}


write_csv(
  final_categorical_imputation,
  here(
    "output",
    "tree_boosting_final_categorical_imputation.csv"
  )
)


final_design_schema <- tibble(
  design_column =
    final_preprocessor$
      schema$
      columns
)


write_csv(
  final_design_schema,
  here(
    "output",
    "tree_boosting_final_design_schema.csv"
  )
)


# =============================================================================
# 26. Validate final results
# =============================================================================

stopifnot(
  nrow(
    all_predictions
  ) ==
    N_STAYS *
      length(
        expected_models
      ),

  all(
    is.finite(
      all_predictions$
        probability
    )
  ),

  all(
    all_predictions$
      probability >=
      0 &
      all_predictions$
        probability <=
      1
  ),

  nrow(
    final_configuration
  ) ==
    2,

  all(
    expected_models %in%
      final_configuration$
        model
  )
)


prediction_check <- all_predictions |>
  count(
    RecordID,
    model,
    name =
      "n_predictions"
  )


stopifnot(
  nrow(
    prediction_check
  ) ==
    N_STAYS *
      length(
        expected_models
      ),

  all(
    prediction_check$
      n_predictions ==
      1
  )
)


# =============================================================================
# 27. Final report
# =============================================================================

cat(
  "\n============================================================\n"
)

cat(
  "Checkpoint 2 tree / boosting modeling complete.\n"
)

cat(
  "============================================================\n"
)


cat(
  "\nNested-CV performance:\n"
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
  "\nFold-averaged nested-CV performance:\n"
)

print(
  performance_fold_summary,
  n = Inf
)


cat(
  "\nSelected configurations by outer fold:\n"
)

print(
  all_tuning |>
    arrange(
      model,
      outer_fold
    ),
  n = Inf
)


cat(
  "\nFinal full-Set-A configurations:\n"
)

print(
  final_configuration,
  n = Inf
)


cat(
  "\nLeakage safeguards:\n"
)

cat(
  paste(
    "  1. Outer evaluation used the fixed folds",
    "created in 07_define_resampling.R.\n"
  )
)

cat(
  paste(
    "  2. Hyperparameter tuning occurred only",
    "inside each outer-training partition.\n"
  )
)

cat(
  paste(
    "  3. Feature filtering, imputation, and",
    "dummy encoding were learned only from",
    "the relevant training partition.\n"
  )
)

cat(
  paste(
    "  4. Tree models were not standardized",
    "because tree splits are invariant to",
    "monotonic rescaling of predictors.\n"
  )
)

cat(
  paste(
    "  5. SAPS-I and SOFA were not included",
    "as engineered predictors.\n"
  )
)

cat(
  "  6. Set B was not accessed.\n"
)


cat(
  "\nGenerated outputs:\n"
)

cat(
  "  output/cv_predictions_tree_boosting.csv\n"
)

cat(
  "  output/cv_performance_tree_boosting.csv\n"
)

cat(
  "  output/cv_performance_tree_boosting_by_fold.csv\n"
)

cat(
  "  output/cv_performance_tree_boosting_fold_summary.csv\n"
)

cat(
  "  output/tree_boosting_tuning_by_fold.csv\n"
)

cat(
  "  output/tree_boosting_preprocessing_by_fold.csv\n"
)

cat(
  "  output/tree_boosting_importance_by_fold.csv\n"
)

cat(
  "  output/tree_boosting_importance_summary.csv\n"
)

cat(
  "  output/tree_boosting_final_configuration_set_a.csv\n"
)

cat(
  "  output/tree_boosting_final_importance_set_a.csv\n"
)

cat(
  "  output/tree_boosting_final_feature_status.csv\n"
)

cat(
  "  output/tree_boosting_final_imputation.csv\n"
)

cat(
  "  output/tree_boosting_final_categorical_imputation.csv\n"
)

cat(
  "  output/tree_boosting_final_design_schema.csv\n"
)

if (
  exists(
    "complete_comparison"
  )
) {

  cat(
    "  output/cv_model_comparison_all.csv\n"
  )
}
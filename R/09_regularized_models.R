# BIOSTAT 707 Checkpoint 2
# 09_regularized_models.R
#
# Purpose:
#   Compare regularized logistic-regression models using nested CV:
#
#     1. Ridge
#     2. Elastic Net
#     3. LASSO
#
# Resampling:
#   Outer loop:
#     fixed 5 folds from 07_define_resampling.R
#
#   Inner loop:
#     5-fold stratified CV within each outer-training set
#
# Leakage control:
#   All learned preprocessing is estimated from training data only:
#
#     - high-missingness feature filtering
#     - median imputation
#     - categorical imputation
#     - dummy-variable construction
#     - standardization
#     - alpha/lambda tuning
#
# Set B is not accessed.
#
# Inputs:
#   output/set-a_features.csv
#   output/set-a_cv_folds.csv
#   output/feature_dictionary.csv
#
# Outputs:
#   output/cv_predictions_regularized.csv
#   output/cv_performance_regularized.csv
#   output/cv_performance_regularized_by_fold.csv
#   output/regularized_tuning_by_fold.csv
#   output/regularized_selected_features.csv
#   output/regularized_feature_frequency.csv


# =============================================================================
# 1. Packages and constants
# =============================================================================

library(readr)
library(dplyr)
library(glmnet)
library(pROC)
library(here)

N_STAYS <- 4000
N_OUTER_FOLDS <- 5
N_INNER_FOLDS <- 5

OUTCOME <- "In-hospital_death"

INNER_SEED <- 20261004

# Review threshold established before regularized model fitting.
#
# Continuous/value features with >=80% missingness in the relevant training
# partition are excluded from that fitted preprocessing pipeline.
#
# Explicit missingness indicators and measurement counts are retained.

MAX_MISSING_PROPORTION <- 0.80

ALPHA_GRID <- c(
  0,
  0.25,
  0.50,
  0.75,
  1
)


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

  OUTCOME %in% names(dat),

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
# 3. Define engineered predictor space
# =============================================================================
#
# Exclude:
#
#   RecordID              identifier
#   fold                  resampling label
#   In-hospital_death     outcome
#   SAPS-I / SOFA         separate clinical benchmarks
#
# All remaining columns are engineered predictors from 05.


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
  "Engineered predictors available:",
  length(all_predictors),
  "\n"
)


# =============================================================================
# 4. Predictor roles
# =============================================================================
#
# ICUType is nominal and must be dummy encoded.
#
# Gender is binary numeric and is retained as 0/1.
#
# All other engineered columns are numeric.


CATEGORICAL_FEATURES <- c(
  "ICUType"
)

BINARY_FEATURES <- c(
  "Gender",
  grep(
    "_missing$",
    all_predictors,
    value = TRUE
  ),
  "MechVent_ever"
)

BINARY_FEATURES <- intersect(
  unique(
    BINARY_FEATURES
  ),
  all_predictors
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


calculate_metrics <- function(
  predictions,
  model_name,
  fold_value = NA_integer_
) {

  tibble(
    model = model_name,
    fold = fold_value,
    n = nrow(predictions),

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
# 6. Stratified inner-fold assignment
# =============================================================================

make_stratified_folds <- function(
  y,
  k,
  seed
) {

  stopifnot(
    all(
      y %in%
        c(0, 1)
    )
  )

  set.seed(
    seed
  )

  fold_id <- integer(
    length(y)
  )

  for (
    outcome_value in c(0, 1)
  ) {

    index <- which(
      y ==
        outcome_value
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
# 7. Preprocessing helpers
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

  tab <- table(x)

  names(
    tab
  )[
    which.max(
      tab
    )
  ]
}


# -----------------------------------------------------------------------------
# 7a. Determine eligible predictors from training data only
# -----------------------------------------------------------------------------

fit_feature_filter <- function(
  training_data,
  predictor_names
) {

  missing_proportion <- vapply(
    predictor_names,
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
      predictor_names,
      feature_dictionary$feature
    )
  ]


  # Always retain explicit missingness indicators and measurement counts,
  # provided they have variation.
  protected <- grepl(
    "^missingness:",
    feature_types
  ) |
    grepl(
      "^measurement:count",
      feature_types
    )


  eligible_missingness <-
    missing_proportion <
      MAX_MISSING_PROPORTION |
      protected


  candidate_features <- predictor_names[
    eligible_missingness
  ]


  # Remove zero-variance features using training data only.
  has_variation <- vapply(
    candidate_features,
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


  selected <- candidate_features[
    has_variation
  ]


  excluded_high_missing <- setdiff(
    predictor_names[
      !eligible_missingness
    ],
    character(0)
  )


  excluded_zero_variance <- candidate_features[
    !has_variation
  ]


  list(
    selected_features =
      selected,

    excluded_high_missing =
      excluded_high_missing,

    excluded_zero_variance =
      excluded_zero_variance,

    missing_proportion =
      missing_proportion
  )
}


# -----------------------------------------------------------------------------
# 7b. Fit imputation parameters
# -----------------------------------------------------------------------------

fit_imputation <- function(
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

      median(x)
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


  modes <- list()

  for (
    feature in categorical
  ) {

    modes[[feature]] <-
      fit_mode(
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


# -----------------------------------------------------------------------------
# 7c. Apply imputation
# -----------------------------------------------------------------------------

apply_imputation <- function(
  data,
  imputation
) {

  output <- data


  for (
    feature in names(
      imputation$medians
    )
  ) {

    missing <- is.na(
      output[[feature]]
    )

    output[[feature]][missing] <-
      imputation$medians[[feature]]
  }


  for (
    feature in names(
      imputation$modes
    )
  ) {

    missing <- is.na(
      output[[feature]]
    )

    output[[feature]][missing] <-
      as.numeric(
        imputation$modes[[feature]]
      )
  }


  output
}


# -----------------------------------------------------------------------------
# 7d. Fit design-matrix schema
# -----------------------------------------------------------------------------

fit_design_schema <- function(
  training_data,
  features
) {

  working <- training_data |>
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
      levels = 1:4
    )
  }


  # model.matrix creates dummy variables for nominal factors.
  #
  # Remove the intercept because glmnet handles its own intercept.

  matrix <- model.matrix(
    ~ .,
    data = working
  )

  matrix <- matrix[
    ,
    colnames(matrix) !=
      "(Intercept)",
    drop = FALSE
  ]


  list(
    columns =
      colnames(
        matrix
      )
  )
}


# -----------------------------------------------------------------------------
# 7e. Apply design-matrix schema
# -----------------------------------------------------------------------------

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
      levels = 1:4
    )
  }


  matrix <- model.matrix(
    ~ .,
    data = working
  )

  matrix <- matrix[
    ,
    colnames(matrix) !=
      "(Intercept)",
    drop = FALSE
  ]


  missing_columns <- setdiff(
    schema$columns,
    colnames(
      matrix
    )
  )


  if (
    length(
      missing_columns
    ) >
      0
  ) {

    zero_matrix <- matrix(
      0,
      nrow = nrow(matrix),
      ncol = length(
        missing_columns
      )
    )

    colnames(
      zero_matrix
    ) <- missing_columns

    matrix <- cbind(
      matrix,
      zero_matrix
    )
  }


  extra_columns <- setdiff(
    colnames(matrix),
    schema$columns
  )


  if (
    length(
      extra_columns
    ) >
      0
  ) {

    matrix <- matrix[
      ,
      setdiff(
        colnames(matrix),
        extra_columns
      ),
      drop = FALSE
    ]
  }


  matrix <- matrix[
    ,
    schema$columns,
    drop = FALSE
  ]


  matrix
}


# -----------------------------------------------------------------------------
# 7f. Fit scaling parameters
# -----------------------------------------------------------------------------

fit_scaler <- function(x) {

  centers <- colMeans(
    x
  )

  scales <- apply(
    x,
    2,
    sd
  )


  bad_scale <- !is.finite(
    scales
  ) |
    scales ==
      0


  if (
    any(
      bad_scale
    )
  ) {

    stop(
      paste(
        "Zero/non-finite SD after preprocessing:",
        paste(
          names(scales)[
            bad_scale
          ],
          collapse = ", "
        )
      )
    )
  }


  list(
    center =
      centers,

    scale =
      scales
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


# -----------------------------------------------------------------------------
# 7g. Fit complete preprocessing object
# -----------------------------------------------------------------------------

fit_preprocessor <- function(
  training_data
) {

  filter_object <- fit_feature_filter(
    training_data =
      training_data,

    predictor_names =
      all_predictors
  )


  features <- filter_object$
    selected_features


  imputation <- fit_imputation(
    training_data =
      training_data,

    features =
      features
  )


  imputed_training <- apply_imputation(
    data =
      training_data,

    imputation =
      imputation
  )


  schema <- fit_design_schema(
    training_data =
      imputed_training,

    features =
      features
  )


  x_train <- make_design_matrix(
    data =
      imputed_training,

    features =
      features,

    schema =
      schema
  )


  scaler <- fit_scaler(
    x_train
  )


  list(
    features =
      features,

    filter =
      filter_object,

    imputation =
      imputation,

    schema =
      schema,

    scaler =
      scaler
  )
}


# -----------------------------------------------------------------------------
# 7h. Apply complete preprocessing object
# -----------------------------------------------------------------------------

apply_preprocessor <- function(
  data,
  preprocessor
) {

  imputed <- apply_imputation(
    data =
      data,

    imputation =
      preprocessor$imputation
  )


  x <- make_design_matrix(
    data =
      imputed,

    features =
      preprocessor$features,

    schema =
      preprocessor$schema
  )


  x <- apply_scaler(
    x =
      x,

    scaler =
      preprocessor$scaler
  )


  stopifnot(
    all(
      is.finite(
        x
      )
    )
  )


  x
}


# =============================================================================
# 8. Inner-CV tuning for one alpha
# =============================================================================
#
# For each alpha:
#
#   - use the same inner folds
#   - fit preprocessing separately inside each inner-training fold
#   - fit glmnet path on inner training
#   - score candidate lambda values on inner validation
#
# Lambda is represented on a relative log scale because the exact glmnet
# lambda sequence can differ slightly across inner-training partitions.


RELATIVE_LAMBDA_GRID <- seq(
  0,
  1,
  length.out = 40
)

tune_alpha_inner <- function(
  outer_training,
  inner_fold_id,
  alpha_value
) {

  # ---------------------------------------------------------------------------
  # 8a. First determine a reference lambda range
  # ---------------------------------------------------------------------------
  #
  # The reference path is fit only on the outer-training data.
  # It is NOT evaluated on the outer-validation fold.
  #
  # This gives a stable common relative lambda grid for the inner loop.

  reference_preprocessor <- fit_preprocessor(
    outer_training
  )

  x_reference <- apply_preprocessor(
    outer_training,
    reference_preprocessor
  )

  y_reference <- outer_training[[OUTCOME]]


  reference_fit <- glmnet(
    x = x_reference,
    y = y_reference,
    family = "binomial",
    alpha = alpha_value,
    standardize = FALSE,
    intercept = TRUE,
    nlambda = 100
  )


  reference_lambda <- reference_fit$lambda

  lambda_max <- max(
    reference_lambda
  )

  lambda_min <- min(
    reference_lambda
  )


  # Work on the log-lambda scale.
  #
  # relative_position = 0 -> strongest penalty
  # relative_position = 1 -> weakest penalty

  candidate_lambda <- exp(
    log(lambda_max) +
      RELATIVE_LAMBDA_GRID *
        (
          log(lambda_min) -
            log(lambda_max)
        )
  )


  # ---------------------------------------------------------------------------
  # 8b. Inner validation predictions for each lambda
  # ---------------------------------------------------------------------------

  inner_results <- vector(
    mode = "list",
    length = N_INNER_FOLDS
  )


  for (inner_fold in seq_len(N_INNER_FOLDS)) {

    inner_train <- outer_training[
      inner_fold_id != inner_fold,
      ,
      drop = FALSE
    ]

    inner_validation <- outer_training[
      inner_fold_id == inner_fold,
      ,
      drop = FALSE
    ]


    # Fit preprocessing ONLY on inner-training data.

    preprocessor <- fit_preprocessor(
      inner_train
    )


    x_train <- apply_preprocessor(
      inner_train,
      preprocessor
    )

    x_validation <- apply_preprocessor(
      inner_validation,
      preprocessor
    )


    y_train <- inner_train[[OUTCOME]]

    y_validation <- inner_validation[[OUTCOME]]


    # Fit one glmnet path covering the candidate lambda values.

    fit <- glmnet(
      x = x_train,
      y = y_train,
      family = "binomial",
      alpha = alpha_value,
      lambda = candidate_lambda,
      standardize = FALSE,
      intercept = TRUE
    )


    probability_matrix <- predict(
      fit,
      newx = x_validation,
      type = "response",
      s = candidate_lambda
    )


    probability_matrix <- as.matrix(
      probability_matrix
    )


    stopifnot(
      nrow(probability_matrix) ==
        nrow(inner_validation),

      ncol(probability_matrix) ==
        length(candidate_lambda)
    )


    # Evaluate every candidate lambda on this held-out inner fold.

    fold_scores <- lapply(
      seq_along(candidate_lambda),
      function(lambda_index) {

        p <- as.numeric(
          probability_matrix[
            ,
            lambda_index
          ]
        )


        tibble(
          inner_fold = inner_fold,
          alpha = alpha_value,
          lambda_index = lambda_index,
          lambda = candidate_lambda[
            lambda_index
          ],
          auc = calc_auc(
            y_validation,
            p
          ),
          brier = calc_brier(
            y_validation,
            p
          ),
          log_loss = calc_log_loss(
            y_validation,
            p
          )
        )
      }
    ) |>
      bind_rows()


    inner_results[[inner_fold]] <- fold_scores
  }


  inner_results <- bind_rows(
    inner_results
  )


  # ---------------------------------------------------------------------------
  # 8c. Average inner-CV performance across folds
  # ---------------------------------------------------------------------------
  #
  # Primary tuning metric:
  #
  #   AUROC
  #
  # Brier and log-loss are retained as secondary diagnostics.

  tuning_summary <- inner_results |>
    summarise(
      mean_auc = mean(
        auc
      ),
      sd_auc = sd(
        auc
      ),
      mean_brier = mean(
        brier
      ),
      mean_log_loss = mean(
        log_loss
      ),
      .by = c(
        alpha,
        lambda_index,
        lambda
      )
    ) |>
    arrange(
      desc(mean_auc),
      mean_log_loss,
      mean_brier
    )


  best <- tuning_summary |>
    slice_head(
      n = 1
    )


  list(
    best = best,
    tuning_summary = tuning_summary,
    inner_results = inner_results
  )
}


# =============================================================================
# 9. Tune all alpha values within one outer-training set
# =============================================================================

tune_regularized_model <- function(
  outer_training,
  outer_fold
) {

  # ---------------------------------------------------------------------------
  # 9a. Create stratified inner folds
  # ---------------------------------------------------------------------------

  inner_fold_id <- make_stratified_folds(
    y = outer_training[[OUTCOME]],
    k = N_INNER_FOLDS,
    seed = INNER_SEED +
      outer_fold
  )


  # Verify inner-fold balance.

  inner_balance <- tibble(
    inner_fold = inner_fold_id,
    outcome = outer_training[[OUTCOME]]
  ) |>
    summarise(
      n = n(),
      deaths = sum(
        outcome == 1
      ),
      mortality = mean(
        outcome == 1
      ),
      .by = inner_fold
    )


  stopifnot(
    nrow(inner_balance) ==
      N_INNER_FOLDS
  )


  # ---------------------------------------------------------------------------
  # 9b. Tune each alpha independently
  # ---------------------------------------------------------------------------

  alpha_results <- lapply(
    ALPHA_GRID,
    function(alpha_value) {

      cat(
        "      alpha =",
        alpha_value,
        "\n"
      )


      result <- tune_alpha_inner(
        outer_training = outer_training,
        inner_fold_id = inner_fold_id,
        alpha_value = alpha_value
      )


      result$best |>
        mutate(
          outer_fold = outer_fold
        )
    }
  ) |>
    bind_rows()


  # ---------------------------------------------------------------------------
  # 9c. Define model families
  # ---------------------------------------------------------------------------
  #
  # Ridge:
  #   alpha = 0
  #
  # LASSO:
  #   alpha = 1
  #
  # Elastic Net:
  #   choose the best alpha among 0.25, 0.50, 0.75


  ridge_best <- alpha_results |>
    filter(
      alpha == 0
    ) |>
    slice_max(
      order_by = mean_auc,
      n = 1,
      with_ties = FALSE
    )


  lasso_best <- alpha_results |>
    filter(
      alpha == 1
    ) |>
    slice_max(
      order_by = mean_auc,
      n = 1,
      with_ties = FALSE
    )


  elastic_best <- alpha_results |>
    filter(
      alpha %in%
        c(
          0.25,
          0.50,
          0.75
        )
    ) |>
    arrange(
      desc(mean_auc),
      mean_log_loss,
      mean_brier
    ) |>
    slice_head(
      n = 1
    )


  bind_rows(
    ridge_best |>
      mutate(
        model = "Ridge"
      ),

    elastic_best |>
      mutate(
        model = "Elastic Net"
      ),

    lasso_best |>
      mutate(
        model = "LASSO"
      )
  ) |>
    select(
      outer_fold,
      model,
      alpha,
      lambda,
      lambda_index,
      mean_auc,
      sd_auc,
      mean_brier,
      mean_log_loss
    )
}


# =============================================================================
# 10. Fit one selected regularized model on one outer fold
# =============================================================================

fit_outer_model <- function(
  outer_training,
  outer_validation,
  tuning_row,
  outer_fold
) {

  model_name <- tuning_row$model[[1]]
  alpha_value <- tuning_row$alpha[[1]]
  lambda_value <- tuning_row$lambda[[1]]


  # ---------------------------------------------------------------------------
  # 10a. Fit preprocessing on ALL outer-training observations
  # ---------------------------------------------------------------------------

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


  # ---------------------------------------------------------------------------
  # 10b. Fit selected model
  # ---------------------------------------------------------------------------

  fit <- glmnet(
    x = x_train,
    y = y_train,
    family = "binomial",
    alpha = alpha_value,
    lambda = lambda_value,
    standardize = FALSE,
    intercept = TRUE
  )


  probability <- as.numeric(
    predict(
      fit,
      newx = x_validation,
      type = "response",
      s = lambda_value
    )
  )


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


  # ---------------------------------------------------------------------------
  # 10c. Extract nonzero coefficients
  # ---------------------------------------------------------------------------

  coefficient_matrix <- as.matrix(
    coef(
      fit,
      s = lambda_value
    )
  )


  coefficient_table <- tibble(
    term = rownames(
      coefficient_matrix
    ),
    coefficient = as.numeric(
      coefficient_matrix[
        ,
        1
      ]
    )
  ) |>
    filter(
      term !=
        "(Intercept)"
    ) |>
    mutate(
      selected =
        coefficient !=
          0,

      model =
        model_name,

      outer_fold =
        outer_fold,

      alpha =
        alpha_value,

      lambda =
        lambda_value
    )


  selected_count <- sum(
    coefficient_table$selected
  )


  # ---------------------------------------------------------------------------
  # 10d. Record preprocessing information
  # ---------------------------------------------------------------------------

  preprocessing_summary <- tibble(
    outer_fold = outer_fold,
    model = model_name,
    n_raw_predictors = length(
      all_predictors
    ),
    n_features_after_filter = length(
      preprocessor$features
    ),
    n_design_columns = ncol(
      x_train
    ),
    n_excluded_high_missing = length(
      preprocessor$filter$
        excluded_high_missing
    ),
    n_excluded_zero_variance = length(
      preprocessor$filter$
        excluded_zero_variance
    ),
    n_nonzero_coefficients = selected_count
  )


  predictions <- tibble(
    RecordID =
      outer_validation$RecordID,

    fold =
      outer_fold,

    outcome =
      outer_validation[[OUTCOME]],

    model =
      model_name,

    alpha =
      alpha_value,

    lambda =
      lambda_value,

    probability =
      probability
  )


  list(
    predictions =
      predictions,

    coefficients =
      coefficient_table,

    preprocessing =
      preprocessing_summary
  )
}


# =============================================================================
# 11. Outer nested-CV loop
# =============================================================================

outer_predictions <- list()
outer_tuning <- list()
outer_coefficients <- list()
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


  # ---------------------------------------------------------------------------
  # Define outer training and validation sets
  # ---------------------------------------------------------------------------

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
    "  Outer training:",
    nrow(outer_training),
    "\n"
  )

  cat(
    "  Outer validation:",
    nrow(outer_validation),
    "\n"
  )

  cat(
    "  Running inner-CV tuning...\n"
  )


  # ---------------------------------------------------------------------------
  # Tune Ridge / Elastic Net / LASSO using only outer-training data
  # ---------------------------------------------------------------------------

  tuning <- tune_regularized_model(
    outer_training = outer_training,
    outer_fold = outer_fold
  )


  outer_tuning[[outer_fold]] <- tuning


  cat(
    "\n  Selected configurations:\n"
  )

  print(
    tuning |>
      select(
        model,
        alpha,
        lambda,
        mean_auc
      ),
    n = Inf
  )


  # ---------------------------------------------------------------------------
  # Fit the three selected model configurations
  # ---------------------------------------------------------------------------

  model_results <- lapply(
    seq_len(nrow(tuning)),
    function(i) {

      tuning_row <- tuning[
        i,
        ,
        drop = FALSE
      ]


      cat(
        "  Fitting",
        tuning_row$model[[1]],
        "on outer-training data...\n"
      )


      fit_outer_model(
        outer_training = outer_training,
        outer_validation = outer_validation,
        tuning_row = tuning_row,
        outer_fold = outer_fold
      )
    }
  )


  # ---------------------------------------------------------------------------
  # Save results from this outer fold
  # ---------------------------------------------------------------------------

  outer_predictions[[outer_fold]] <- bind_rows(
    lapply(
      model_results,
      function(x) {
        x$predictions
      }
    )
  )


  outer_coefficients[[outer_fold]] <- bind_rows(
    lapply(
      model_results,
      function(x) {
        x$coefficients
      }
    )
  )


  outer_preprocessing[[outer_fold]] <- bind_rows(
    lapply(
      model_results,
      function(x) {
        x$preprocessing
      }
    )
  )
}


# =============================================================================
# 12. Combine nested-CV results
# =============================================================================

all_predictions <- bind_rows(
  outer_predictions
)


all_tuning <- bind_rows(
  outer_tuning
)


all_coefficients <- bind_rows(
  outer_coefficients
)


all_preprocessing <- bind_rows(
  outer_preprocessing
)


expected_models <- c(
  "Ridge",
  "Elastic Net",
  "LASSO"
)


# =============================================================================
# 13. Validate nested out-of-fold predictions
# =============================================================================

for (
  model_name in expected_models
) {

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
    ),

    all(
      is.finite(
        model_predictions$probability
      )
    )
  )
}


# =============================================================================
# 14. Overall nested-CV performance
# =============================================================================

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
  "\n============================================================\n"
)

cat(
  "Overall nested-CV performance\n"
)

cat(
  "============================================================\n"
)

print(
  overall_performance,
  n = Inf
)


# =============================================================================
# 15. Fold-specific nested-CV performance
# =============================================================================

performance_by_fold <- lapply(
  expected_models,
  function(
    model_name
  ) {

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
# 16. Selected-feature frequency
# =============================================================================
#
# Ridge normally retains all coefficients, whereas LASSO and Elastic Net can
# produce exact zeros.
#
# This table is useful for examining selection stability across outer folds.


feature_frequency <- all_coefficients |>
  filter(
    model %in%
      c(
        "Elastic Net",
        "LASSO"
      )
  ) |>
  summarise(
    folds_available =
      n_distinct(
        outer_fold
      ),

    folds_selected =
      sum(
        selected
      ),

    selection_frequency =
      mean(
        selected
      ),

    mean_coefficient =
      mean(
        coefficient
      ),

    mean_abs_coefficient =
      mean(
        abs(
          coefficient
        )
      ),

    .by =
      c(
        model,
        term
      )
  ) |>
  arrange(
    model,
    desc(
      selection_frequency
    ),
    desc(
      mean_abs_coefficient
    )
  )


# =============================================================================
# 17. Save nested-CV outputs
# =============================================================================

write_csv(
  all_predictions,
  here(
    "output",
    "cv_predictions_regularized.csv"
  )
)

write_csv(
  overall_performance,
  here(
    "output",
    "cv_performance_regularized.csv"
  )
)

write_csv(
  performance_by_fold,
  here(
    "output",
    "cv_performance_regularized_by_fold.csv"
  )
)

write_csv(
  performance_fold_summary,
  here(
    "output",
    "cv_performance_regularized_fold_summary.csv"
  )
)

write_csv(
  all_tuning,
  here(
    "output",
    "regularized_tuning_by_fold.csv"
  )
)

write_csv(
  all_coefficients,
  here(
    "output",
    "regularized_selected_features.csv"
  )
)

write_csv(
  feature_frequency,
  here(
    "output",
    "regularized_feature_frequency.csv"
  )
)

write_csv(
  all_preprocessing,
  here(
    "output",
    "regularized_preprocessing_by_fold.csv"
  )
)


# =============================================================================
# 18. Compare against the existing logistic / clinical benchmarks
# =============================================================================
#
# This comparison uses only Set A held-out predictions.
#
# The regularized models use nested outer-CV predictions.
# The earlier models use the fixed 5-fold held-out predictions from 08.
#
# Set B remains untouched.


baseline_performance_path <- here(
  "output",
  "cv_performance_benchmarks_logistic.csv"
)


if (
  file.exists(
    baseline_performance_path
  )
) {

  baseline_performance <- read_csv(
    baseline_performance_path,
    show_col_types = FALSE
  )


  comparison_performance <- bind_rows(

    baseline_performance |>
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
    comparison_performance,
    here(
      "output",
      "cv_model_comparison_through_regularized.csv"
    )
  )


  cat(
    "\nModel comparison through regularized models:\n"
  )

  print(
    comparison_performance,
    n = Inf
  )
}


# =============================================================================
# 19. Inspect hyperparameter stability across outer folds
# =============================================================================

tuning_stability <- all_tuning |>
  summarise(

    mean_alpha =
      mean(
        alpha
      ),

    sd_alpha =
      sd(
        alpha
      ),

    median_lambda =
      median(
        lambda
      ),

    min_lambda =
      min(
        lambda
      ),

    max_lambda =
      max(
        lambda
      ),

    mean_inner_auc =
      mean(
        mean_auc
      ),

    sd_inner_auc =
      sd(
        mean_auc
      ),

    .by =
      model
  ) |>
  arrange(
    model
  )


write_csv(
  tuning_stability,
  here(
    "output",
    "regularized_tuning_stability.csv"
  )
)


cat(
  "\nHyperparameter stability across outer folds:\n"
)

print(
  tuning_stability,
  n = Inf
)


# =============================================================================
# 20. Inspect number of selected coefficients
# =============================================================================

selected_count_summary <- all_coefficients |>
  summarise(

    n_coefficients =
      n(),

    n_selected =
      sum(
        selected
      ),

    proportion_selected =
      mean(
        selected
      ),

    .by =
      c(
        model,
        outer_fold
      )
  ) |>
  arrange(
    model,
    outer_fold
  )


write_csv(
  selected_count_summary,
  here(
    "output",
    "regularized_selected_count_by_fold.csv"
  )
)


cat(
  "\nSelected coefficients by outer fold:\n"
)

print(
  selected_count_summary,
  n = Inf
)


# =============================================================================
# 21. Final full-Set-A hyperparameter selection
# =============================================================================
#
# After nested CV has provided an honest development-performance estimate,
# we may use ALL of Set A to select the configuration that will eventually
# be refit before the single-look Set B evaluation.
#
# This does NOT change the nested-CV performance estimates above.
#
# We again use internal stratified CV within all Set A.
#
# Set B remains untouched.


cat(
  "\n============================================================\n"
)

cat(
  "Selecting final regularized configurations using all Set A\n"
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


final_alpha_results <- lapply(
  ALPHA_GRID,
  function(
    alpha_value
  ) {

    cat(
      "  Final Set-A tuning: alpha =",
      alpha_value,
      "\n"
    )


    result <- tune_alpha_inner(
      outer_training =
        dat,

      inner_fold_id =
        final_inner_fold_id,

      alpha_value =
        alpha_value
    )


    result$best
  }
) |>
  bind_rows()


final_ridge <- final_alpha_results |>
  filter(
    alpha ==
      0
  ) |>
  arrange(
    desc(
      mean_auc
    ),
    mean_log_loss,
    mean_brier
  ) |>
  slice_head(
    n =
      1
  ) |>
  mutate(
    model =
      "Ridge"
  )


final_lasso <- final_alpha_results |>
  filter(
    alpha ==
      1
  ) |>
  arrange(
    desc(
      mean_auc
    ),
    mean_log_loss,
    mean_brier
  ) |>
  slice_head(
    n =
      1
  ) |>
  mutate(
    model =
      "LASSO"
  )


final_elastic <- final_alpha_results |>
  filter(
    alpha %in%
      c(
        0.25,
        0.50,
        0.75
      )
  ) |>
  arrange(
    desc(
      mean_auc
    ),
    mean_log_loss,
    mean_brier
  ) |>
  slice_head(
    n =
      1
  ) |>
  mutate(
    model =
      "Elastic Net"
  )


final_configuration <- bind_rows(
  final_ridge,
  final_elastic,
  final_lasso
) |>
  select(
    model,
    alpha,
    lambda,
    lambda_index,
    mean_auc,
    sd_auc,
    mean_brier,
    mean_log_loss
  )


write_csv(
  final_configuration,
  here(
    "output",
    "regularized_final_configuration_set_a.csv"
  )
)


cat(
  "\nFinal Set-A regularized configurations:\n"
)

print(
  final_configuration,
  n = Inf
)


# =============================================================================
# 22. Fit final preprocessing object on all Set A
# =============================================================================
#
# This is allowed only after nested-CV evaluation is complete.
#
# The resulting preprocessing parameters are learned exclusively from Set A.
# They can later be applied unchanged to Set B.


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
  "  Features after training-data filter:",
  length(
    final_preprocessor$features
  ),
  "\n"
)

cat(
  "  Final design-matrix columns:",
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
# 23. Fit final Ridge / Elastic Net / LASSO models on all Set A
# =============================================================================
#
# These models are NOT evaluated on Set A to estimate generalization.
#
# Their purpose is to freeze the final development models that could later
# be applied exactly once to Set B.


final_model_coefficients <- list()


for (i in seq_len(nrow(final_configuration))) {

  configuration <- final_configuration[
    i,
    ,
    drop = FALSE
  ]


  model_name <- configuration$model[[1]]
  alpha_value <- configuration$alpha[[1]]
  lambda_value <- configuration$lambda[[1]]


  cat(
    "  Fitting final",
    model_name,
    "model on all Set A...\n"
  )


  # ---------------------------------------------------------------------------
  # Fit final regularized model
  # ---------------------------------------------------------------------------

  fit <- glmnet(
    x = x_final,
    y = y_final,
    family = "binomial",
    alpha = alpha_value,
    lambda = lambda_value,
    standardize = FALSE,
    intercept = TRUE
  )


  # ---------------------------------------------------------------------------
  # Extract coefficients
  # ---------------------------------------------------------------------------

  coefficient_matrix <- as.matrix(
    coef(
      fit,
      s = lambda_value
    )
  )


  coefficient_table <- tibble(
    term = rownames(
      coefficient_matrix
    ),

    coefficient = as.numeric(
      coefficient_matrix[, 1]
    ),

    selected =
      as.numeric(
        coefficient_matrix[, 1]
      ) != 0,

    model =
      model_name,

    alpha =
      alpha_value,

    lambda =
      lambda_value
  )


  # ---------------------------------------------------------------------------
  # Store model coefficients
  # ---------------------------------------------------------------------------

  final_model_coefficients[[model_name]] <-
    coefficient_table
}


# Combine the coefficient tables from the three final models.

final_model_coefficients <- bind_rows(
  final_model_coefficients
)


write_csv(
  final_model_coefficients,
  here(
    "output",
    "regularized_final_coefficients_set_a.csv"
  )
)


# =============================================================================
# 24. Save final preprocessing specification
# =============================================================================
#
# We save the human-readable parts required to audit the final preprocessing.
#
# The actual future Set-B script will reconstruct the preprocessing from Set A
# rather than learn anything from Set B.


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
    "regularized_final_feature_status.csv"
  )
)


final_numeric_imputation <- tibble(

  feature =
    names(
      final_preprocessor$imputation$
        medians
    ),

  median_set_a =
    as.numeric(
      final_preprocessor$imputation$
        medians
    )
)


write_csv(
  final_numeric_imputation,
  here(
    "output",
    "regularized_final_imputation.csv"
  )
)


if (
  length(
    final_preprocessor$imputation$
      modes
  ) >
    0
) {

  final_categorical_imputation <- tibble(

    feature =
      names(
        final_preprocessor$imputation$
          modes
      ),

    mode_set_a =
      as.numeric(
        unlist(
          final_preprocessor$imputation$
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
    "regularized_final_categorical_imputation.csv"
  )
)


final_scaling <- tibble(

  design_column =
    names(
      final_preprocessor$scaler$
        center
    ),

  center_set_a =
    as.numeric(
      final_preprocessor$scaler$
        center
    ),

  scale_set_a =
    as.numeric(
      final_preprocessor$scaler$
        scale
    )
)


write_csv(
  final_scaling,
  here(
    "output",
    "regularized_final_scaling.csv"
  )
)


# =============================================================================
# 25. Validate final results
# =============================================================================

stopifnot(

  nrow(
    all_predictions
  ) ==
    N_STAYS *
      length(
        expected_models
      ),

  nrow(
    all_tuning
  ) ==
    N_OUTER_FOLDS *
      length(
        expected_models
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
  ),

  nrow(
    final_configuration
  ) ==
    3,

  all(
    expected_models %in%
      final_configuration$model
  )
)


# Every patient must receive exactly one outer-held-out prediction
# from every regularized model family.

prediction_validation <- all_predictions |>
  count(
    RecordID,
    model,
    name =
      "n_predictions"
  )


stopifnot(
  nrow(
    prediction_validation
  ) ==
    N_STAYS *
      length(
        expected_models
      ),

  all(
    prediction_validation$
      n_predictions ==
      1
  )
)


# =============================================================================
# 26. Final report
# =============================================================================

cat(
  "\n============================================================\n"
)

cat(
  "Checkpoint 2 regularized modeling complete.\n"
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
  "\nSelected hyperparameters by outer fold:\n"
)

print(
  all_tuning |>
    select(
      outer_fold,
      model,
      alpha,
      lambda,
      mean_auc
    ) |>
    arrange(
      model,
      outer_fold
    ),
  n = Inf
)


cat(
  "\nLeakage safeguards:\n"
)

cat(
  paste(
    "  1. Outer folds are the fixed folds created by",
    "07_define_resampling.R.\n"
  )
)

cat(
  paste(
    "  2. Hyperparameter tuning occurs only inside",
    "each outer-training partition.\n"
  )
)

cat(
  paste(
    "  3. Feature filtering, imputation, dummy encoding,",
    "and scaling are fit only on the relevant training data.\n"
  )
)

cat(
  paste(
    "  4. Outer-validation observations do not contribute",
    "to inner tuning or preprocessing.\n"
  )
)

cat(
  "  5. Set B was not accessed.\n"
)


cat(
  "\nGenerated outputs:\n"
)

cat(
  "  output/cv_predictions_regularized.csv\n"
)

cat(
  "  output/cv_performance_regularized.csv\n"
)

cat(
  "  output/cv_performance_regularized_by_fold.csv\n"
)

cat(
  "  output/cv_performance_regularized_fold_summary.csv\n"
)

cat(
  "  output/regularized_tuning_by_fold.csv\n"
)

cat(
  "  output/regularized_tuning_stability.csv\n"
)

cat(
  "  output/regularized_selected_features.csv\n"
)

cat(
  "  output/regularized_feature_frequency.csv\n"
)

cat(
  "  output/regularized_selected_count_by_fold.csv\n"
)

cat(
  "  output/regularized_preprocessing_by_fold.csv\n"
)

cat(
  "  output/regularized_final_configuration_set_a.csv\n"
)

cat(
  "  output/regularized_final_coefficients_set_a.csv\n"
)

cat(
  "  output/regularized_final_feature_status.csv\n"
)

cat(
  "  output/regularized_final_imputation.csv\n"
)

cat(
  "  output/regularized_final_categorical_imputation.csv\n"
)

cat(
  "  output/regularized_final_scaling.csv\n"
)

if (
  exists(
    "comparison_performance"
  )
) {

  cat(
    "  output/cv_model_comparison_through_regularized.csv\n"
  )
}
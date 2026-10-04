# BIOSTAT 707 Checkpoint 2
# 05_build_features.R
#
# Purpose:
#   Construct deterministic, clinically motivated patient-level features
#   from the first 48 hours of PhysioNet Challenge 2012 Set A and Set B.
#
# Important:
#   This script DOES:
#     - apply fixed data-cleaning rules
#     - enforce the 0-48 hour observation window
#     - construct summary statistics
#     - construct trend features
#     - construct explicit missingness indicators
#     - construct selected clinically motivated special features
#
#   This script DOES NOT:
#     - impute missing values
#     - standardize predictors
#     - perform feature selection
#     - use outcomes to construct predictors
#     - tune models
#
#   Those learned preprocessing steps must occur inside cross-validation.
#
# Outputs:
#   output/set-a_features.csv
#   output/set-b_features.csv
#   output/feature_dictionary.csv


# =============================================================================
# 1. Packages and constants
# =============================================================================

library(readr)
library(dplyr)
library(tidyr)
library(here)

N_A <- 4000
N_B <- 4000

OBS_WINDOW_HOURS <- 48


# =============================================================================
# 2. Paths
# =============================================================================

set_a_dir <- here(
  "data",
  "set-a"
)

set_b_dir <- here(
  "data",
  "set-b"
)

outcomes_a_path <- here(
  "data",
  "Outcomes-a.txt"
)

output_dir <- here(
  "output"
)

dir.create(
  output_dir,
  showWarnings = FALSE,
  recursive = TRUE
)

stopifnot(
  dir.exists(set_a_dir),
  dir.exists(set_b_dir),
  file.exists(outcomes_a_path)
)


# =============================================================================
# 3. Identify raw files
# =============================================================================

set_a_files <- list.files(
  set_a_dir,
  pattern = "\\.txt$",
  full.names = TRUE
)

set_b_files <- list.files(
  set_b_dir,
  pattern = "\\.txt$",
  full.names = TRUE
)

cat(
  "Set A files:",
  length(set_a_files),
  "\n"
)

cat(
  "Set B files:",
  length(set_b_files),
  "\n"
)

stopifnot(
  length(set_a_files) == N_A,
  length(set_b_files) == N_B
)


# =============================================================================
# 4. Parse Challenge 2012 time
# =============================================================================

parse_time_hours <- function(x) {

  x <- as.character(x)

  parts <- strsplit(
    x,
    ":",
    fixed = TRUE
  )

  hours <- as.numeric(
    vapply(
      parts,
      `[`,
      character(1),
      1
    )
  )

  minutes <- as.numeric(
    vapply(
      parts,
      `[`,
      character(1),
      2
    )
  )

  hours + minutes / 60
}


# =============================================================================
# 5. Fixed cleaning rules
# =============================================================================
#
# These are deterministic rules defined before modeling.
# They do not use outcomes or information from other patients.
#
# This preserves the cleaning philosophy used in Checkpoint 1.


clean_value <- function(parameter, value) {

  cleaned <- value

  # Explicit missing-value code.
  cleaned[
    value == -1 &
      parameter != "RecordID"
  ] <- NA_real_

  # Invalid pH scale.
  cleaned[
    parameter == "pH" &
      !is.na(value) &
      (value < 0 | value > 14)
  ] <- NA_real_

  # Clearly implausible body temperature.
  cleaned[
    parameter == "Temp" &
      !is.na(value) &
      (value < 30 | value > 45)
  ] <- NA_real_

  # Height is predominantly represented in centimeters.
  cleaned[
    parameter == "Height" &
      !is.na(value) &
      (value < 100 | value > 250)
  ] <- NA_real_

  # Zero HR is treated as a monitor/data artifact.
  cleaned[
    parameter == "HR" &
      !is.na(value) &
      value == 0
  ] <- NA_real_

  # Zero or negative blood pressure is not treated as a
  # physiological measurement.
  bp_parameters <- c(
    "SysABP",
    "DiasABP",
    "MAP",
    "NISysABP",
    "NIDiasABP",
    "NIMAP"
  )

  cleaned[
    parameter %in% bp_parameters &
      !is.na(value) &
      value <= 0
  ] <- NA_real_

  cleaned
}


# =============================================================================
# 6. Read and validate one ICU record
# =============================================================================

read_one_record <- function(file) {

  dat <- read_csv(
    file,
    show_col_types = FALSE
  )

  required <- c(
    "Time",
    "Parameter",
    "Value"
  )

  stopifnot(
    all(required %in% names(dat))
  )

  record_id <- dat |>
    filter(
      Parameter == "RecordID"
    ) |>
    pull(Value)

  stopifnot(
    length(record_id) == 1,
    !is.na(record_id)
  )

  dat |>
    mutate(
      RecordID =
        as.integer(record_id),

      hours =
        parse_time_hours(Time),

      Value_clean =
        clean_value(
          Parameter,
          Value
        )
    ) |>
    select(
      RecordID,
      Time,
      hours,
      Parameter,
      Value,
      Value_clean
    )
}


# =============================================================================
# 7. Validate Set A / Set B parameter vocabularies
# =============================================================================

get_parameter_vocabulary <- function(files) {

  unique(
    unlist(
      lapply(
        files,
        function(file) {

          dat <- read_csv(
            file,
            show_col_types = FALSE
          )

          dat$Parameter
        }
      )
    )
  ) |>
    sort()
}


cat(
  "\nChecking Set A / Set B parameter vocabularies...\n"
)

params_a <- get_parameter_vocabulary(
  set_a_files
)

params_b <- get_parameter_vocabulary(
  set_b_files
)

stopifnot(
  identical(
    params_a,
    params_b
  )
)

cat(
  "Shared parameter vocabulary:",
  length(params_a),
  "parameters\n"
)


# =============================================================================
# 8. Feature groups
# =============================================================================

# Static variables measured at admission.

static_parameters <- c(
  "Age",
  "Gender",
  "Height",
  "ICUType"
)


# Repeated vital-sign / physiological measurements.

vital_parameters <- c(
  "HR",
  "Temp",
  "RespRate",
  "SysABP",
  "DiasABP",
  "MAP",
  "NISysABP",
  "NIDiasABP",
  "NIMAP",
  "SaO2"
)


# Laboratory measurements.

lab_parameters <- c(
  "Albumin",
  "ALP",
  "ALT",
  "AST",
  "Bilirubin",
  "BUN",
  "Cholesterol",
  "Creatinine",
  "Glucose",
  "HCO3",
  "HCT",
  "K",
  "Lactate",
  "Mg",
  "Na",
  "PaCO2",
  "PaO2",
  "pH",
  "Platelets",
  "TroponinI",
  "TroponinT",
  "WBC"
)


# GCS is treated separately because it is an ordinal clinical score.

gcs_parameters <- c(
  "GCS"
)


# FiO2 is handled separately.

fio2_parameters <- c(
  "FiO2"
)


# Special variables handled with custom summaries:
#
# Weight   -> first valid weight
# MechVent -> ever mechanically ventilated
# Urine    -> total urine output + measurement process

special_parameters <- c(
  "Weight",
  "MechVent",
  "Urine"
)


expected_model_parameters <- c(
  static_parameters,
  vital_parameters,
  lab_parameters,
  gcs_parameters,
  fio2_parameters,
  special_parameters
)


# RecordID itself is not a predictor.

observed_non_id_parameters <- setdiff(
  params_a,
  "RecordID"
)


# Confirm that every raw predictor has been assigned a role.

unassigned_parameters <- setdiff(
  observed_non_id_parameters,
  expected_model_parameters
)

unexpected_parameters <- setdiff(
  expected_model_parameters,
  observed_non_id_parameters
)

if (
  length(unassigned_parameters) > 0
) {

  stop(
    paste(
      "Unassigned raw parameters:",
      paste(
        unassigned_parameters,
        collapse = ", "
      )
    )
  )
}


if (
  length(unexpected_parameters) > 0
) {

  stop(
    paste(
      "Expected parameters absent from raw data:",
      paste(
        unexpected_parameters,
        collapse = ", "
      )
    )
  )
}


# =============================================================================
# 9. Safe summary helpers
# =============================================================================

safe_first <- function(x, time) {

  ok <- !is.na(x) &
    !is.na(time)

  if (!any(ok)) {
    return(NA_real_)
  }

  x[ok][
    order(
      time[ok]
    )[1]
  ]
}


safe_last <- function(x, time) {

  ok <- !is.na(x) &
    !is.na(time)

  if (!any(ok)) {
    return(NA_real_)
  }

  ordered <- order(
    time[ok]
  )

  x[ok][
    ordered[
      length(ordered)
    ]
  ]
}


safe_mean <- function(x) {

  if (
    all(
      is.na(x)
    )
  ) {
    return(NA_real_)
  }

  mean(
    x,
    na.rm = TRUE
  )
}


safe_sd <- function(x) {

  x <- x[
    !is.na(x)
  ]

  if (
    length(x) < 2
  ) {
    return(NA_real_)
  }

  sd(x)
}


safe_min <- function(x) {

  if (
    all(
      is.na(x)
    )
  ) {
    return(NA_real_)
  }

  min(
    x,
    na.rm = TRUE
  )
}


safe_max <- function(x) {

  if (
    all(
      is.na(x)
    )
  ) {
    return(NA_real_)
  }

  max(
    x,
    na.rm = TRUE
  )
}


safe_sum <- function(x) {

  if (
    all(
      is.na(x)
    )
  ) {
    return(NA_real_)
  }

  sum(
    x,
    na.rm = TRUE
  )
}


safe_delta <- function(x, time) {

  first <- safe_first(
    x,
    time
  )

  last <- safe_last(
    x,
    time
  )

  if (
    is.na(first) ||
      is.na(last)
  ) {
    return(NA_real_)
  }

  last - first
}


safe_slope <- function(x, time) {

  ok <- !is.na(x) &
    !is.na(time)

  x <- x[ok]
  time <- time[ok]

  # At least two valid observations at distinct times are required.
  if (
    length(x) < 2 ||
      length(
        unique(time)
      ) < 2
  ) {
    return(NA_real_)
  }

  fit <- lm(
    x ~ time
  )

  unname(
    coef(fit)[
      "time"
    ]
  )
}


# =============================================================================
# 10. Generic feature constructors
# =============================================================================

make_vital_features <- function(dat, parameter) {

  d <- dat |>
    filter(
      Parameter == parameter,
      !is.na(Value_clean)
    )

  n <- nrow(d)

  values <- if (n > 0) {
    d$Value_clean
  } else {
    numeric(0)
  }

  times <- if (n > 0) {
    d$hours
  } else {
    numeric(0)
  }

  tibble(
    !!paste0(
      parameter,
      "_first"
    ) :=
      safe_first(
        values,
        times
      ),

    !!paste0(
      parameter,
      "_last"
    ) :=
      safe_last(
        values,
        times
      ),

    !!paste0(
      parameter,
      "_min"
    ) :=
      safe_min(
        values
      ),

    !!paste0(
      parameter,
      "_max"
    ) :=
      safe_max(
        values
      ),

    !!paste0(
      parameter,
      "_mean"
    ) :=
      safe_mean(
        values
      ),

    !!paste0(
      parameter,
      "_sd"
    ) :=
      safe_sd(
        values
      ),

    !!paste0(
      parameter,
      "_slope"
    ) :=
      safe_slope(
        values,
        times
      ),

    !!paste0(
      parameter,
      "_count"
    ) :=
      n,

    !!paste0(
      parameter,
      "_missing"
    ) :=
      as.integer(
        n == 0
      )
  )
}


make_lab_features <- function(dat, parameter) {

  d <- dat |>
    filter(
      Parameter == parameter,
      !is.na(Value_clean)
    )

  n <- nrow(d)

  values <- if (n > 0) {
    d$Value_clean
  } else {
    numeric(0)
  }

  times <- if (n > 0) {
    d$hours
  } else {
    numeric(0)
  }

  tibble(
    !!paste0(
      parameter,
      "_first"
    ) :=
      safe_first(
        values,
        times
      ),

    !!paste0(
      parameter,
      "_last"
    ) :=
      safe_last(
        values,
        times
      ),

    !!paste0(
      parameter,
      "_min"
    ) :=
      safe_min(
        values
      ),

    !!paste0(
      parameter,
      "_max"
    ) :=
      safe_max(
        values
      ),

    !!paste0(
      parameter,
      "_mean"
    ) :=
      safe_mean(
        values
      ),

    !!paste0(
      parameter,
      "_delta"
    ) :=
      safe_delta(
        values,
        times
      ),

    !!paste0(
      parameter,
      "_slope"
    ) :=
      safe_slope(
        values,
        times
      ),

    !!paste0(
      parameter,
      "_count"
    ) :=
      n,

    !!paste0(
      parameter,
      "_missing"
    ) :=
      as.integer(
        n == 0
      )
  )
}


# =============================================================================
# 11. GCS features
# =============================================================================

make_gcs_features <- function(dat) {

  d <- dat |>
    filter(
      Parameter == "GCS",
      !is.na(Value_clean)
    )

  n <- nrow(d)

  values <- if (n > 0) {
    d$Value_clean
  } else {
    numeric(0)
  }

  times <- if (n > 0) {
    d$hours
  } else {
    numeric(0)
  }

  tibble(
    GCS_first =
      safe_first(
        values,
        times
      ),

    GCS_last =
      safe_last(
        values,
        times
      ),

    GCS_min =
      safe_min(
        values
      ),

    GCS_mean =
      safe_mean(
        values
      ),

    GCS_delta =
      safe_delta(
        values,
        times
      ),

    GCS_count =
      n,

    GCS_missing =
      as.integer(
        n == 0
      )
  )
}


# =============================================================================
# 12. FiO2 features
# =============================================================================

make_fio2_features <- function(dat) {

  d <- dat |>
    filter(
      Parameter == "FiO2",
      !is.na(Value_clean)
    )

  n <- nrow(d)

  values <- if (n > 0) {
    d$Value_clean
  } else {
    numeric(0)
  }

  times <- if (n > 0) {
    d$hours
  } else {
    numeric(0)
  }

  tibble(
    FiO2_first =
      safe_first(
        values,
        times
      ),

    FiO2_last =
      safe_last(
        values,
        times
      ),

    FiO2_max =
      safe_max(
        values
      ),

    FiO2_mean =
      safe_mean(
        values
      ),

    FiO2_count =
      n,

    FiO2_missing =
      as.integer(
        n == 0
      )
  )
}


# =============================================================================
# 13. Static features
# =============================================================================

make_static_features <- function(dat) {

  get_static <- function(parameter) {

    values <- dat |>
      filter(
        Parameter == parameter,
        !is.na(Value_clean)
      ) |>
      arrange(hours) |>
      pull(Value_clean)

    if (
      length(values) == 0
    ) {
      return(NA_real_)
    }

    values[1]
  }

  tibble(
    Age =
      get_static("Age"),

    Gender =
      get_static("Gender"),

    Height =
      get_static("Height"),

    ICUType =
      get_static("ICUType")
  )
}


# =============================================================================
# 14. Special features
# =============================================================================

make_special_features <- function(dat) {

  # ---------------------------------------------------------------------------
  # Weight
  # ---------------------------------------------------------------------------
  # Use the first valid weight measurement during the observation window.

  weight <- dat |>
    filter(
      Parameter == "Weight",
      !is.na(Value_clean)
    )

  weight_first <- safe_first(
    weight$Value_clean,
    weight$hours
  )


  # ---------------------------------------------------------------------------
  # Mechanical ventilation
  # ---------------------------------------------------------------------------
  # MechVent is treated as a binary status/intervention variable.
  #
  # MechVent_ever = 1:
  #   at least one observed MechVent value equals 1
  #
  # MechVent_ever = 0:
  #   no observed value equals 1
  #
  # MechVent_missing separately records whether MechVent was never observed.

  mech <- dat |>
    filter(
      Parameter == "MechVent",
      !is.na(Value_clean)
    )

  mech_count <- nrow(mech)

  ever_mechvent <- if (
    mech_count == 0
  ) {
    0L
  } else {
    as.integer(
      any(
        mech$Value_clean == 1
      )
    )
  }


  # ---------------------------------------------------------------------------
  # Urine
  # ---------------------------------------------------------------------------
  # Urine is treated as an accumulated quantity rather than an instantaneous
  # physiological measurement.
  #
  # If there are no valid urine observations, Urine_total remains NA.
  # Missingness is represented explicitly by Urine_missing.

  urine <- dat |>
    filter(
      Parameter == "Urine",
      !is.na(Value_clean)
    )

  urine_count <- nrow(urine)

  urine_total <- if (
    urine_count == 0
  ) {
    NA_real_
  } else {
    sum(
      urine$Value_clean,
      na.rm = TRUE
    )
  }


  # ---------------------------------------------------------------------------
  # Return special features
  # ---------------------------------------------------------------------------

  tibble(
    Weight_first =
      weight_first,

    Weight_missing =
      as.integer(
        is.na(weight_first)
      ),

    MechVent_ever =
      ever_mechvent,

    MechVent_count =
      mech_count,

    MechVent_missing =
      as.integer(
        mech_count == 0
      ),

    Urine_total =
      urine_total,

    Urine_count =
      urine_count,

    Urine_missing =
      as.integer(
        urine_count == 0
      )
  )
}


# =============================================================================
# 15. Construct features for one ICU stay
# =============================================================================

build_one_patient_features <- function(file) {

  dat <- read_one_record(file)

  record_id <- unique(
    dat$RecordID
  )

  stopifnot(
    length(record_id) == 1
  )


  # ---------------------------------------------------------------------------
  # Enforce the observation window
  # ---------------------------------------------------------------------------

  if (
    any(
      is.na(dat$hours)
    )
  ) {
    stop(
      paste(
        "Unparseable time in RecordID",
        record_id
      )
    )
  }

  if (
    any(
      dat$hours < 0 |
        dat$hours > OBS_WINDOW_HOURS
    )
  ) {
    stop(
      paste(
        "Observation outside 0-48 hour window in RecordID",
        record_id
      )
    )
  }


  # ---------------------------------------------------------------------------
  # Static features
  # ---------------------------------------------------------------------------

  static_features <-
    make_static_features(
      dat
    )


  # ---------------------------------------------------------------------------
  # Vital-sign features
  # ---------------------------------------------------------------------------

  vital_features <- lapply(
    vital_parameters,
    function(parameter) {

      make_vital_features(
        dat,
        parameter
      )
    }
  ) |>
    bind_cols()


  # ---------------------------------------------------------------------------
  # Laboratory features
  # ---------------------------------------------------------------------------

  lab_features <- lapply(
    lab_parameters,
    function(parameter) {

      make_lab_features(
        dat,
        parameter
      )
    }
  ) |>
    bind_cols()


  # ---------------------------------------------------------------------------
  # Other clinically motivated feature groups
  # ---------------------------------------------------------------------------

  gcs_features <-
    make_gcs_features(
      dat
    )

  fio2_features <-
    make_fio2_features(
      dat
    )

  special_features <-
    make_special_features(
      dat
    )


  # ---------------------------------------------------------------------------
  # Combine into one patient-level row
  # ---------------------------------------------------------------------------

  bind_cols(
    tibble(
      RecordID =
        as.integer(
          record_id
        )
    ),
    static_features,
    vital_features,
    lab_features,
    gcs_features,
    fio2_features,
    special_features
  )
}


# =============================================================================
# 16. Construct one complete feature table
# =============================================================================

build_feature_table <- function(
  files,
  expected_n,
  dataset_name
) {

  cat(
    "\nBuilding features for",
    dataset_name,
    "...\n"
  )

  features <- lapply(
    seq_along(files),
    function(i) {

      if (
        i %% 500 == 0
      ) {
        cat(
          dataset_name,
          ": processed",
          i,
          "of",
          length(files),
          "records\n"
        )
      }

      build_one_patient_features(
        files[[i]]
      )
    }
  ) |>
    bind_rows()


  # ---------------------------------------------------------------------------
  # Dataset-level validation
  # ---------------------------------------------------------------------------

  stopifnot(
    nrow(features) ==
      expected_n,

    n_distinct(
      features$RecordID
    ) ==
      expected_n,

    !anyDuplicated(
      features$RecordID
    )
  )


  # RecordID is an identifier, not a model feature.
  stopifnot(
    !any(
      is.na(
        features$RecordID
      )
    )
  )


  cat(
    dataset_name,
    "feature table:",
    nrow(features),
    "rows x",
    ncol(features),
    "columns\n"
  )

  features
}


# =============================================================================
# 17. Build Set A and Set B using exactly the same feature code
# =============================================================================

features_a <- build_feature_table(
  files =
    set_a_files,

  expected_n =
    N_A,

  dataset_name =
    "Set A"
)


features_b <- build_feature_table(
  files =
    set_b_files,

  expected_n =
    N_B,

  dataset_name =
    "Set B"
)


# =============================================================================
# 18. Verify identical predictor schemas
# =============================================================================
#
# This is one of the most important safeguards in this script.
#
# The same deterministic feature map phi(.) must be applied to both:
#
#   phi(Set A)
#   phi(Set B)
#
# Set B must never receive special feature engineering based on its outcomes.


stopifnot(
  identical(
    names(features_a),
    names(features_b)
  )
)

cat(
  "\nSet A / Set B feature schemas are identical.\n"
)

cat(
  "Predictor columns including RecordID:",
  ncol(features_a),
  "\n"
)


# =============================================================================
# 19. Load Set A outcomes
# =============================================================================
#
# Outcomes are joined only AFTER predictor construction.
#
# This guarantees that feature engineering itself does not use the outcome.


outcomes_a <- read_csv(
  outcomes_a_path,
  show_col_types = FALSE
)


required_outcome_columns <- c(
  "RecordID",
  "SAPS-I",
  "SOFA",
  "Length_of_stay",
  "Survival",
  "In-hospital_death"
)


stopifnot(
  all(
    required_outcome_columns %in%
      names(outcomes_a)
  ),

  nrow(outcomes_a) ==
    N_A,

  n_distinct(
    outcomes_a$RecordID
  ) ==
    N_A,

  !anyDuplicated(
    outcomes_a$RecordID
  )
)


# Keep only:
#
#   - the primary outcome
#   - SAPS-I and SOFA for later benchmark comparisons
#
# Length_of_stay and Survival are deliberately not copied into the
# modeling table because they are unavailable at the 48-hour prediction
# time and could create outcome/time leakage.


outcomes_a_model <- outcomes_a |>
  select(
    RecordID,
    `In-hospital_death`,
    `SAPS-I`,
    SOFA
  )


features_a_model <- features_a |>
  left_join(
    outcomes_a_model,
    by =
      "RecordID"
  )


stopifnot(
  !any(
    is.na(
      features_a_model$
        `In-hospital_death`
    )
  )
)


# =============================================================================
# 20. Explicitly verify forbidden predictors
# =============================================================================

forbidden_predictors <- c(
  "Length_of_stay",
  "Survival"
)


stopifnot(
  !any(
    forbidden_predictors %in%
      names(features_a_model)
  ),

  !any(
    forbidden_predictors %in%
      names(features_b)
  )
)


# =============================================================================
# 21. Feature dictionary
# =============================================================================
#
# Build a simple machine-readable dictionary documenting every feature.
#
# This will be useful in the Checkpoint 2 report and makes the feature
# engineering decisions auditable.


feature_names <- setdiff(
  names(features_a),
  "RecordID"
)


classify_feature <- function(feature) {

  case_when(

    feature %in%
      static_parameters ~
      "static",

    feature ==
      "Weight_first" ~
      "static/first",

    grepl(
      "_first$",
      feature
    ) ~
      "summary:first",

    grepl(
      "_last$",
      feature
    ) ~
      "summary:last",

    grepl(
      "_min$",
      feature
    ) ~
      "summary:min",

    grepl(
      "_max$",
      feature
    ) ~
      "summary:max",

    grepl(
      "_mean$",
      feature
    ) ~
      "summary:mean",

    grepl(
      "_sd$",
      feature
    ) ~
      "summary:sd",

    grepl(
      "_delta$",
      feature
    ) ~
      "trend:delta",

    grepl(
      "_slope$",
      feature
    ) ~
      "trend:slope",

    grepl(
      "_count$",
      feature
    ) ~
      "measurement:count",

    grepl(
      "_missing$",
      feature
    ) ~
      "missingness:indicator",

    feature ==
      "MechVent_ever" ~
      "clinical:binary",

    feature ==
      "Urine_total" ~
      "clinical:aggregate",

    TRUE ~
      "other"
  )
}


extract_source_parameter <- function(feature) {

  if (
    feature %in%
      static_parameters
  ) {
    return(feature)
  }

  if (
    startsWith(
      feature,
      "Weight_"
    )
  ) {
    return("Weight")
  }

  if (
    startsWith(
      feature,
      "MechVent_"
    )
  ) {
    return("MechVent")
  }

  if (
    startsWith(
      feature,
      "Urine_"
    )
  ) {
    return("Urine")
  }


  suffixes <- c(
    "_first",
    "_last",
    "_min",
    "_max",
    "_mean",
    "_sd",
    "_delta",
    "_slope",
    "_count",
    "_missing"
  )


  source <- feature

  for (
    suffix in suffixes
  ) {

    source <- sub(
      paste0(
        suffix,
        "$"
      ),
      "",
      source
    )
  }

  source
}


feature_dictionary <- tibble(

  feature =
    feature_names,

  source_parameter =
    vapply(
      feature_names,
      extract_source_parameter,
      character(1)
    ),

  feature_type =
    vapply(
      feature_names,
      classify_feature,
      character(1)
    )
)


# =============================================================================
# 22. Validate feature values
# =============================================================================

# Explicit missingness indicators must contain only 0/1.

missing_cols <- grep(
  "_missing$",
  names(features_a),
  value = TRUE
)


for (
  col in missing_cols
) {

  values_a <- unique(
    features_a[[col]]
  )

  values_b <- unique(
    features_b[[col]]
  )

  stopifnot(
    all(
      values_a %in%
        c(0, 1)
    ),

    all(
      values_b %in%
        c(0, 1)
    )
  )
}


# Counts must never be negative.

count_cols <- grep(
  "_count$",
  names(features_a),
  value = TRUE
)


for (
  col in count_cols
) {

  stopifnot(
    all(
      features_a[[col]] >= 0,
      na.rm = TRUE
    ),

    all(
      features_b[[col]] >= 0,
      na.rm = TRUE
    )
  )
}


# Mechanical ventilation feature must be binary.

stopifnot(
  all(
    features_a$MechVent_ever %in%
      c(0, 1)
  ),

  all(
    features_b$MechVent_ever %in%
      c(0, 1)
  )
)


# =============================================================================
# 23. Save outputs
# =============================================================================

write_csv(
  features_a_model,
  here(
    "output",
    "set-a_features.csv"
  )
)


write_csv(
  features_b,
  here(
    "output",
    "set-b_features.csv"
  )
)


write_csv(
  feature_dictionary,
  here(
    "output",
    "feature_dictionary.csv"
  )
)


# =============================================================================
# 24. Final summary
# =============================================================================

n_predictors <- ncol(
  features_a
) - 1


n_missing_indicators <- length(
  grep(
    "_missing$",
    names(features_a),
    value = TRUE
  )
)


n_slope_features <- length(
  grep(
    "_slope$",
    names(features_a),
    value = TRUE
  )
)


n_delta_features <- length(
  grep(
    "_delta$",
    names(features_a),
    value = TRUE
  )
)


cat(
  "\n============================================================\n"
)

cat(
  "Checkpoint 2 feature construction complete.\n"
)

cat(
  "============================================================\n"
)

cat(
  "Set A ICU stays:",
  nrow(features_a),
  "\n"
)

cat(
  "Set B ICU stays:",
  nrow(features_b),
  "\n"
)

cat(
  "Engineered predictor columns:",
  n_predictors,
  "\n"
)

cat(
  "Missingness indicators:",
  n_missing_indicators,
  "\n"
)

cat(
  "Slope features:",
  n_slope_features,
  "\n"
)

cat(
  "Delta features:",
  n_delta_features,
  "\n"
)

cat(
  "\nImportant modeling rule:\n"
)

cat(
  paste(
    "No imputation, scaling, or feature selection",
    "has been performed in this script.\n"
  )
)

cat(
  paste(
    "Those steps must be estimated inside each",
    "cross-validation training fold.\n"
  )
)

cat(
  "\nGenerated outputs:\n"
)

cat(
  "  output/set-a_features.csv\n"
)

cat(
  "  output/set-b_features.csv\n"
)

cat(
  "  output/feature_dictionary.csv\n"
)
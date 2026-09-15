# BIOSTAT 707 Checkpoint 1
# Explore and clean Challenge 2012 Set A long-format data.

library(readr)
library(dplyr)
library(tidyr)
library(ggplot2)
library(here)

N_STAYS <- 4000

BP_VARS <- c(
  "SysABP", "DiasABP", "MAP",
  "NISysABP", "NIDiasABP", "NIMAP"
)


# -----------------------------------------------------------------------------
# 1. Load and validate long table
# -----------------------------------------------------------------------------

long_raw <- read_csv(
  here("output", "set-a_long.csv"),
  show_col_types = FALSE
)

stopifnot(
  all(c("RecordID", "Time", "Parameter", "Value") %in% names(long_raw)),
  n_distinct(long_raw$RecordID) == N_STAYS
)

cat("Rows:", nrow(long_raw), "\n")
cat("ICU stays:", n_distinct(long_raw$RecordID), "\n")
cat("Parameters:", n_distinct(long_raw$Parameter), "\n")


# -----------------------------------------------------------------------------
# 2. Parameter inventory and explicit -1 values
# -----------------------------------------------------------------------------

parameter_inventory <- long_raw |>
  summarise(
    n_measurements = n(),
    n_records = n_distinct(RecordID),
    .by = Parameter
  ) |>
  mutate(
    proportion_records = n_records / N_STAYS
  ) |>
  arrange(
    desc(n_records),
    Parameter
  )

write_csv(
  parameter_inventory,
  here("output", "parameter_inventory.csv")
)


minus_one_summary <- long_raw |>
  filter(Value == -1) |>
  count(
    Parameter,
    sort = TRUE,
    name = "n_minus_one"
  )

cat("\nExplicit -1 values:\n")
print(minus_one_summary, n = Inf)


# -----------------------------------------------------------------------------
# 3. Conservative cleaning and time conversion
# -----------------------------------------------------------------------------
#
# Cleaning decisions:
#
# - -1 is the documented missing-value code.
# - Clearly invalid pH, temperature, height, HR, and BP values become NA.
# - Extreme but not clearly invalid values are retained.
# - Original Value is preserved; cleaning is stored in Value_clean.


long_clean <- long_raw |>
  mutate(
    Value_clean = case_when(

      Value == -1 &
        Parameter != "RecordID" ~ NA_real_,

      Parameter == "pH" &
        (Value < 0 | Value > 14) ~ NA_real_,

      Parameter == "Temp" &
        (Value < 30 | Value > 45) ~ NA_real_,

      Parameter == "Height" &
        (Value < 100 | Value > 250) ~ NA_real_,

      Parameter == "HR" &
        Value == 0 ~ NA_real_,

      Parameter %in% BP_VARS &
        Value <= 0 ~ NA_real_,

      TRUE ~ Value
    ),

    Time_chr = as.character(Time),

    time_hour = as.numeric(
      sub(":.*", "", Time_chr)
    ),

    time_minute = as.numeric(
      sub(".*:", "", Time_chr)
    ),

    hours_since_admission =
      time_hour + time_minute / 60
  ) |>
  select(-Time_chr)


stopifnot(
  all(!is.na(long_clean$hours_since_admission)),
  all(between(long_clean$hours_since_admission, 0, 48))
)


# -----------------------------------------------------------------------------
# 4. Post-cleaning value summaries
# -----------------------------------------------------------------------------

safe_stat <- function(x, fun, ...) {
  if (all(is.na(x))) {
    NA_real_
  } else {
    as.numeric(fun(x, na.rm = TRUE, ...))
  }
}


value_summary <- long_clean |>
  filter(Parameter != "RecordID") |>
  summarise(
    n_rows = n(),

    n_missing_after_cleaning =
      sum(is.na(Value_clean)),

    proportion_missing_after_cleaning =
      mean(is.na(Value_clean)),

    min =
      safe_stat(Value_clean, min),

    q01 =
      safe_stat(
        Value_clean,
        quantile,
        probs = 0.01
      ),

    median =
      safe_stat(Value_clean, median),

    mean =
      safe_stat(Value_clean, mean),

    q99 =
      safe_stat(
        Value_clean,
        quantile,
        probs = 0.99
      ),

    max =
      safe_stat(Value_clean, max),

    .by = Parameter
  ) |>
  arrange(Parameter)


write_csv(
  value_summary,
  here("output", "long_value_summary.csv")
)


# -----------------------------------------------------------------------------
# 5. Post-cleaning validation
# -----------------------------------------------------------------------------

suspicious_values <- long_clean |>
  filter(
    !is.na(Value_clean),

    case_when(

      Parameter == "Gender" ~
        !Value_clean %in% c(0, 1),

      Parameter == "ICUType" ~
        !Value_clean %in% 1:4,

      Parameter == "GCS" ~
        !between(Value_clean, 3, 15),

      Parameter == "pH" ~
        !between(Value_clean, 0, 14),

      Parameter == "FiO2" ~
        !between(Value_clean, 0, 1),

      Parameter == "Temp" ~
        !between(Value_clean, 30, 45),

      Parameter == "Height" ~
        !between(Value_clean, 100, 250),

      Parameter == "HR" ~
        Value_clean == 0,

      Parameter %in% BP_VARS ~
        Value_clean <= 0,

      TRUE ~ FALSE
    )
  ) |>
  select(
    RecordID,
    Time,
    Parameter,
    Value,
    Value_clean
  )


write_csv(
  suspicious_values,
  here("output", "suspicious_values.csv")
)

stopifnot(
  nrow(suspicious_values) == 0
)


# -----------------------------------------------------------------------------
# 6. Measurement presence by parameter
# -----------------------------------------------------------------------------

presence_by_parameter <- long_clean |>
  filter(
    Parameter != "RecordID",
    !is.na(Value_clean)
  ) |>
  distinct(
    RecordID,
    Parameter
  ) |>
  count(
    Parameter,
    name = "n_records_measured"
  ) |>
  mutate(
    proportion_measured =
      n_records_measured / N_STAYS,

    proportion_not_measured =
      1 - proportion_measured
  ) |>
  arrange(
    desc(proportion_measured)
  )


write_csv(
  presence_by_parameter,
  here("output", "presence_by_parameter.csv")
)


presence_plot <- ggplot(
  presence_by_parameter,
  aes(
    x = reorder(
      Parameter,
      proportion_measured
    ),
    y = proportion_measured
  )
) +
  geom_col() +
  coord_flip() +
  scale_y_continuous(
    labels =
      scales::percent_format(
        accuracy = 1
      )
  ) +
  labs(
    title =
      "Challenge 2012 Set A: measurement presence",

    subtitle =
      "Proportion of 4,000 ICU stays with at least one observed value",

    x = NULL,

    y =
      "Admissions with at least one measurement"
  ) +
  theme_minimal(
    base_size = 11
  )


ggsave(
  here("output", "presence_by_parameter.png"),
  presence_plot,
  width = 8,
  height = 8,
  dpi = 150
)


# -----------------------------------------------------------------------------
# 7. Temporal measurement presence
# -----------------------------------------------------------------------------
#
# Divide the 48-hour observation period into 4-hour bins.


long_temporal <- long_clean |>
  filter(
    Parameter != "RecordID",
    !is.na(Value_clean)
  ) |>
  mutate(
    time_bin_start =
      pmin(
        floor(hours_since_admission / 4) * 4,
        44
      )
  )


measurement_rate <- long_temporal |>
  distinct(
    RecordID,
    Parameter,
    time_bin_start
  ) |>
  count(
    Parameter,
    time_bin_start,
    name = "n_records_measured"
  ) |>
  mutate(
    proportion_measured =
      n_records_measured / N_STAYS
  )


write_csv(
  measurement_rate,
  here("output", "measurement_rate_4h.csv")
)


# -----------------------------------------------------------------------------
# 8. Plot temporal measurement rates
# -----------------------------------------------------------------------------

plot_vars <- intersect(
  c(
    "HR",
    "Temp",
    "Glucose",
    "Creatinine",
    "Lactate"
  ),
  unique(measurement_rate$Parameter)
)


temporal_plot <- measurement_rate |>
  filter(
    Parameter %in% plot_vars
  ) |>
  ggplot(
    aes(
      x = time_bin_start + 2,
      y = proportion_measured,
      linetype = Parameter
    )
  ) +
  geom_line(
    linewidth = 0.8
  ) +
  geom_point(
    size = 2
  ) +
  scale_x_continuous(
    breaks = seq(2, 46, 4),
    labels = paste0(
      seq(0, 44, 4),
      "-",
      seq(4, 48, 4)
    )
  ) +
  scale_y_continuous(
    labels =
      scales::percent_format(
        accuracy = 1
      )
  ) +
  labs(
    title =
      "Measurement presence over time",

    subtitle =
      "Selected variables in 4-hour bins",

    x =
      "Hours since ICU admission",

    y =
      "Admissions with at least one measurement"
  ) +
  theme_minimal(
    base_size = 11
  )


ggsave(
  here("output", "measurement_rate_over_time.png"),
  temporal_plot,
  width = 8,
  height = 6,
  dpi = 150
)


# -----------------------------------------------------------------------------
# 9. Lasagna plot
# -----------------------------------------------------------------------------
#
# Each row represents one ICU stay.
# Each column represents one 4-hour time interval.
# Presence means at least one measurement occurred in that interval.


lasagna_parameter <- if (
  "Glucose" %in% long_temporal$Parameter
) {
  "Glucose"
} else {
  "HR"
}


lasagna_presence <- long_temporal |>
  filter(
    Parameter == lasagna_parameter
  ) |>
  distinct(
    RecordID,
    time_bin_start
  ) |>
  mutate(
    present = 1L
  )


lasagna_matrix <- expand_grid(
  RecordID =
    unique(long_clean$RecordID),

  time_bin_start =
    seq(0, 44, 4)
) |>
  left_join(
    lasagna_presence,
    by = c(
      "RecordID",
      "time_bin_start"
    )
  ) |>
  mutate(
    present =
      replace_na(
        present,
        0L
      )
  )


record_order <- lasagna_matrix |>
  summarise(
    total_present =
      sum(present),
    .by = RecordID
  ) |>
  arrange(
    desc(total_present),
    RecordID
  ) |>
  pull(RecordID)


lasagna_matrix <- lasagna_matrix |>
  mutate(
    record_order =
      match(
        RecordID,
        record_order
      )
  )


lasagna_plot <- ggplot(
  lasagna_matrix,
  aes(
    x = time_bin_start,
    y = record_order,
    fill = factor(present)
  )
) +
  geom_tile(
    width = 4,
    height = 1
  ) +
  scale_x_continuous(
    breaks = seq(0, 44, 4),

    labels = paste0(
      seq(0, 44, 4),
      "-",
      seq(4, 48, 4)
    )
  ) +
  scale_fill_manual(
    values = c(
      "0" = "grey90",
      "1" = "#377eb8"
    ),

    labels = c(
      "Absent",
      "Present"
    )
  ) +
  labs(
    title =
      paste(
        lasagna_parameter,
        "measurement presence over time"
      ),

    x =
      "Hours since ICU admission",

    y =
      "ICU stays",

    fill =
      "Measurement"
  ) +
  theme_minimal(
    base_size = 10
  ) +
  theme(
    axis.text.x =
      element_text(
        angle = 45,
        hjust = 1
      ),

    axis.text.y =
      element_blank(),

    axis.ticks.y =
      element_blank(),

    panel.grid =
      element_blank()
  )


ggsave(
  here("output", "lasagna_presence.png"),
  lasagna_plot,
  width = 8,
  height = 8,
  dpi = 150
)


# -----------------------------------------------------------------------------
# 10. Save cleaned long table
# -----------------------------------------------------------------------------

write_csv(
  long_clean,
  here("output", "set-a_long_clean.csv")
)


# -----------------------------------------------------------------------------
# 11. Final summary
# -----------------------------------------------------------------------------

cat(
  "\n============================================================\n",
  "Long-table exploration and cleaning complete.\n",
  "============================================================\n",
  sep = ""
)

cat(
  "Rows:",
  nrow(long_clean),
  "\n"
)

cat(
  "ICU stays:",
  n_distinct(long_clean$RecordID),
  "\n"
)

cat(
  "Post-cleaning validation violations:",
  nrow(suspicious_values),
  "\n"
)
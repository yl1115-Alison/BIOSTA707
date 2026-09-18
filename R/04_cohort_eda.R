# BIOSTAT 707 Checkpoint 1
# Cohort characterization, outcomes, and wide-table missingness.

library(readr)
library(dplyr)
library(tidyr)
library(ggplot2)
library(here)

N_STAYS <- 4000


# -----------------------------------------------------------------------------
# 1. Load and validate the wide table
# -----------------------------------------------------------------------------

wide <- read_csv(
  here("output", "set-a_wide.csv"),
  show_col_types = FALSE
)

required <- c(
  "RecordID", "Age", "Gender", "Height", "ICUType", "Weight_first",
  "SAPS-I", "SOFA", "Length_of_stay", "Survival", "In-hospital_death"
)

stopifnot(
  all(required %in% names(wide)),
  nrow(wide) == N_STAYS,
  n_distinct(wide$RecordID) == N_STAYS,
  !anyDuplicated(wide$RecordID)
)


# Make variables easier to read in tables and plots.

cohort <- wide |>
  transmute(
    RecordID,
    Age,

    Gender = factor(
      Gender,
      levels = c(0, 1),
      labels = c("Female", "Male")
    ),

    Height,

    Weight = Weight_first,

    ICUType = factor(
      ICUType,
      levels = 1:4,
      labels = c(
        "Coronary Care Unit",
        "Cardiac Surgery Recovery Unit",
        "Medical ICU",
        "Surgical ICU"
      )
    ),

    Death = factor(
      `In-hospital_death`,
      levels = c(0, 1),
      labels = c("Survived", "Died")
    ),

    SAPS_I = `SAPS-I`,
    SOFA,
    Length_of_stay,
    Survival
  )

stopifnot(
  !any(is.na(cohort$Death))
)


# -----------------------------------------------------------------------------
# 2. Table 1
# -----------------------------------------------------------------------------

# Continuous variables: mean (SD)

fmt_cont <- function(x) {

  if (all(is.na(x))) {
    return("NA")
  }

  sprintf(
    "%.1f (%.1f)",
    mean(x, na.rm = TRUE),
    sd(x, na.rm = TRUE)
  )
}


# Categorical variables: n (%)

fmt_cat <- function(x, level) {

  n <- sum(
    x == level,
    na.rm = TRUE
  )

  denominator <- sum(
    !is.na(x)
  )

  if (denominator == 0) {
    return("0 (NA)")
  }

  sprintf(
    "%d (%.1f%%)",
    n,
    100 * n / denominator
  )
}


make_table1_column <- function(df) {

  c(
    as.character(nrow(df)),

    fmt_cont(df$Age),

    fmt_cat(
      df$Gender,
      "Male"
    ),

    fmt_cont(df$Height),

    fmt_cont(df$Weight),

    fmt_cat(
      df$ICUType,
      "Coronary Care Unit"
    ),

    fmt_cat(
      df$ICUType,
      "Cardiac Surgery Recovery Unit"
    ),

    fmt_cat(
      df$ICUType,
      "Medical ICU"
    ),

    fmt_cat(
      df$ICUType,
      "Surgical ICU"
    )
  )
}


survivors <- cohort |>
  filter(
    Death == "Survived"
  )

deaths <- cohort |>
  filter(
    Death == "Died"
  )


table1 <- tibble(

  Characteristic = c(
    "N",
    "Age, mean (SD)",
    "Male, n (%)",
    "Height, mean (SD)",
    "First weight, mean (SD)",
    "ICU type: Coronary Care Unit, n (%)",
    "ICU type: Cardiac Surgery Recovery Unit, n (%)",
    "ICU type: Medical ICU, n (%)",
    "ICU type: Surgical ICU, n (%)"
  ),

  Overall =
    make_table1_column(
      cohort
    ),

  Survived =
    make_table1_column(
      survivors
    ),

  Died =
    make_table1_column(
      deaths
    )
)


write_csv(
  table1,
  here(
    "output",
    "table1.csv"
  )
)


# -----------------------------------------------------------------------------
# 3. Outcome summary
# -----------------------------------------------------------------------------

q1 <- function(x) {
  quantile(
    x,
    0.25,
    na.rm = TRUE
  )
}

q3 <- function(x) {
  quantile(
    x,
    0.75,
    na.rm = TRUE
  )
}


outcome_summary <- tibble(

  Measure = c(
    "ICU stays",
    "Survivors",
    "Deaths",
    "In-hospital mortality",
    "SAPS-I, median (IQR)",
    "SOFA, median (IQR)",
    "Length of stay, median (IQR)"
  ),

  Value = c(

    as.character(
      nrow(cohort)
    ),

    as.character(
      sum(
        cohort$Death ==
          "Survived"
      )
    ),

    as.character(
      sum(
        cohort$Death ==
          "Died"
      )
    ),

    sprintf(
      "%.1f%%",
      100 *
        mean(
          cohort$Death ==
            "Died"
        )
    ),

    sprintf(
      "%.1f (%.1f-%.1f)",
      median(
        cohort$SAPS_I,
        na.rm = TRUE
      ),
      q1(cohort$SAPS_I),
      q3(cohort$SAPS_I)
    ),

    sprintf(
      "%.1f (%.1f-%.1f)",
      median(
        cohort$SOFA,
        na.rm = TRUE
      ),
      q1(cohort$SOFA),
      q3(cohort$SOFA)
    ),

    sprintf(
      "%.1f (%.1f-%.1f)",
      median(
        cohort$Length_of_stay,
        na.rm = TRUE
      ),
      q1(
        cohort$Length_of_stay
      ),
      q3(
        cohort$Length_of_stay
      )
    )
  )
)


write_csv(
  outcome_summary,
  here(
    "output",
    "outcome_summary.csv"
  )
)


# Mortality plot

mortality_counts <- cohort |>
  count(Death) |>
  mutate(
    proportion =
      n / sum(n)
  )


mortality_plot <- ggplot(
  mortality_counts,
  aes(
    x = Death,
    y = n
  )
) +
  geom_col() +

  geom_text(
    aes(
      label = sprintf(
        "%d (%.1f%%)",
        n,
        100 * proportion
      )
    ),
    vjust = -0.4
  ) +

  labs(
    title =
      "In-hospital mortality in Challenge 2012 Set A",

    x = NULL,

    y =
      "ICU stays"
  ) +

  theme_minimal(
    base_size = 11
  )


ggsave(
  here(
    "output",
    "outcome_mortality.png"
  ),
  mortality_plot,
  width = 6,
  height = 4,
  dpi = 150
)


# -----------------------------------------------------------------------------
# 4. Wide-table missingness
# -----------------------------------------------------------------------------

# First summarize literal NA values in all wide-table columns.

wide_missingness <- tibble(

  variable =
    names(wide),

  n_missing =
    colSums(
      is.na(wide)
    ),

  proportion_missing =
    colMeans(
      is.na(wide)
    )
) |>
  arrange(
    desc(
      proportion_missing
    ),
    variable
  )


write_csv(
  wide_missingness,
  here(
    "output",
    "wide_missingness.csv"
  )
)


# For longitudinal variables, _count tells us whether the variable
# was measured at all.
#
# Example:
#
# Lactate_count = 0
#
# means no valid Lactate measurement was observed during the 48 hours.


count_cols <- grep(
  "_count$",
  names(wide),
  value = TRUE
)

stopifnot(
  length(count_cols) > 0
)


presence <- wide |>
  select(
    RecordID,
    all_of(count_cols)
  ) |>
  rename_with(
    ~ sub(
      "_count$",
      "",
      .x
    ),
    all_of(count_cols)
  ) |>
  pivot_longer(
    -RecordID,
    names_to = "Parameter",
    values_to = "count"
  ) |>
  mutate(
    present =
      as.integer(
        count > 0
      )
  )


presence_summary <- presence |>
  summarise(

    n_measured =
      sum(present),

    n_not_measured =
      N_STAYS -
      n_measured,

    proportion_measured =
      mean(present),

    proportion_not_measured =
      1 -
      proportion_measured,

    .by =
      Parameter
  ) |>
  arrange(
    proportion_measured
  )


write_csv(
  presence_summary,
  here(
    "output",
    "wide_presence_summary.csv"
  )
)

# -------------------------------------------------------------------------
# Wide-table missingness bar plot
# -------------------------------------------------------------------------

wide_missingness_plot <- ggplot(
  presence_summary,
  aes(
    x = proportion_not_measured,
    y = reorder(
      Parameter,
      proportion_not_measured
    )
  )
) +
  geom_col() +
  scale_x_continuous(
    labels = scales::percent_format(
      accuracy = 1
    ),
    limits = c(0, 1)
  ) +
  labs(
    title = "Missingness in the Wide Table",
    subtitle =
      "Percent of ICU stays without any valid measurement during the first 48 hours",
    x = "% missing",
    y = "Clinical variable"
  ) +
  theme_minimal(
    base_size = 11
  )

ggsave(
  here(
    "output",
    "wide_missingness_bar.png"
  ),
  wide_missingness_plot,
  width = 8,
  height = 9,
  dpi = 150
)

# -----------------------------------------------------------------------------
# 5. Wide-table presence map
# -----------------------------------------------------------------------------

parameter_order <-
  presence_summary$Parameter


record_order <- presence |>
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


presence_plot_data <- presence |>
  mutate(

    Parameter =
      factor(
        Parameter,
        levels =
          parameter_order
      ),

    record_order =
      match(
        RecordID,
        record_order
      )
  )


wide_presence_plot <- ggplot(
  presence_plot_data,
  aes(
    x = Parameter,
    y = record_order,
    fill = factor(present)
  )
) +

  geom_tile() +

  scale_fill_manual(
    values = c(
      "0" = "grey90",
      "1" = "#377eb8"
    ),

    labels = c(
      "Not measured",
      "Measured"
    )
  ) +

  labs(
    title =
      "Measurement presence across ICU stays",

    subtitle =
      paste(
        "One column per longitudinal variable",
        "over the full 48-hour window"
      ),

    x = NULL,

    y =
      "ICU stays",

    fill = NULL
  ) +

  theme_minimal(
    base_size = 9
  ) +

  theme(
    axis.text.x =
      element_text(
        angle = 60,
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
  here(
    "output",
    "wide_presence_map.png"
  ),
  wide_presence_plot,
  width = 11,
  height = 7,
  dpi = 150
)


# -----------------------------------------------------------------------------
# 6. Informative presence
# -----------------------------------------------------------------------------
#
# Compare:
#
# P(death | variable measured)
#
# with
#
# P(death | variable not measured)
#
# This is descriptive association, not a causal effect.


presence_outcome <- presence |>
  left_join(

    wide |>
      select(
        RecordID,
        `In-hospital_death`
      ),

    by =
      "RecordID"
  )


informative_presence <- presence_outcome |>
  summarise(

    n_measured =
      sum(
        present == 1
      ),

    n_not_measured =
      sum(
        present == 0
      ),

    mortality_if_measured =
      if (
        any(
          present == 1
        )
      ) {

        mean(
          `In-hospital_death`[
            present == 1
          ] == 1
        )

      } else {

        NA_real_
      },

    mortality_if_not_measured =
      if (
        any(
          present == 0
        )
      ) {

        mean(
          `In-hospital_death`[
            present == 0
          ] == 1
        )

      } else {

        NA_real_
      },

    .by =
      Parameter
  ) |>
  mutate(

    mortality_difference =
      mortality_if_measured -
      mortality_if_not_measured
  ) |>
  arrange(
    desc(
      abs(
        mortality_difference
      )
    )
  )


write_csv(
  informative_presence,
  here(
    "output",
    "informative_presence.csv"
  )
)


# Plot only variables with enough patients in both groups.

informative_plot_data <-
  informative_presence |>
  filter(
    n_measured >= 20,
    n_not_measured >= 20,
    !is.na(
      mortality_difference
    )
  ) |>
  slice_max(
    abs(
      mortality_difference
    ),
    n = 15,
    with_ties = FALSE
  )


informative_plot <- ggplot(
  informative_plot_data,
  aes(
    x = reorder(
      Parameter,
      mortality_difference
    ),
    y =
      mortality_difference
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
      "Outcome differences associated with measurement presence",

    subtitle =
      "Mortality when measured minus mortality when not measured",

    x = NULL,

    y =
      "Mortality-rate difference"
  ) +

  theme_minimal(
    base_size = 11
  )


ggsave(
  here(
    "output",
    "informative_presence.png"
  ),
  informative_plot,
  width = 8,
  height = 6,
  dpi = 150
)


# -----------------------------------------------------------------------------
# 7. Final validation
# -----------------------------------------------------------------------------

stopifnot(

  nrow(cohort) ==
    N_STAYS,

  sum(
    mortality_counts$n
  ) ==
    N_STAYS,

  n_distinct(
    presence$RecordID
  ) ==
    N_STAYS
)


cat(
  "\n============================================================\n"
)

cat(
  "Cohort EDA complete.\n"
)

cat(
  "============================================================\n"
)

cat(
  "ICU stays:",
  nrow(cohort),
  "\n"
)

cat(
  "Deaths:",
  sum(
    cohort$Death ==
      "Died"
  ),
  "\n"
)

cat(
  "Mortality rate:",
  sprintf(
    "%.1f%%",
    100 *
      mean(
        cohort$Death ==
          "Died"
      )
  ),
  "\n"
)

cat(
  "Longitudinal variables in presence analysis:",
  length(count_cols),
  "\n"
)

cat(
  "\nGenerated outputs:\n"
)

cat(
  "  output/table1.csv\n"
)

cat(
  "  output/outcome_summary.csv\n"
)

cat(
  "  output/outcome_mortality.png\n"
)

cat(
  "  output/wide_missingness.csv\n"
)

cat(
  "  output/wide_missingness_bar.png\n"
)

cat(
  "  output/wide_presence_map.png\n"
)

cat(
  "  output/informative_presence.csv\n"
)

cat(
  "  output/informative_presence.png\n"
)
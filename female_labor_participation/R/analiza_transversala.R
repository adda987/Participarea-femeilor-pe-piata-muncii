#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  required_packages <- c(
    "readxl", "dplyr", "tidyr", "ggplot2", "car", "lmtest", "tseries",
    "nortest", "moments", "caret", "glmnet", "MLmetrics", "broom",
    "sandwich", "jsonlite", "corrplot", "patchwork", "scales"
  )
  missing_packages <- required_packages[!vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)]
  if (length(missing_packages) > 0) {
    stop(
      paste0(
        "Pachete R lipsă: ", paste(missing_packages, collapse = ", "),
        ". Instalează-le manual înainte de rularea analizei transversale."
      ),
      call. = FALSE
    )
  }
  invisible(lapply(required_packages, require, character.only = TRUE))
})

ANALYSIS_YEAR <- 2023L
SIMPLE_X <- "inflation"

NAVY <- "#10243E"
TEAL <- "#147D7E"
BURGUNDY <- "#7C2748"
PLUM <- "#67406F"
GOLD <- "#D3A13B"
IVORY <- "#F7F3EA"
INK <- "#172033"
MUTED <- "#687083"

find_project_root <- function() {
  candidates <- c(normalizePath(getwd(), winslash = "/", mustWork = TRUE))
  candidates <- c(candidates, file.path(candidates, "female_labor_participation"))
  for (candidate in candidates) {
    dataset <- file.path(candidate, "data", "P_Data_Extract_From_World_Development_Indicators.xlsx")
    script <- file.path(candidate, "R", "analiza_transversala.R")
    if (file.exists(dataset) && file.exists(script)) {
      return(normalizePath(candidate, winslash = "/", mustWork = TRUE))
    }
  }
  stop("Nu pot identifica rădăcina proiectului pentru analiza transversală.", call. = FALSE)
}

ROOT <- find_project_root()
DATA_PATH <- file.path(ROOT, "data", "P_Data_Extract_From_World_Development_Indicators.xlsx")
COUNTRY_CONFIG_PATH <- file.path(ROOT, "config", "europe_countries.csv")
OUTPUT_ROOT <- file.path(ROOT, "outputs", "cross_sectional")
TABLE_DIR <- file.path(OUTPUT_ROOT, "tables")
FIGURE_DIR <- file.path(OUTPUT_ROOT, "figures")
MODEL_DIR <- file.path(OUTPUT_ROOT, "models")
META_DIR <- file.path(OUTPUT_ROOT, "metadata")

dir.create(TABLE_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create(FIGURE_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create(MODEL_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create(META_DIR, recursive = TRUE, showWarnings = FALSE)

theme_project <- function(base_size = 12) {
  ggplot2::theme_minimal(base_size = base_size) +
    ggplot2::theme(
      plot.background = ggplot2::element_rect(fill = "white", color = NA),
      panel.background = ggplot2::element_rect(fill = "white", color = NA),
      panel.grid.major = ggplot2::element_line(color = "#E8E2D8", linewidth = 0.35),
      panel.grid.minor = ggplot2::element_blank(),
      plot.title = ggplot2::element_text(color = NAVY, face = "bold", size = base_size + 4),
      plot.subtitle = ggplot2::element_text(color = INK, size = base_size),
      axis.title = ggplot2::element_text(color = NAVY, face = "bold"),
      axis.text = ggplot2::element_text(color = INK),
      legend.position = "bottom",
      legend.title = ggplot2::element_text(color = NAVY, face = "bold"),
      legend.text = ggplot2::element_text(color = INK)
    )
}

variables <- tibble::tribble(
  ~slug, ~code, ~label, ~short_label, ~axis_label,
  "female_lfpr", "SL.TLF.ACTI.FE.ZS", "Participarea femeilor", "Participarea femeilor", "Participarea femeilor la forța de muncă (%)",
  "gdp_pc_ppp", "NY.GDP.PCAP.PP.KD", "PIB real pe locuitor, PPP", "PIB/locuitor PPP", "PIB real pe locuitor, PPP",
  "gdp_growth", "NY.GDP.MKTP.KD.ZG", "Creșterea PIB real", "Creștere PIB", "Creșterea PIB real (%)",
  "inflation", "FP.CPI.TOTL.ZG", "Inflația, prețurile de consum", "Inflație", "Inflația, prețurile de consum (%)",
  "female_unemployment", "SL.UEM.TOTL.FE.ZS", "Șomajul femeilor", "Șomaj feminin", "Șomajul femeilor (%)",
  "fertility", "SP.DYN.TFRT.IN", "Rata totală a fertilității", "Fertilitate", "Rata totală a fertilității",
  "urbanization", "SP.URB.TOTL.IN.ZS", "Urbanizarea", "Urbanizare", "Urbanizarea (% din populație)",
  "female_services", "SL.SRV.EMPL.FE.ZS", "Femeile ocupate în servicii", "Femei în servicii", "Femei ocupate în servicii (%)",
  "child_dependency", "SP.POP.DPND.YG", "Rata de dependență a copiilor", "Dependența copiilor", "Rata de dependență a copiilor",
  "female_vulnerable", "SL.EMP.VULN.FE.ZS", "Ocuparea vulnerabilă a femeilor", "Ocupare vulnerabilă", "Ocuparea vulnerabilă a femeilor (%)",
  "internet", "IT.NET.USER.ZS", "Utilizarea internetului", "Internet", "Utilizarea internetului (% din populație)",
  "women_parliament", "SG.GEN.PARL.ZS", "Femeile în parlamentele naționale", "Femei în parlament", "Femei în parlamentele naționale (%)",
  "female_tertiary", "SE.TER.ENRR.FE", "Educația terțiară a femeilor", "Educație terțiară", "Înscrierea femeilor în învățământul terțiar (%)",
  "control_corruption", "GOV_WGI_CC_EST", "Controlul corupției", "Controlul corupției", "Controlul corupției"
)

label_for <- function(term) {
  if (term == "(Intercept)") {
    return("Intercept")
  }
  if (term == "LogGDP") {
    return("Log PIB/locuitor PPP")
  }
  if (term == "DigitalizareRidicata") {
    return("Digitalizare ridicată")
  }
  if (term == "CenteredLogGDP") {
    return("Log PIB centrat")
  }
  if (term == "CenteredLogGDP2") {
    return("Log PIB centrat^2")
  }
  if (grepl(":", term, fixed = TRUE)) {
    parts <- strsplit(term, ":", fixed = TRUE)[[1]]
    return(paste(vapply(parts, label_for, character(1)), collapse = " × "))
  }
  match <- variables[variables$slug == term, ]
  if (nrow(match) == 1) {
    return(match$short_label[[1]])
  }
  term
}

axis_for <- function(slug) {
  if (slug == "LogGDP") {
    return("Log PIB real pe locuitor, PPP")
  }
  match <- variables[variables$slug == slug, ]
  if (nrow(match) == 1) {
    return(match$axis_label[[1]])
  }
  label_for(slug)
}

clean_output <- function(dataframe) {
  dataframe <- as.data.frame(dataframe)
  for (column in names(dataframe)) {
    if (is.numeric(dataframe[[column]])) {
      dataframe[[column]][!is.finite(dataframe[[column]])] <- NA_real_
    }
  }
  dataframe
}

write_csv_output <- function(dataframe, filename) {
  utils::write.csv(
    clean_output(dataframe),
    file.path(TABLE_DIR, filename),
    row.names = FALSE,
    na = "",
    fileEncoding = "UTF-8"
  )
}

save_plot <- function(plot, filename, width = 10, height = 6) {
  ggplot2::ggsave(
    filename = file.path(FIGURE_DIR, filename),
    plot = plot,
    width = width,
    height = height,
    dpi = 160,
    bg = "white"
  )
}

find_column_by_code <- function(column_names, code) {
  matches <- column_names[grepl(paste0("[", code, "]"), column_names, fixed = TRUE)]
  if (length(matches) == 0) {
    stop(paste0("Nu am găsit coloana pentru codul World Bank: ", code), call. = FALSE)
  }
  matches[[1]]
}

coerce_numeric <- function(x) {
  suppressWarnings(as.numeric(x))
}

if (!file.exists(DATA_PATH)) {
  stop(paste0("Fișierul Excel nu există: ", DATA_PATH), call. = FALSE)
}
if (!file.exists(COUNTRY_CONFIG_PATH)) {
  stop(paste0("Fișierul config/europe_countries.csv nu există: ", COUNTRY_CONFIG_PATH), call. = FALSE)
}

raw_data <- suppressWarnings(readxl::read_excel(DATA_PATH, na = c("", "NA", "..")))
europe_countries <- read.csv(COUNTRY_CONFIG_PATH, stringsAsFactors = FALSE, fileEncoding = "UTF-8")
column_names <- names(raw_data)
mapped_columns <- stats::setNames(vapply(variables$code, function(code) find_column_by_code(column_names, code), character(1)), variables$slug)

analysis_data <- raw_data |>
  dplyr::transmute(
    year = coerce_numeric(.data[["Time"]]),
    country_original = as.character(.data[["Country Name"]]),
    iso3 = as.character(.data[["Country Code"]]),
    female_lfpr = coerce_numeric(.data[[mapped_columns[["female_lfpr"]]]]),
    gdp_pc_ppp = coerce_numeric(.data[[mapped_columns[["gdp_pc_ppp"]]]]),
    gdp_growth = coerce_numeric(.data[[mapped_columns[["gdp_growth"]]]]),
    inflation = coerce_numeric(.data[[mapped_columns[["inflation"]]]]),
    female_unemployment = coerce_numeric(.data[[mapped_columns[["female_unemployment"]]]]),
    fertility = coerce_numeric(.data[[mapped_columns[["fertility"]]]]),
    urbanization = coerce_numeric(.data[[mapped_columns[["urbanization"]]]]),
    female_services = coerce_numeric(.data[[mapped_columns[["female_services"]]]]),
    child_dependency = coerce_numeric(.data[[mapped_columns[["child_dependency"]]]]),
    female_vulnerable = coerce_numeric(.data[[mapped_columns[["female_vulnerable"]]]]),
    internet = coerce_numeric(.data[[mapped_columns[["internet"]]]]),
    women_parliament = coerce_numeric(.data[[mapped_columns[["women_parliament"]]]]),
    female_tertiary = coerce_numeric(.data[[mapped_columns[["female_tertiary"]]]]),
    control_corruption = coerce_numeric(.data[[mapped_columns[["control_corruption"]]]])
  ) |>
  dplyr::filter(!is.na(.data$year), !is.na(.data$iso3)) |>
  dplyr::mutate(
    year = as.integer(.data$year),
    LogGDP = dplyr::if_else(!is.na(.data$gdp_pc_ppp) & .data$gdp_pc_ppp > 0, log(.data$gdp_pc_ppp), NA_real_)
  )

data_2023 <- analysis_data |>
  dplyr::filter(.data$year == ANALYSIS_YEAR, .data$iso3 %in% europe_countries$iso3) |>
  dplyr::left_join(europe_countries, by = "iso3") |>
  dplyr::mutate(country = dplyr::coalesce(.data$country, .data$country_original)) |>
  dplyr::select("country", "iso3", dplyr::everything(), -"country_original", -"year")

complete_model_data <- function(dataframe, variables_required) {
  dataframe |>
    dplyr::select("country", "iso3", dplyr::all_of(variables_required)) |>
    dplyr::filter(dplyr::if_all(dplyr::all_of(variables_required), ~ !is.na(.x)))
}

fit_lm_checked <- function(formula, dataframe, predictor_count) {
  residual_df <- nrow(dataframe) - predictor_count - 1
  if (nrow(dataframe) <= predictor_count + 2 || residual_df <= 1) {
    return(NULL)
  }
  stats::lm(formula, data = dataframe)
}

format_number <- function(x, digits = 4) {
  if (length(x) == 0 || is.na(x) || !is.finite(x)) {
    return(NA_character_)
  }
  formatC(x, format = "f", digits = digits)
}

format_p <- function(x) {
  if (length(x) == 0 || is.na(x) || !is.finite(x)) {
    return(NA_character_)
  }
  if (x < 0.001) {
    return("<0.001")
  }
  formatC(x, format = "f", digits = 4)
}

model_metrics <- function(model, n_used, predictor_count, include_information = FALSE) {
  model_summary <- summary(model)
  fstat <- model_summary$fstatistic
  f_p <- if (!is.null(fstat) && length(fstat) >= 3) {
    stats::pf(fstat[[1]], fstat[[2]], fstat[[3]], lower.tail = FALSE)
  } else {
    NA_real_
  }
  metrics <- tibble::tibble(
    Indicator = c("N", "Predictori", "Grade de libertate reziduale", "R²", "R² ajustat", "Residual standard error", "F statistic", "p-value model"),
    Valoare = c(
      n_used,
      predictor_count,
      stats::df.residual(model),
      model_summary$r.squared,
      model_summary$adj.r.squared,
      model_summary$sigma,
      if (!is.null(fstat)) unname(fstat[[1]]) else NA_real_,
      f_p
    )
  )
  if (include_information) {
    metrics <- dplyr::bind_rows(
      metrics,
      tibble::tibble(
        Indicator = c("AIC", "BIC", "RMSE in-sample"),
        Valoare = c(stats::AIC(model), stats::BIC(model), sqrt(mean(stats::residuals(model)^2)))
      )
    )
  }
  metrics
}

tidy_coefficients <- function(model) {
  broom::tidy(model, conf.int = TRUE, conf.level = 0.95) |>
    dplyr::mutate(Termen = vapply(.data$term, label_for, character(1))) |>
    dplyr::transmute(
      Termen,
      Coeficient = .data$estimate,
      `Standard error` = .data$std.error,
      `Statistică t` = .data$statistic,
      `p-value` = .data$p.value,
      `IC 95% inferior` = .data$conf.low,
      `IC 95% superior` = .data$conf.high
    )
}

conclude_test <- function(p_value, issue_text, no_issue_text) {
  if (is.na(p_value) || !is.finite(p_value)) {
    return("Testul nu a putut fi calculat stabil pentru eșantionul disponibil.")
  }
  if (p_value < 0.05) {
    paste("Respingem H0 la pragul de 5%.", issue_text)
  } else {
    paste("Nu respingem H0 la pragul de 5%.", no_issue_text)
  }
}

test_row <- function(category, test, h0, statistic, p_value, issue_text, no_issue_text) {
  tibble::tibble(
    Categorie = category,
    Test = test,
    H0 = h0,
    Statistică = statistic,
    `p-value` = p_value,
    Decizie = dplyr::case_when(
      is.na(p_value) ~ "Nu poate fi calculat",
      p_value < 0.05 ~ "Respingem H0",
      TRUE ~ "Nu respingem H0"
    ),
    Interpretare = conclude_test(p_value, issue_text, no_issue_text)
  )
}

safe_htest <- function(expr) {
  tryCatch(expr, error = function(e) NULL, warning = function(w) suppressWarnings(expr))
}

extract_htest <- function(result, category, test, h0, issue_text, no_issue_text) {
  if (is.null(result)) {
    return(test_row(category, test, h0, NA_real_, NA_real_, issue_text, no_issue_text))
  }
  statistic <- suppressWarnings(as.numeric(result$statistic[[1]]))
  p_value <- suppressWarnings(as.numeric(result$p.value))
  test_row(category, test, h0, statistic, p_value, issue_text, no_issue_text)
}

white_test <- function(model) {
  aux <- data.frame(resid2 = stats::residuals(model)^2, fitted_value = stats::fitted(model))
  aux$fitted_value2 <- aux$fitted_value^2
  aux_model <- stats::lm(resid2 ~ fitted_value + fitted_value2, data = aux)
  statistic <- stats::nobs(aux_model) * summary(aux_model)$r.squared
  df <- 2
  p_value <- stats::pchisq(statistic, df = df, lower.tail = FALSE)
  list(statistic = statistic, parameter = df, p.value = p_value)
}

model_tests <- function(model, simple = FALSE) {
  residual_values <- stats::residuals(model)
  rows <- list(
    extract_htest(
      safe_htest(lmtest::bptest(model)),
      "Heteroscedasticitate",
      "Breusch-Pagan",
      "H0: varianța reziduurilor este constantă",
      "Există dovezi privind heteroscedasticitatea reziduurilor.",
      "Testul nu oferă suficiente dovezi pentru existența heteroscedasticității."
    ),
    extract_htest(
      white_test(model),
      "Heteroscedasticitate",
      "White",
      "H0: varianța reziduurilor este constantă",
      "Există dovezi privind heteroscedasticitatea reziduurilor.",
      "Testul nu oferă suficiente dovezi pentru existența heteroscedasticității."
    ),
    extract_htest(
      safe_htest(lmtest::dwtest(model)),
      "Autocorelare",
      "Durbin-Watson",
      "H0: reziduurile nu sunt autocorelate",
      "Testul indică o posibilă dependență a reziduurilor.",
      "Testul nu oferă suficiente dovezi privind autocorelarea reziduurilor."
    ),
    extract_htest(
      safe_htest(lmtest::bgtest(model, order = 1)),
      "Autocorelare",
      "Breusch-Godfrey ordin 1",
      "H0: reziduurile nu prezintă autocorelare de ordin 1",
      "Testul indică o posibilă autocorelare de ordin 1.",
      "Testul nu oferă suficiente dovezi privind autocorelarea de ordin 1."
    ),
    extract_htest(
      safe_htest(lmtest::bgtest(model, order = 2)),
      "Autocorelare",
      "Breusch-Godfrey ordin 2",
      "H0: reziduurile nu prezintă autocorelare de ordin 2",
      "Testul indică o posibilă autocorelare de ordin 2.",
      "Testul nu oferă suficiente dovezi privind autocorelarea de ordin 2."
    ),
    extract_htest(
      safe_htest(lmtest::bgtest(model, order = 3)),
      "Autocorelare",
      "Breusch-Godfrey ordin 3",
      "H0: reziduurile nu prezintă autocorelare de ordin 3",
      "Testul indică o posibilă autocorelare de ordin 3.",
      "Testul nu oferă suficiente dovezi privind autocorelarea de ordin 3."
    ),
    extract_htest(
      safe_htest(stats::shapiro.test(residual_values)),
      "Normalitate",
      "Shapiro-Wilk",
      "H0: reziduurile urmează o distribuție normală",
      "Testul indică abatere de la normalitatea reziduurilor.",
      "Testul nu oferă suficiente dovezi privind abaterea de la normalitate."
    ),
    extract_htest(
      safe_htest(tseries::jarque.bera.test(residual_values)),
      "Normalitate",
      "Jarque-Bera",
      "H0: reziduurile urmează o distribuție normală",
      "Testul indică abatere de la normalitatea reziduurilor.",
      "Testul nu oferă suficiente dovezi privind abaterea de la normalitate."
    ),
    extract_htest(
      safe_htest(nortest::lillie.test(residual_values)),
      "Normalitate",
      "Lilliefors / Kolmogorov-Smirnov",
      "H0: reziduurile urmează o distribuție normală",
      "Testul indică abatere de la normalitatea reziduurilor.",
      "Testul nu oferă suficiente dovezi privind abaterea de la normalitate."
    ),
    extract_htest(
      safe_htest(nortest::cvm.test(residual_values)),
      "Normalitate",
      "Cramer-von Mises",
      "H0: reziduurile urmează o distribuție normală",
      "Testul indică abatere de la normalitatea reziduurilor.",
      "Testul nu oferă suficiente dovezi privind abaterea de la normalitate."
    ),
    extract_htest(
      safe_htest(nortest::ad.test(residual_values)),
      "Normalitate",
      "Anderson-Darling",
      "H0: reziduurile urmează o distribuție normală",
      "Testul indică abatere de la normalitatea reziduurilor.",
      "Testul nu oferă suficiente dovezi privind abaterea de la normalitate."
    ),
    extract_htest(
      safe_htest(lmtest::resettest(model, power = 2:3, type = "fitted")),
      "Specificare",
      "Ramsey RESET",
      "H0: modelul nu prezintă erori funcționale detectabile prin termenii testați",
      "Testul indică o posibilă problemă de specificare.",
      "Testul nu oferă suficiente dovezi privind o eroare de specificare."
    )
  )
  shape_rows <- tibble::tibble(
    Categorie = "Forma reziduurilor",
    Test = c("Skewness", "Kurtosis"),
    H0 = c("Indicator descriptiv", "Indicator descriptiv"),
    Statistică = c(moments::skewness(residual_values), moments::kurtosis(residual_values)),
    `p-value` = c(NA_real_, NA_real_),
    Decizie = c("Indicator descriptiv", "Indicator descriptiv"),
    Interpretare = c(
      "Skewness descrie asimetria distribuției reziduurilor.",
      "Kurtosis este raportată în convenția Pearson."
    )
  )
  dplyr::bind_rows(rows, shape_rows)
}

diagnostic_frame <- function(model, dataframe) {
  data.frame(
    country = dataframe$country,
    observed = dataframe$female_lfpr,
    fitted = stats::fitted(model),
    residual = stats::residuals(model),
    std_residual = stats::rstandard(model),
    studentized_residual = stats::rstudent(model),
    cooks_distance = stats::cooks.distance(model),
    leverage = stats::hatvalues(model)
  )
}

save_diagnostic_plots <- function(model, dataframe, prefix, predictor = NULL) {
  diag <- diagnostic_frame(model, dataframe)
  influential <- diag |>
    dplyr::mutate(flag = .data$cooks_distance > 4 / nrow(diag) | abs(.data$studentized_residual) > 2) |>
    dplyr::filter(.data$flag)

  save_plot(
    ggplot2::ggplot(diag, ggplot2::aes(.data$fitted, .data$residual)) +
      ggplot2::geom_hline(yintercept = 0, color = GOLD, linewidth = 0.8) +
      ggplot2::geom_point(color = TEAL, size = 2.6, alpha = 0.85) +
      ggplot2::geom_smooth(method = "loess", formula = y ~ x, se = FALSE, color = BURGUNDY, linewidth = 0.9) +
      ggplot2::labs(title = "Residuals vs Fitted", x = "Valori estimate", y = "Reziduuri") +
      theme_project(),
    paste0(prefix, "_residuals_fitted.png")
  )

  save_plot(
    ggplot2::ggplot(diag, ggplot2::aes(sample = .data$std_residual)) +
      ggplot2::stat_qq(color = TEAL, size = 2.4, alpha = 0.85) +
      ggplot2::stat_qq_line(color = BURGUNDY, linewidth = 0.9) +
      ggplot2::labs(title = "Q-Q Plot", x = "Cuantile teoretice", y = "Cuantile eșantion") +
      theme_project(),
    paste0(prefix, "_qq_plot.png")
  )

  save_plot(
    ggplot2::ggplot(diag, ggplot2::aes(.data$residual)) +
      ggplot2::geom_histogram(ggplot2::aes(y = ggplot2::after_stat(density)), bins = 14, fill = TEAL, color = "white", alpha = 0.72) +
      ggplot2::geom_density(color = BURGUNDY, linewidth = 1) +
      ggplot2::labs(title = "Distribuția reziduurilor", x = "Reziduuri", y = "Densitate") +
      theme_project(),
    paste0(prefix, "_residual_histogram.png")
  )

  save_plot(
    ggplot2::ggplot(diag, ggplot2::aes(x = "", y = .data$residual)) +
      ggplot2::geom_boxplot(fill = GOLD, color = NAVY, alpha = 0.55, width = 0.35) +
      ggplot2::geom_jitter(color = TEAL, width = 0.08, alpha = 0.75) +
      ggplot2::labs(title = "Box plot reziduuri", x = "", y = "Reziduuri") +
      theme_project(),
    paste0(prefix, "_residual_boxplot.png"),
    width = 7,
    height = 5
  )

  if (!is.null(predictor) && predictor %in% names(dataframe)) {
    plot_data <- dplyr::bind_cols(diag, dataframe[predictor])
    names(plot_data)[ncol(plot_data)] <- "predictor_value"
    save_plot(
      ggplot2::ggplot(plot_data, ggplot2::aes(.data$predictor_value, .data$residual)) +
        ggplot2::geom_hline(yintercept = 0, color = GOLD, linewidth = 0.8) +
        ggplot2::geom_point(color = TEAL, size = 2.6, alpha = 0.85) +
        ggplot2::geom_smooth(method = "loess", formula = y ~ x, se = FALSE, color = BURGUNDY, linewidth = 0.9) +
        ggplot2::labs(title = paste("Reziduuri vs", label_for(predictor)), x = axis_for(predictor), y = "Reziduuri") +
        theme_project(),
      paste0(prefix, "_residuals_predictor.png")
    )
  }

  save_plot(
    ggplot2::ggplot(diag, ggplot2::aes(seq_along(.data$cooks_distance), .data$cooks_distance)) +
      ggplot2::geom_col(fill = TEAL, alpha = 0.78) +
      ggplot2::geom_hline(yintercept = 4 / nrow(diag), color = BURGUNDY, linewidth = 0.8, linetype = "dashed") +
      ggplot2::geom_text(data = influential, ggplot2::aes(label = .data$country), vjust = -0.25, size = 3, color = NAVY, check_overlap = TRUE) +
      ggplot2::labs(title = "Cook's Distance", x = "Observații", y = "Cook's distance") +
      theme_project(),
    paste0(prefix, "_cooks_distance.png")
  )

  save_plot(
    ggplot2::ggplot(diag, ggplot2::aes(.data$fitted, sqrt(abs(.data$std_residual)))) +
      ggplot2::geom_point(color = TEAL, size = 2.6, alpha = 0.85) +
      ggplot2::geom_smooth(method = "loess", formula = y ~ x, se = FALSE, color = BURGUNDY, linewidth = 0.9) +
      ggplot2::labs(title = "Scale-Location", x = "Valori estimate", y = "sqrt(|reziduuri standardizate|)") +
      theme_project(),
    paste0(prefix, "_scale_location.png")
  )

  save_plot(
    ggplot2::ggplot(diag, ggplot2::aes(.data$leverage, .data$studentized_residual)) +
      ggplot2::geom_hline(yintercept = c(-2, 2), color = GOLD, linetype = "dashed") +
      ggplot2::geom_point(color = TEAL, size = 2.6, alpha = 0.85) +
      ggplot2::geom_text(data = influential, ggplot2::aes(label = .data$country), vjust = -0.45, size = 3, color = NAVY, check_overlap = TRUE) +
      ggplot2::labs(title = "Leverage vs reziduuri studentizate", x = "Leverage", y = "Reziduuri studentizate") +
      theme_project(),
    paste0(prefix, "_leverage_studentized.png")
  )

  save_plot(
    ggplot2::ggplot(diag, ggplot2::aes(.data$fitted, .data$observed)) +
      ggplot2::geom_abline(slope = 1, intercept = 0, color = GOLD, linewidth = 0.8) +
      ggplot2::geom_point(color = TEAL, size = 2.6, alpha = 0.85) +
      ggplot2::labs(title = "Valori observate vs estimate", x = "Valori estimate", y = "Valori observate") +
      theme_project(),
    paste0(prefix, "_observed_fitted.png")
  )
}

simple_required <- c("female_lfpr", SIMPLE_X)
simple_data <- complete_model_data(data_2023, simple_required)
simple_formula <- stats::as.formula(paste("female_lfpr ~", SIMPLE_X))
simple_model <- fit_lm_checked(simple_formula, simple_data, predictor_count = 1)

if (is.null(simple_model)) {
  stop("Modelul de regresie simplă nu poate fi estimat cu eșantionul disponibil.", call. = FALSE)
}

simple_coefficients <- tidy_coefficients(simple_model)
simple_metrics <- model_metrics(simple_model, nrow(simple_data), predictor_count = 1)
simple_tests <- model_tests(simple_model, simple = TRUE) |>
  dplyr::select("Test", `Ipoteza nulă` = "H0", "Statistică", "p-value", Concluzie = "Interpretare")

simple_beta <- stats::coef(simple_model)[[SIMPLE_X]]
simple_p <- simple_coefficients |>
  dplyr::filter(.data$Termen == label_for(SIMPLE_X)) |>
  dplyr::pull(.data$`p-value`)
simple_conclusion <- if (length(simple_p) == 1 && !is.na(simple_p) && simple_p < 0.05 && simple_beta > 0) {
  "Inflația prezintă o asociere pozitivă și semnificativă statistic cu rata participării femeilor în secțiunea transversală analizată."
} else if (length(simple_p) == 1 && !is.na(simple_p) && simple_p < 0.05 && simple_beta < 0) {
  "Inflația prezintă o asociere negativă și semnificativă statistic cu rata participării femeilor în secțiunea transversală analizată."
} else {
  "Nu există suficiente dovezi statistice pentru identificarea unei relații liniare semnificative între inflație și participarea femeilor în eșantionul analizat."
}
simple_conclusion <- paste(simple_conclusion, "Regresia transversală identifică asocieri statistice și nu demonstrează automat efecte cauzale.")

simple_r <- stats::cor(simple_data[[SIMPLE_X]], simple_data$female_lfpr, use = "complete.obs")
simple_equation <- paste0(
  "y = ", format_number(stats::coef(simple_model)[["(Intercept)"]], 2),
  ifelse(simple_beta >= 0, " + ", " - "),
  format_number(abs(simple_beta), 2), "x"
)
simple_r2 <- summary(simple_model)$r.squared

save_plot(
  ggplot2::ggplot(simple_data, ggplot2::aes(x = .data[[SIMPLE_X]], y = .data$female_lfpr)) +
    ggplot2::geom_point(color = TEAL, size = 3, alpha = 0.85) +
    ggplot2::geom_smooth(method = "lm", formula = y ~ x, se = TRUE, color = BURGUNDY, fill = GOLD, alpha = 0.18, linewidth = 1) +
    ggplot2::annotate(
      "label",
      x = Inf,
      y = -Inf,
      hjust = 1.02,
      vjust = -0.35,
      label = paste0("r Pearson = ", format_number(simple_r, 3), "\n", simple_equation, "\nR² = ", format_number(simple_r2, 3)),
      color = NAVY,
      fill = IVORY,
      linewidth = 0.2
    ) +
    ggplot2::labs(
      title = "Participarea femeilor și inflația",
      subtitle = "Economii europene, 2023",
      x = axis_for(SIMPLE_X),
      y = "Participarea femeilor la forța de muncă (%)"
    ) +
    theme_project(),
  "simple_scatter_regression.png",
  width = 10.5,
  height = 6
)

save_diagnostic_plots(simple_model, simple_data, "simple", predictor = SIMPLE_X)

simple_quantiles <- stats::quantile(simple_data[[SIMPLE_X]], probs = c(0.10, 0.25, 0.50, 0.75, 0.90), na.rm = TRUE, names = TRUE)
simple_newdata <- data.frame(value = as.numeric(simple_quantiles))
names(simple_newdata) <- SIMPLE_X
simple_conf <- stats::predict(simple_model, newdata = simple_newdata, interval = "confidence", level = 0.90)
simple_pred <- stats::predict(simple_model, newdata = simple_newdata, interval = "prediction", level = 0.90)
simple_predictions <- tibble::tibble(
  Nivel = names(simple_quantiles),
  Predictor = label_for(SIMPLE_X),
  Valoare = as.numeric(simple_quantiles),
  Predicție = simple_conf[, "fit"],
  `IC 90% inferior` = simple_conf[, "lwr"],
  `IC 90% superior` = simple_conf[, "upr"],
  `IP 90% inferior` = simple_pred[, "lwr"],
  `IP 90% superior` = simple_pred[, "upr"]
)

write_csv_output(simple_coefficients, "simple_coefficients.csv")
write_csv_output(simple_metrics, "simple_model_metrics.csv")
write_csv_output(simple_tests, "simple_tests.csv")
write_csv_output(simple_predictions, "simple_predictions.csv")
saveRDS(simple_model, file.path(MODEL_DIR, "simple_model.rds"))

multiple_predictors <- c(
  "LogGDP", "gdp_growth", "inflation", "female_unemployment", "fertility",
  "urbanization", "female_services", "child_dependency", "female_vulnerable",
  "internet", "women_parliament", "female_tertiary", "control_corruption"
)
multiple_required <- c("female_lfpr", multiple_predictors)
multiple_data <- complete_model_data(data_2023, multiple_required)
multiple_formula <- stats::as.formula(paste("female_lfpr ~", paste(multiple_predictors, collapse = " + ")))
multiple_model <- fit_lm_checked(multiple_formula, multiple_data, predictor_count = length(multiple_predictors))

if (is.null(multiple_model)) {
  stop("Modelul de regresie multiplă nu poate fi estimat cu eșantionul disponibil.", call. = FALSE)
}

multiple_coefficients <- tidy_coefficients(multiple_model)
multiple_metrics <- model_metrics(multiple_model, nrow(multiple_data), length(multiple_predictors), include_information = TRUE)
multiple_tests <- model_tests(multiple_model, simple = FALSE)
multiple_tests$Interpretare[multiple_tests$Categorie == "Autocorelare"] <- paste(
  multiple_tests$Interpretare[multiple_tests$Categorie == "Autocorelare"],
  "Testele Durbin-Watson și Breusch-Godfrey sunt prezentate pentru consistență cu cadrul analitic de referință, însă interpretarea autocorelării seriale este mai relevantă pentru date ordonate temporal."
)

robust_hc3 <- tryCatch(
  lmtest::coeftest(multiple_model, vcov. = sandwich::vcovHC(multiple_model, type = "HC3")),
  error = function(e) NULL
)
if (is.null(robust_hc3)) {
  robust_hc3_coefficients <- tibble::tibble()
} else {
  robust_matrix <- as.matrix(robust_hc3)
  robust_hc3_coefficients <- tibble::tibble(
    Termen = vapply(rownames(robust_matrix), label_for, character(1)),
    Coeficient = robust_matrix[, 1],
    `Standard error HC3` = robust_matrix[, 2],
    `Statistică t` = robust_matrix[, 3],
    `p-value` = robust_matrix[, 4]
  )
}

vif_values <- tryCatch(car::vif(multiple_model), error = function(e) NULL)
if (is.null(vif_values)) {
  multiple_vif <- tibble::tibble(Predictor = character(), VIF = numeric(), Interpretare = character())
} else {
  if (is.matrix(vif_values)) {
    vif_numeric <- vif_values[, 1]
  } else {
    vif_numeric <- vif_values
  }
  multiple_vif <- tibble::tibble(
    Predictor = vapply(names(vif_numeric), label_for, character(1)),
    VIF = as.numeric(vif_numeric),
    Interpretare = dplyr::case_when(
      VIF < 5 ~ "risc redus / moderat",
      VIF < 10 ~ "necesită atenție",
      TRUE ~ "multicoliniaritate severă potențială"
    )
  )
}

if (nrow(multiple_vif) > 0) {
  max_vif <- max(multiple_vif$VIF, na.rm = TRUE)
  multiple_tests <- dplyr::bind_rows(
    multiple_tests,
    tibble::tibble(
      Categorie = "Multicoliniaritate",
      Test = "VIF maxim",
      H0 = "Prag orientativ: VIF < 5",
      Statistică = max_vif,
      `p-value` = NA_real_,
      Decizie = dplyr::case_when(
        max_vif < 5 ~ "Nu apar dovezi clare ale problemei",
        max_vif < 10 ~ "Necesită atenție",
        TRUE ~ "Problemă potențială"
      ),
      Interpretare = dplyr::case_when(
        max_vif < 5 ~ "Valorile VIF indică risc redus sau moderat de multicoliniaritate.",
        max_vif < 10 ~ "Unele valori VIF necesită atenție în interpretarea coeficienților.",
        TRUE ~ "Valorile VIF indică multicoliniaritate severă potențială."
      )
    )
  )
}

multiple_residual_correlations <- lapply(multiple_predictors, function(predictor) {
  test <- tryCatch(stats::cor.test(stats::residuals(multiple_model), multiple_data[[predictor]], method = "pearson"), error = function(e) NULL)
  tibble::tibble(
    Predictor = label_for(predictor),
    r = if (is.null(test)) NA_real_ else unname(test$estimate),
    `p-value` = if (is.null(test)) NA_real_ else test$p.value
  )
}) |>
  dplyr::bind_rows()

multiple_influence <- diagnostic_frame(multiple_model, multiple_data) |>
  dplyr::mutate(
    Prag = .data$cooks_distance > 4 / nrow(multiple_data) | .data$leverage > 2 * (length(multiple_predictors) + 1) / nrow(multiple_data) | abs(.data$studentized_residual) > 2,
    Observație = dplyr::if_else(.data$Prag, "Influentă / necesită inspecție", "Fără semnal major")
  ) |>
  dplyr::arrange(dplyr::desc(.data$cooks_distance)) |>
  dplyr::transmute(
    Țară = .data$country,
    `Cook's distance` = .data$cooks_distance,
    Leverage = .data$leverage,
    `Reziduu studentizat` = .data$studentized_residual,
    Observație = .data$Observație
  )

coef_plot_data <- broom::tidy(multiple_model, conf.int = TRUE, conf.level = 0.95) |>
  dplyr::filter(.data$term != "(Intercept)") |>
  dplyr::mutate(Termen = vapply(.data$term, label_for, character(1)))
save_plot(
  ggplot2::ggplot(coef_plot_data, ggplot2::aes(x = .data$estimate, y = stats::reorder(.data$Termen, .data$estimate))) +
    ggplot2::geom_vline(xintercept = 0, color = GOLD, linewidth = 0.8) +
    ggplot2::geom_errorbar(ggplot2::aes(xmin = .data$conf.low, xmax = .data$conf.high), width = 0.18, color = BURGUNDY) +
    ggplot2::geom_point(color = TEAL, size = 2.8) +
    ggplot2::labs(title = "Coeficienții modelului multiplu", x = "Coeficient estimat", y = "") +
    theme_project(),
  "multiple_coefficient_plot.png",
  width = 10.5,
  height = 7
)

correlation_variables <- c("female_lfpr", multiple_predictors)
correlation_matrix <- stats::cor(multiple_data[, correlation_variables], use = "pairwise.complete.obs", method = "pearson")
colnames(correlation_matrix) <- vapply(colnames(correlation_matrix), label_for, character(1))
rownames(correlation_matrix) <- vapply(rownames(correlation_matrix), label_for, character(1))
correlation_long <- as.data.frame(as.table(correlation_matrix))
names(correlation_long) <- c("Variabila_1", "Variabila_2", "r")
save_plot(
  ggplot2::ggplot(correlation_long, ggplot2::aes(.data$Variabila_1, .data$Variabila_2, fill = .data$r)) +
    ggplot2::geom_tile(color = "white", linewidth = 0.25) +
    ggplot2::geom_text(ggplot2::aes(label = sprintf("%.2f", .data$r)), size = 2.8, color = INK) +
    ggplot2::scale_fill_gradient2(low = BURGUNDY, mid = IVORY, high = TEAL, midpoint = 0, limits = c(-1, 1), name = "r") +
    ggplot2::labs(title = "Matricea de corelație Pearson", x = "", y = "") +
    ggplot2::theme(axis.text.x = ggplot2::element_text(angle = 45, hjust = 1)) +
    theme_project(base_size = 10),
  "multiple_correlation_heatmap.png",
  width = 11.5,
  height = 9
)

if (nrow(multiple_vif) > 0) {
  save_plot(
    ggplot2::ggplot(multiple_vif, ggplot2::aes(x = stats::reorder(.data$Predictor, .data$VIF), y = .data$VIF, fill = .data$Interpretare)) +
      ggplot2::geom_col(alpha = 0.88) +
      ggplot2::coord_flip() +
      ggplot2::scale_fill_manual(values = c("risc redus / moderat" = TEAL, "necesită atenție" = GOLD, "multicoliniaritate severă potențială" = BURGUNDY)) +
      ggplot2::labs(title = "VIF pentru predictorii modelului multiplu", x = "", y = "VIF") +
      theme_project(),
    "multiple_vif_bar.png",
    width = 9.5,
    height = 6
  )
}

save_diagnostic_plots(multiple_model, multiple_data, "multiple")

write_csv_output(multiple_coefficients, "multiple_coefficients.csv")
write_csv_output(multiple_metrics, "multiple_model_metrics.csv")
write_csv_output(multiple_tests, "multiple_tests.csv")
write_csv_output(multiple_vif, "multiple_vif.csv")
write_csv_output(multiple_residual_correlations, "multiple_residual_correlations.csv")
write_csv_output(multiple_influence, "multiple_influence.csv")
write_csv_output(robust_hc3_coefficients, "robust_hc3_coefficients.csv")
saveRDS(multiple_model, file.path(MODEL_DIR, "multiple_model.rds"))

data_with_dummy <- data_2023 |>
  dplyr::mutate(DigitalizareRidicata = dplyr::if_else(.data$internet >= stats::median(.data$internet, na.rm = TRUE), 1, 0))
digital_threshold <- stats::median(data_2023$internet, na.rm = TRUE)
dummy_counts <- data_with_dummy |>
  dplyr::filter(!is.na(.data$DigitalizareRidicata)) |>
  dplyr::count(.data$DigitalizareRidicata, name = "Număr țări") |>
  dplyr::mutate(Categorie = dplyr::if_else(.data$DigitalizareRidicata == 1, "Digitalizare ridicată", "Digitalizare redusă")) |>
  dplyr::select("Categorie", "Număr țări")
write_csv_output(dummy_counts, "dummy_counts.csv")

dummy_predictors <- c("LogGDP", "gdp_growth", "inflation", "female_unemployment", "fertility", "female_tertiary", "internet", "DigitalizareRidicata", "control_corruption")
dummy_data <- complete_model_data(data_with_dummy, c("female_lfpr", dummy_predictors))
dummy_model <- fit_lm_checked(
  stats::as.formula(paste("female_lfpr ~", paste(dummy_predictors, collapse = " + "))),
  dummy_data,
  predictor_count = length(dummy_predictors)
)
if (!is.null(dummy_model)) {
  write_csv_output(tidy_coefficients(dummy_model), "dummy_coefficients.csv")
  write_csv_output(model_tests(dummy_model) |>
    dplyr::select("Categorie", "Test", "H0", "Statistică", "p-value", "Decizie", "Interpretare"), "dummy_tests.csv")
  saveRDS(dummy_model, file.path(MODEL_DIR, "dummy_model.rds"))
} else {
  write_csv_output(tibble::tibble(), "dummy_coefficients.csv")
  write_csv_output(tibble::tibble(), "dummy_tests.csv")
}

interaction1_formula <- female_lfpr ~ LogGDP + gdp_growth + inflation + female_unemployment + fertility + internet * female_tertiary + control_corruption
interaction1_vars <- c("female_lfpr", "LogGDP", "gdp_growth", "inflation", "female_unemployment", "fertility", "internet", "female_tertiary", "control_corruption")
interaction1_data <- complete_model_data(data_2023, interaction1_vars)
interaction1_model <- fit_lm_checked(interaction1_formula, interaction1_data, predictor_count = 8)

interaction2_formula <- female_lfpr ~ LogGDP * control_corruption + female_unemployment + fertility + internet + female_tertiary
interaction2_vars <- c("female_lfpr", "LogGDP", "control_corruption", "female_unemployment", "fertility", "internet", "female_tertiary")
interaction2_data <- complete_model_data(data_2023, interaction2_vars)
interaction2_model <- fit_lm_checked(interaction2_formula, interaction2_data, predictor_count = 6)

if (!is.null(interaction1_model)) {
  write_csv_output(tidy_coefficients(interaction1_model), "interaction1_coefficients.csv")
  saveRDS(interaction1_model, file.path(MODEL_DIR, "interaction1_model.rds"))

  levels_tertiary <- stats::quantile(interaction1_data$female_tertiary, probs = c(0.25, 0.50, 0.75), na.rm = TRUE)
  internet_seq <- seq(min(interaction1_data$internet, na.rm = TRUE), max(interaction1_data$internet, na.rm = TRUE), length.out = 80)
  base_values <- lapply(interaction1_vars[-1], function(variable) stats::median(interaction1_data[[variable]], na.rm = TRUE))
  names(base_values) <- interaction1_vars[-1]
  grid1 <- dplyr::bind_rows(lapply(names(levels_tertiary), function(level_name) {
    frame <- as.data.frame(base_values)
    frame <- frame[rep(1, length(internet_seq)), ]
    frame$internet <- internet_seq
    frame$female_tertiary <- as.numeric(levels_tertiary[[level_name]])
    frame$Nivel <- paste0("Educație terțiară ", level_name)
    frame
  }))
  grid1$fit <- stats::predict(interaction1_model, newdata = grid1)
  save_plot(
    ggplot2::ggplot(grid1, ggplot2::aes(.data$internet, .data$fit, color = .data$Nivel)) +
      ggplot2::geom_line(linewidth = 1.1) +
      ggplot2::scale_color_manual(values = c(TEAL, GOLD, BURGUNDY)) +
      ggplot2::labs(
        title = "Interacțiunea Internet × Educație terțiară",
        subtitle = "Predicții pe niveluri P25, P50 și P75 ale educației terțiare feminine",
        x = axis_for("internet"),
        y = "Participarea estimată a femeilor (%)"
      ) +
      theme_project(),
    "interaction1_plot.png",
    width = 10,
    height = 6
  )
} else {
  write_csv_output(tibble::tibble(), "interaction1_coefficients.csv")
}

if (!is.null(interaction2_model)) {
  write_csv_output(tidy_coefficients(interaction2_model), "interaction2_coefficients.csv")
  saveRDS(interaction2_model, file.path(MODEL_DIR, "interaction2_model.rds"))

  levels_cc <- stats::quantile(interaction2_data$control_corruption, probs = c(0.25, 0.50, 0.75), na.rm = TRUE)
  loggdp_seq <- seq(min(interaction2_data$LogGDP, na.rm = TRUE), max(interaction2_data$LogGDP, na.rm = TRUE), length.out = 80)
  base_values2 <- lapply(interaction2_vars[-1], function(variable) stats::median(interaction2_data[[variable]], na.rm = TRUE))
  names(base_values2) <- interaction2_vars[-1]
  grid2 <- dplyr::bind_rows(lapply(names(levels_cc), function(level_name) {
    frame <- as.data.frame(base_values2)
    frame <- frame[rep(1, length(loggdp_seq)), ]
    frame$LogGDP <- loggdp_seq
    frame$control_corruption <- as.numeric(levels_cc[[level_name]])
    frame$Nivel <- paste0("Controlul corupției ", level_name)
    frame
  }))
  grid2$fit <- stats::predict(interaction2_model, newdata = grid2)
  save_plot(
    ggplot2::ggplot(grid2, ggplot2::aes(.data$LogGDP, .data$fit, color = .data$Nivel)) +
      ggplot2::geom_line(linewidth = 1.1) +
      ggplot2::scale_color_manual(values = c(TEAL, GOLD, BURGUNDY)) +
      ggplot2::labs(
        title = "Interacțiunea PIB × Controlul corupției",
        subtitle = "Predicții pe niveluri P25, P50 și P75 ale controlului corupției",
        x = axis_for("LogGDP"),
        y = "Participarea estimată a femeilor (%)"
      ) +
      theme_project(),
    "interaction2_plot.png",
    width = 10,
    height = 6
  )
} else {
  write_csv_output(tibble::tibble(), "interaction2_coefficients.csv")
}

nonlinear_data <- multiple_data |>
  dplyr::mutate(
    CenteredLogGDP = .data$LogGDP - mean(.data$LogGDP, na.rm = TRUE),
    CenteredLogGDP2 = .data$CenteredLogGDP^2
  )
nonlinear_predictors <- c("CenteredLogGDP", "CenteredLogGDP2", "gdp_growth", "inflation", "female_unemployment", "fertility", "internet", "female_tertiary", "control_corruption")
nonlinear_model <- fit_lm_checked(
  stats::as.formula(paste("female_lfpr ~", paste(nonlinear_predictors, collapse = " + "))),
  nonlinear_data,
  predictor_count = length(nonlinear_predictors)
)
if (!is.null(nonlinear_model)) {
  write_csv_output(tidy_coefficients(nonlinear_model), "nonlinear_gdp_coefficients.csv")
  saveRDS(nonlinear_model, file.path(MODEL_DIR, "nonlinear_gdp_model.rds"))
} else {
  write_csv_output(tibble::tibble(), "nonlinear_gdp_coefficients.csv")
}

scenario_probs <- c(0.25, 0.50, 0.75)
scenario_names <- c("Niveluri reduse", "Niveluri mediane", "Niveluri ridicate")
scenario_rows <- lapply(seq_along(scenario_probs), function(index) {
  probability <- scenario_probs[[index]]
  values <- lapply(multiple_predictors, function(variable) stats::quantile(multiple_data[[variable]], probs = probability, na.rm = TRUE))
  names(values) <- multiple_predictors
  as.data.frame(values) |>
    dplyr::mutate(Scenariu = scenario_names[[index]], .before = 1)
})
scenario_newdata <- dplyr::bind_rows(scenario_rows)
scenario_conf <- stats::predict(multiple_model, newdata = scenario_newdata, interval = "confidence", level = 0.90)
scenario_pred <- stats::predict(multiple_model, newdata = scenario_newdata, interval = "prediction", level = 0.90)
scenario_predictions <- tibble::tibble(
  Scenariu = scenario_newdata$Scenariu,
  Predicție = scenario_conf[, "fit"],
  `IC 90% inferior` = scenario_conf[, "lwr"],
  `IC 90% superior` = scenario_conf[, "upr"],
  `IP 90% inferior` = scenario_pred[, "lwr"],
  `IP 90% superior` = scenario_pred[, "upr"]
)
save_plot(
  ggplot2::ggplot(scenario_predictions, ggplot2::aes(.data$Scenariu, .data$Predicție)) +
    ggplot2::geom_errorbar(ggplot2::aes(ymin = .data$`IP 90% inferior`, ymax = .data$`IP 90% superior`), width = 0.18, color = GOLD, linewidth = 1) +
    ggplot2::geom_errorbar(ggplot2::aes(ymin = .data$`IC 90% inferior`, ymax = .data$`IC 90% superior`), width = 0.1, color = BURGUNDY, linewidth = 1.2) +
    ggplot2::geom_point(color = TEAL, size = 3.2) +
    ggplot2::labs(title = "Predicții pe scenarii data-driven", x = "", y = "Participarea estimată a femeilor (%)") +
    theme_project(),
  "scenario_predictions.png",
  width = 8.5,
  height = 5.5
)
write_csv_output(scenario_predictions, "scenario_predictions.csv")

set.seed(123)
ml_data <- multiple_data
train_size <- floor(0.8 * nrow(ml_data))
train_index <- sort(sample(seq_len(nrow(ml_data)), size = train_size))
train_data <- ml_data[train_index, ]
test_data <- ml_data[-train_index, ]
train_x <- stats::model.matrix(multiple_formula, data = train_data)[, -1, drop = FALSE]
test_x <- stats::model.matrix(multiple_formula, data = test_data)[, -1, drop = FALSE]
train_y <- train_data$female_lfpr
test_y <- test_data$female_lfpr

ols_train <- stats::lm(multiple_formula, data = train_data)
ols_pred <- as.numeric(stats::predict(ols_train, newdata = test_data))
nfolds <- min(5, max(3, floor(nrow(train_data) / 3)))
ridge_cv <- glmnet::cv.glmnet(train_x, train_y, alpha = 0, nfolds = nfolds, standardize = TRUE)
lasso_cv <- glmnet::cv.glmnet(train_x, train_y, alpha = 1, nfolds = nfolds, standardize = TRUE)
ridge_pred <- as.numeric(stats::predict(ridge_cv, newx = test_x, s = "lambda.min"))
lasso_pred <- as.numeric(stats::predict(lasso_cv, newx = test_x, s = "lambda.min"))

metric_row <- function(model_name, actual, predicted) {
  rss <- sum((actual - predicted)^2)
  tss <- sum((actual - mean(actual))^2)
  tibble::tibble(
    Model = model_name,
    RMSE = sqrt(mean((actual - predicted)^2)),
    MAE = mean(abs(actual - predicted)),
    MAPE = if (all(abs(actual) > 1e-8)) mean(abs((actual - predicted) / actual)) * 100 else NA_real_,
    `R² test` = if (tss > 0) 1 - rss / tss else NA_real_
  )
}
ml_metrics <- dplyr::bind_rows(
  metric_row("OLS", test_y, ols_pred),
  metric_row("Ridge", test_y, ridge_pred),
  metric_row("LASSO", test_y, lasso_pred)
)

lasso_coef <- as.matrix(stats::coef(lasso_cv, s = "lambda.min"))
lasso_nonzero <- data.frame(term = rownames(lasso_coef), coefficient = as.numeric(lasso_coef[, 1])) |>
  dplyr::filter(.data$term != "(Intercept)", abs(.data$coefficient) > 1e-10) |>
  dplyr::mutate(Termen = vapply(.data$term, label_for, character(1))) |>
  dplyr::transmute(Termen, Coeficient = .data$coefficient)

cv_plot_data <- function(cv_model) {
  tibble::tibble(
    log_lambda = log(cv_model$lambda),
    cvm = cv_model$cvm,
    cvsd = cv_model$cvsd
  )
}
save_cv_plot <- function(cv_model, title, filename) {
  plot_data <- cv_plot_data(cv_model)
  save_plot(
    ggplot2::ggplot(plot_data, ggplot2::aes(.data$log_lambda, .data$cvm)) +
      ggplot2::geom_ribbon(ggplot2::aes(ymin = .data$cvm - .data$cvsd, ymax = .data$cvm + .data$cvsd), fill = GOLD, alpha = 0.16) +
      ggplot2::geom_line(color = TEAL, linewidth = 1) +
      ggplot2::geom_vline(xintercept = log(cv_model$lambda.min), color = BURGUNDY, linetype = "dashed") +
      ggplot2::labs(title = title, x = "log(lambda)", y = "Eroare CV") +
      theme_project(),
    filename
  )
}
save_cv_plot(ridge_cv, "Ridge: eroare CV în funcție de log(lambda)", "ridge_cv_error.png")
save_cv_plot(lasso_cv, "LASSO: eroare CV în funcție de log(lambda)", "lasso_cv_error.png")

path_plot_data <- function(fit) {
  beta <- as.matrix(fit$beta)
  data.frame(
    Termen = rep(vapply(rownames(beta), label_for, character(1)), times = ncol(beta)),
    log_lambda = rep(log(fit$lambda), each = nrow(beta)),
    Coeficient = as.vector(beta)
  )
}
save_path_plot <- function(fit, title, filename) {
  plot_data <- path_plot_data(fit)
  save_plot(
    ggplot2::ggplot(plot_data, ggplot2::aes(.data$log_lambda, .data$Coeficient, color = .data$Termen)) +
      ggplot2::geom_line(linewidth = 0.85, alpha = 0.9) +
      ggplot2::labs(title = title, x = "log(lambda)", y = "Coeficient") +
      theme_project(base_size = 10) +
      ggplot2::theme(legend.position = "right"),
    filename,
    width = 11,
    height = 7
  )
}
save_path_plot(ridge_cv$glmnet.fit, "Ridge: traseele coeficienților", "ridge_coefficient_paths.png")
save_path_plot(lasso_cv$glmnet.fit, "LASSO: traseele coeficienților", "lasso_coefficient_paths.png")

save_predicted_actual <- function(actual, predicted, title, filename) {
  plot_data <- tibble::tibble(Observat = actual, Estimat = predicted)
  save_plot(
    ggplot2::ggplot(plot_data, ggplot2::aes(.data$Estimat, .data$Observat)) +
      ggplot2::geom_abline(slope = 1, intercept = 0, color = GOLD, linewidth = 0.9) +
      ggplot2::geom_point(color = TEAL, size = 3, alpha = 0.86) +
      ggplot2::labs(title = title, x = "Valori estimate", y = "Valori observate") +
      theme_project(),
    filename,
    width = 7.5,
    height = 5.2
  )
}
save_predicted_actual(test_y, ols_pred, "OLS: valori estimate vs observate", "ols_predicted_actual.png")
save_predicted_actual(test_y, ridge_pred, "Ridge: valori estimate vs observate", "ridge_predicted_actual.png")
save_predicted_actual(test_y, lasso_pred, "LASSO: valori estimate vs observate", "lasso_predicted_actual.png")

save_plot(
  ggplot2::ggplot(ml_metrics, ggplot2::aes(stats::reorder(.data$Model, .data$RMSE), .data$RMSE, fill = .data$Model)) +
    ggplot2::geom_col(alpha = 0.9) +
    ggplot2::scale_fill_manual(values = c("OLS" = NAVY, "Ridge" = TEAL, "LASSO" = BURGUNDY)) +
    ggplot2::labs(title = "Comparație RMSE", x = "", y = "RMSE") +
    theme_project(),
  "ml_rmse_comparison.png",
  width = 7.5,
  height = 5
)
save_plot(
  ggplot2::ggplot(ml_metrics, ggplot2::aes(stats::reorder(.data$Model, .data$MAE), .data$MAE, fill = .data$Model)) +
    ggplot2::geom_col(alpha = 0.9) +
    ggplot2::scale_fill_manual(values = c("OLS" = NAVY, "Ridge" = TEAL, "LASSO" = BURGUNDY)) +
    ggplot2::labs(title = "Comparație MAE", x = "", y = "MAE") +
    theme_project(),
  "ml_mae_comparison.png",
  width = 7.5,
  height = 5
)
if (nrow(lasso_nonzero) > 0) {
  save_plot(
    ggplot2::ggplot(lasso_nonzero, ggplot2::aes(stats::reorder(.data$Termen, .data$Coeficient), .data$Coeficient)) +
      ggplot2::geom_col(fill = TEAL, alpha = 0.86) +
      ggplot2::coord_flip() +
      ggplot2::labs(title = "Coeficienții LASSO nenuli", x = "", y = "Coeficient") +
      theme_project(),
    "lasso_nonzero_coefficients.png",
    width = 8.5,
    height = 5.8
  )
}

best_model <- ml_metrics |>
  dplyr::arrange(.data$RMSE) |>
  dplyr::slice(1) |>
  dplyr::pull(.data$Model)
ml_conclusion <- paste0(
  "În această partiționare train/test, modelul ", best_model,
  " a obținut cea mai redusă valoare RMSE. Rezultatul reflectă performanța predictivă pe setul de test și nu implică automat o superioritate cauzală sau structurală față de modelele econometrice."
)

write_csv_output(ml_metrics, "ml_metrics.csv")
write_csv_output(lasso_nonzero, "lasso_nonzero_coefficients.csv")
saveRDS(ols_train, file.path(MODEL_DIR, "ols_train_model.rds"))
saveRDS(ridge_cv, file.path(MODEL_DIR, "ridge_cv_model.rds"))
saveRDS(lasso_cv, file.path(MODEL_DIR, "lasso_cv_model.rds"))

metadata <- list(
  anul_analizat = ANALYSIS_YEAR,
  n_total = nrow(data_2023),
  n_simple = nrow(simple_data),
  n_multiple = nrow(multiple_data),
  train_n = nrow(train_data),
  test_n = nrow(test_data),
  data_analizata = DATA_PATH,
  timestamp_rulare = format(Sys.time(), "%Y-%m-%d %H:%M:%S %Z"),
  predictor_simplu = SIMPLE_X,
  predictor_simplu_label = label_for(SIMPLE_X),
  prag_digitalizare = digital_threshold,
  lambda_ridge = list(lambda_min = ridge_cv$lambda.min, lambda_1se = ridge_cv$lambda.1se),
  lambda_lasso = list(lambda_min = lasso_cv$lambda.min, lambda_1se = lasso_cv$lambda.1se),
  concluzii = list(
    model_simplu = simple_conclusion,
    ridge_lasso = ml_conclusion
  ),
  note = list(
    transversal = "Analiza transversală utilizează observațiile disponibile pentru economiile europene într-un singur moment temporal – anul 2023. Rezultatele descriu diferențele dintre țări și nu trebuie confundate cu efectele estimate din modelele panel.",
    cauzalitate = "Asocierile identificate prin regresie transversală nu demonstrează automat relații cauzale.",
    autocorelare = "Testele Durbin-Watson și Breusch-Godfrey sunt prezentate pentru consistență cu cadrul analitic de referință, însă interpretarea autocorelării seriale este mai relevantă pentru date ordonate temporal.",
    robust_hc3 = "Coeficienții cu erori standard robuste HC3 sunt prezentați ca verificare suplimentară atunci când heteroscedasticitatea poate afecta erorile standard convenționale.",
    influenta = "Observațiile influente sunt investigate, nu eliminate automat.",
    ridge_lasso = "Ridge și LASSO sunt utilizate ca instrumente predictive și de regularizare, nu ca substitut automat pentru interpretarea econometrică."
  )
)

jsonlite::write_json(
  metadata,
  path = file.path(META_DIR, "analysis_metadata.json"),
  pretty = TRUE,
  auto_unbox = TRUE,
  na = "null"
)

message("Analiza transversală R s-a încheiat cu succes.")

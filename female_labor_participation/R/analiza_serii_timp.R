#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  required_packages <- c(
    "readxl", "dplyr", "tidyr", "ggplot2", "forecast", "tseries",
    "urca", "lmtest", "moments", "nortest", "zoo", "broom",
    "jsonlite", "scales", "patchwork", "vars"
  )
  missing_packages <- required_packages[!vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)]
  if (length(missing_packages) > 0) {
    stop(
      paste0(
        "Pachete R lipsă: ", paste(missing_packages, collapse = ", "),
        ". Instalează-le manual o singură dată înainte de rularea analizei de serii de timp."
      ),
      call. = FALSE
    )
  }
  invisible(lapply(required_packages, require, character.only = TRUE))
})

START_YEAR <- 2001L
END_YEAR <- 2023L
FORECAST_END_YEAR <- 2026L

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
    script <- file.path(candidate, "R", "analiza_serii_timp.R")
    if (file.exists(dataset) && file.exists(script)) {
      return(normalizePath(candidate, winslash = "/", mustWork = TRUE))
    }
  }
  stop("Nu pot identifica rădăcina proiectului pentru analiza seriilor de timp.", call. = FALSE)
}

ROOT <- find_project_root()
DATA_PATH <- file.path(ROOT, "data", "P_Data_Extract_From_World_Development_Indicators.xlsx")
COUNTRY_CONFIG_PATH <- file.path(ROOT, "config", "europe_countries.csv")
OUTPUT_ROOT <- file.path(ROOT, "outputs", "time_series")
TABLE_DIR <- file.path(OUTPUT_ROOT, "tables")
FIGURE_DIR <- file.path(OUTPUT_ROOT, "figures")
MODEL_DIR <- file.path(OUTPUT_ROOT, "models")
META_DIR <- file.path(OUTPUT_ROOT, "metadata")

dir.create(TABLE_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create(FIGURE_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create(MODEL_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create(META_DIR, recursive = TRUE, showWarnings = FALSE)

warnings_list <- character()
add_warning <- function(message) {
  warnings_list <<- unique(c(warnings_list, message))
}

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

coerce_numeric <- function(x) suppressWarnings(as.numeric(x))

format_decision <- function(rejects_h0, reject_text, no_reject_text) {
  if (is.na(rejects_h0)) return("Testul nu a putut fi evaluat stabil.")
  if (rejects_h0) {
    paste("Respingem H0 la pragul de 5%.", reject_text)
  } else {
    paste("Nu respingem H0 la pragul de 5%.", no_reject_text)
  }
}

critical_value_5 <- function(cval) {
  if (is.null(cval)) return(NA_real_)
  if (is.matrix(cval) || is.data.frame(cval)) {
    columns <- colnames(cval)
    target <- which(columns %in% c("5pct", "5%", "5 pct"))
    if (length(target) == 0) target <- min(2, ncol(cval))
    return(suppressWarnings(as.numeric(cval[1, target[1]])))
  }
  values <- suppressWarnings(as.numeric(cval))
  names_values <- names(cval)
  if (!is.null(names_values)) {
    target <- which(names_values %in% c("5pct", "5%", "5 pct"))
    if (length(target) > 0) return(values[target[1]])
  }
  if (length(values) >= 2) return(values[2])
  if (length(values) == 1) return(values[1])
  NA_real_
}

select_adf_lag <- function(series, type = "drift") {
  max_lag <- min(2L, max(0L, floor((length(series) - 5L) / 2L)))
  candidates <- 0:max_lag
  scores <- vapply(
    candidates,
    function(lag_value) {
      fit <- tryCatch(urca::ur.df(series, type = type, lags = lag_value), error = function(e) NULL)
      if (is.null(fit)) return(Inf)
      tryCatch(stats::AIC(fit@testreg), error = function(e) Inf)
    },
    numeric(1)
  )
  candidates[which.min(scores)]
}

stationarity_tests <- function(series, stage_label) {
  rows <- list()

  add_test_row <- function(test, specification, h0, statistic, critical_5, rejects_h0, support_stationary) {
    conclusion <- if (test == "KPSS") {
      format_decision(
        rejects_h0,
        "Testul indică abatere de la staționaritatea specificată.",
        "Testul nu oferă dovezi suficiente împotriva staționarității specificate."
      )
    } else {
      format_decision(
        rejects_h0,
        "Testul oferă dovezi împotriva rădăcinii unitare.",
        "Testul nu oferă dovezi suficiente împotriva rădăcinii unitare."
      )
    }
    rows[[length(rows) + 1]] <<- tibble::tibble(
      Etapă = stage_label,
      Test = test,
      Specificație = specification,
      H0 = h0,
      Statistică = statistic,
      `Valoare critică 5%` = critical_5,
      Decizie = ifelse(is.na(rejects_h0), "Neconcludent", ifelse(rejects_h0, "Respingem H0", "Nu respingem H0")),
      Concluzie = conclusion,
      `Susține staționaritatea` = support_stationary
    )
  }

  adf_drift_lag <- select_adf_lag(series, "drift")
  adf_drift <- tryCatch(urca::ur.df(series, type = "drift", lags = adf_drift_lag), error = function(e) NULL)
  if (!is.null(adf_drift)) {
    statistic <- suppressWarnings(as.numeric(adf_drift@teststat[1]))
    critical <- critical_value_5(adf_drift@cval)
    reject <- ifelse(is.na(statistic) || is.na(critical), NA, statistic < critical)
    add_test_row("ADF", paste0("Intercept, lag ", adf_drift_lag), "Seria are rădăcină unitară și este nestaționară.", statistic, critical, reject, reject)
  }

  adf_trend_lag <- select_adf_lag(series, "trend")
  adf_trend <- tryCatch(urca::ur.df(series, type = "trend", lags = adf_trend_lag), error = function(e) NULL)
  if (!is.null(adf_trend)) {
    statistic <- suppressWarnings(as.numeric(adf_trend@teststat[1]))
    critical <- critical_value_5(adf_trend@cval)
    reject <- ifelse(is.na(statistic) || is.na(critical), NA, statistic < critical)
    add_test_row("ADF", paste0("Intercept și trend, lag ", adf_trend_lag), "Seria are rădăcină unitară și este nestaționară.", statistic, critical, reject, reject)
  }

  pp_constant <- tryCatch(urca::ur.pp(series, type = "Z-tau", model = "constant", lags = "short"), error = function(e) NULL)
  if (!is.null(pp_constant)) {
    statistic <- suppressWarnings(as.numeric(pp_constant@teststat[1]))
    critical <- critical_value_5(pp_constant@cval)
    reject <- ifelse(is.na(statistic) || is.na(critical), NA, statistic < critical)
    add_test_row("PP", "Intercept", "Seria are rădăcină unitară.", statistic, critical, reject, reject)
  }

  pp_trend <- tryCatch(urca::ur.pp(series, type = "Z-tau", model = "trend", lags = "short"), error = function(e) NULL)
  if (!is.null(pp_trend)) {
    statistic <- suppressWarnings(as.numeric(pp_trend@teststat[1]))
    critical <- critical_value_5(pp_trend@cval)
    reject <- ifelse(is.na(statistic) || is.na(critical), NA, statistic < critical)
    add_test_row("PP", "Intercept și trend", "Seria are rădăcină unitară.", statistic, critical, reject, reject)
  }

  kpss_mu <- tryCatch(urca::ur.kpss(series, type = "mu", lags = "short"), error = function(e) NULL)
  if (!is.null(kpss_mu)) {
    statistic <- suppressWarnings(as.numeric(kpss_mu@teststat[1]))
    critical <- critical_value_5(kpss_mu@cval)
    reject <- ifelse(is.na(statistic) || is.na(critical), NA, statistic > critical)
    add_test_row("KPSS", "Staționaritate în nivel", "Seria este staționară.", statistic, critical, reject, !reject)
  }

  kpss_tau <- tryCatch(urca::ur.kpss(series, type = "tau", lags = "short"), error = function(e) NULL)
  if (!is.null(kpss_tau)) {
    statistic <- suppressWarnings(as.numeric(kpss_tau@teststat[1]))
    critical <- critical_value_5(kpss_tau@cval)
    reject <- ifelse(is.na(statistic) || is.na(critical), NA, statistic > critical)
    add_test_row("KPSS", "Staționaritate în jurul trendului", "Seria este staționară în jurul trendului.", statistic, critical, reject, !reject)
  }

  dplyr::bind_rows(rows)
}

stationarity_vote <- function(test_table) {
  if (nrow(test_table) == 0) return(list(status = "neconcludentă", supports = 0L, total = 0L))
  votes <- test_table |>
    dplyr::filter(!is.na(.data[["Susține staționaritatea"]])) |>
    dplyr::group_by(.data$Test) |>
    dplyr::summarise(support = any(.data[["Susține staționaritatea"]]), .groups = "drop")
  supports <- sum(votes$support)
  total <- nrow(votes)
  status <- if (total == 0) {
    "neconcludentă"
  } else if (supports >= ceiling(total / 2)) {
    "compatibilă cu staționaritatea"
  } else {
    "compatibilă cu nestaționaritatea"
  }
  list(status = status, supports = supports, total = total)
}

acf_table <- function(series, type = "acf", lag_max = 8L) {
  if (type == "acf") {
    object <- stats::acf(series, lag.max = lag_max, plot = FALSE, na.action = na.pass)
  } else {
    object <- stats::pacf(series, lag.max = lag_max, plot = FALSE, na.action = na.pass)
  }
  tibble::tibble(
    Lag = as.numeric(object$lag),
    Valoare = as.numeric(object$acf)
  ) |>
    dplyr::filter(.data$Lag > 0)
}

correlogram_plot <- function(series, type, title) {
  lag_max <- min(8L, floor(length(series) / 3L))
  data <- acf_table(series, type = type, lag_max = lag_max)
  limit <- 1.96 / sqrt(length(series))
  ggplot2::ggplot(data, ggplot2::aes(x = .data$Lag, y = .data$Valoare)) +
    ggplot2::geom_hline(yintercept = 0, color = MUTED, linewidth = 0.35) +
    ggplot2::geom_hline(yintercept = c(-limit, limit), color = BURGUNDY, linetype = "dashed", linewidth = 0.45) +
    ggplot2::geom_col(fill = TEAL, width = 0.58) +
    ggplot2::scale_x_continuous(breaks = data$Lag) +
    ggplot2::labs(title = title, x = "Lag", y = toupper(type)) +
    theme_project()
}

safe_aicc <- function(model) {
  value <- tryCatch(model$aicc, error = function(e) NA_real_)
  if (is.null(value) || length(value) == 0) value <- tryCatch(model$model$aicc, error = function(e) NA_real_)
  suppressWarnings(as.numeric(value[1]))
}

safe_sse <- function(model) {
  value <- tryCatch(model$sse, error = function(e) NA_real_)
  if (is.null(value) || length(value) == 0) value <- tryCatch(model$model$sigma2 * length(stats::residuals(model)), error = function(e) NA_real_)
  suppressWarnings(as.numeric(value[1]))
}

fit_arima_safe <- function(series, order, include_drift = FALSE, include_mean = TRUE) {
  tryCatch(
    forecast::Arima(series, order = order, include.drift = include_drift, include.mean = include_mean, method = "ML"),
    error = function(e) NULL
  )
}

fit_forecast_model <- function(model_name, train_series, horizon, arima_order = c(0L, 1L, 0L)) {
  tryCatch(
    {
      if (model_name == "Naive") {
        return(forecast::naive(train_series, h = horizon))
      }
      if (model_name == "Drift") {
        return(forecast::rwf(train_series, h = horizon, drift = TRUE))
      }
      if (model_name == "SES") {
        return(forecast::ses(train_series, h = horizon))
      }
      if (model_name == "Holt") {
        return(forecast::holt(train_series, h = horizon, damped = FALSE))
      }
      if (model_name == "ETS") {
        ets_model <- forecast::ets(train_series, model = "ZZN")
        return(forecast::forecast(ets_model, h = horizon))
      }
      if (model_name == "ARIMA selectat") {
        arima_model <- forecast::Arima(
          train_series,
          order = arima_order,
          include.drift = arima_order[2] == 1,
          include.mean = arima_order[2] == 0,
          method = "ML"
        )
        return(forecast::forecast(arima_model, h = horizon))
      }
      NULL
    },
    error = function(e) NULL
  )
}

forecast_rows <- function(model_name, forecast_object, years, actual = rep(NA_real_, length(years))) {
  if (is.null(forecast_object)) return(tibble::tibble())
  lower <- as.matrix(forecast_object$lower)
  upper <- as.matrix(forecast_object$upper)
  lower80 <- if ("80%" %in% colnames(lower)) lower[, "80%"] else rep(NA_real_, length(years))
  upper80 <- if ("80%" %in% colnames(upper)) upper[, "80%"] else rep(NA_real_, length(years))
  lower95 <- if ("95%" %in% colnames(lower)) lower[, "95%"] else rep(NA_real_, length(years))
  upper95 <- if ("95%" %in% colnames(upper)) upper[, "95%"] else rep(NA_real_, length(years))
  tibble::tibble(
    Model = model_name,
    An = years,
    Real = as.numeric(actual),
    Prognoză = as.numeric(forecast_object$mean),
    `Lower 80%` = as.numeric(lower80),
    `Upper 80%` = as.numeric(upper80),
    `Lower 95%` = as.numeric(lower95),
    `Upper 95%` = as.numeric(upper95)
  )
}

forecast_accuracy <- function(model_name, actual, predicted, train_series, naive_predicted, aicc = NA_real_) {
  valid <- is.finite(actual) & is.finite(predicted)
  if (sum(valid) == 0) {
    return(tibble::tibble(
      Model = model_name, RMSE = NA_real_, MAE = NA_real_, MAPE = NA_real_,
      sMAPE = NA_real_, MASE = NA_real_, `Theil U2` = NA_real_, AICc = aicc
    ))
  }
  errors <- actual[valid] - predicted[valid]
  rmse <- sqrt(mean(errors^2))
  mae <- mean(abs(errors))
  mape <- if (all(abs(actual[valid]) > .Machine$double.eps)) mean(abs(errors / actual[valid])) * 100 else NA_real_
  smape <- mean(200 * abs(errors) / (abs(actual[valid]) + abs(predicted[valid])))
  scale <- mean(abs(diff(as.numeric(train_series))), na.rm = TRUE)
  mase <- ifelse(is.finite(scale) && scale > 0, mae / scale, NA_real_)
  naive_valid <- is.finite(actual) & is.finite(naive_predicted)
  naive_rmse <- if (sum(naive_valid) > 0) sqrt(mean((actual[naive_valid] - naive_predicted[naive_valid])^2)) else NA_real_
  theil <- ifelse(is.finite(naive_rmse) && naive_rmse > 0, rmse / naive_rmse, NA_real_)
  tibble::tibble(
    Model = model_name,
    RMSE = rmse,
    MAE = mae,
    MAPE = mape,
    sMAPE = smape,
    MASE = mase,
    `Theil U2` = theil,
    AICc = aicc
  )
}

extract_forecast_aicc <- function(forecast_object) {
  if (is.null(forecast_object)) return(NA_real_)
  value <- tryCatch(forecast_object$model$aicc, error = function(e) NA_real_)
  if (is.null(value) || length(value) == 0) value <- NA_real_
  suppressWarnings(as.numeric(value[1]))
}

if (!file.exists(DATA_PATH)) stop(paste0("Fișierul Excel nu există: ", DATA_PATH), call. = FALSE)
if (!file.exists(COUNTRY_CONFIG_PATH)) stop(paste0("Fișierul config/europe_countries.csv nu există: ", COUNTRY_CONFIG_PATH), call. = FALSE)

variables <- tibble::tribble(
  ~slug, ~code, ~label, ~short_label,
  "female_lfpr", "SL.TLF.ACTI.FE.ZS", "Rata participării femeilor la forța de muncă, 15–64 ani", "FLFP",
  "inflation", "FP.CPI.TOTL.ZG", "Inflația, prețurile de consum", "Inflație",
  "gdp_growth", "NY.GDP.MKTP.KD.ZG", "Creșterea PIB real", "Creștere PIB",
  "female_unemployment", "SL.UEM.TOTL.FE.ZS", "Șomajul femeilor", "Șomaj feminin"
)

raw_data <- suppressWarnings(readxl::read_excel(DATA_PATH, na = c("", "NA", "..")))
europe_countries <- read.csv(COUNTRY_CONFIG_PATH, stringsAsFactors = FALSE, fileEncoding = "UTF-8")
column_names <- names(raw_data)
mapped_columns <- stats::setNames(vapply(variables$code, function(code) find_column_by_code(column_names, code), character(1)), variables$slug)

analysis_data <- raw_data |>
  dplyr::transmute(
    Year = as.integer(coerce_numeric(.data[["Time"]])),
    country_original = as.character(.data[["Country Name"]]),
    ISO3 = as.character(.data[["Country Code"]]),
    female_lfpr = coerce_numeric(.data[[mapped_columns[["female_lfpr"]]]]),
    inflation = coerce_numeric(.data[[mapped_columns[["inflation"]]]]),
    gdp_growth = coerce_numeric(.data[[mapped_columns[["gdp_growth"]]]]),
    female_unemployment = coerce_numeric(.data[[mapped_columns[["female_unemployment"]]]])
  ) |>
  dplyr::filter(
    !is.na(.data$Year),
    .data$Year >= START_YEAR,
    .data$Year <= END_YEAR,
    .data$ISO3 %in% europe_countries$iso3
  ) |>
  dplyr::left_join(europe_countries, by = c("ISO3" = "iso3")) |>
  dplyr::mutate(country = dplyr::coalesce(.data$country, .data$country_original))

expected_years <- START_YEAR:END_YEAR
country_counts <- analysis_data |>
  dplyr::filter(!is.na(.data$female_lfpr)) |>
  dplyr::group_by(.data$Year) |>
  dplyr::summarise(n_countries = dplyr::n_distinct(.data$ISO3), .groups = "drop") |>
  tidyr::complete(Year = expected_years, fill = list(n_countries = 0L))

balanced_countries <- analysis_data |>
  dplyr::filter(!is.na(.data$female_lfpr)) |>
  dplyr::group_by(.data$ISO3) |>
  dplyr::summarise(n_years = dplyr::n_distinct(.data$Year), .groups = "drop") |>
  dplyr::filter(.data$n_years == length(expected_years)) |>
  dplyr::pull(.data$ISO3)

sample_change_ratio <- if (max(country_counts$n_countries) > 0) {
  (max(country_counts$n_countries) - min(country_counts$n_countries)) / max(country_counts$n_countries)
} else {
  NA_real_
}
sample_changes_significantly <- is.finite(sample_change_ratio) && sample_change_ratio > 0.10
balanced_sample_used <- sample_changes_significantly && length(balanced_countries) >= 10
if (sample_changes_significantly) {
  add_warning("Numărul de țări disponibile pentru seria principală variază între ani; a fost construită o serie alternativă pe eșantion echilibrat.")
}
if (balanced_sample_used) {
  add_warning("Analiza principală folosește eșantionul echilibrat pentru comparabilitate temporală.")
} else {
  add_warning("Analiza principală folosește eșantionul disponibil anual; variația acoperirii nu impune schimbarea eșantionului principal.")
}

available_stats <- analysis_data |>
  dplyr::filter(!is.na(.data$female_lfpr)) |>
  dplyr::group_by(.data$Year) |>
  dplyr::summarise(
    n_available = dplyr::n_distinct(.data$ISO3),
    available_mean = mean(.data$female_lfpr, na.rm = TRUE),
    available_sd = stats::sd(.data$female_lfpr, na.rm = TRUE),
    available_q1 = stats::quantile(.data$female_lfpr, 0.25, na.rm = TRUE, names = FALSE),
    available_median = stats::median(.data$female_lfpr, na.rm = TRUE),
    available_q3 = stats::quantile(.data$female_lfpr, 0.75, na.rm = TRUE, names = FALSE),
    .groups = "drop"
  )

balanced_stats <- analysis_data |>
  dplyr::filter(.data$ISO3 %in% balanced_countries, !is.na(.data$female_lfpr)) |>
  dplyr::group_by(.data$Year) |>
  dplyr::summarise(
    n_balanced = dplyr::n_distinct(.data$ISO3),
    balanced_mean = mean(.data$female_lfpr, na.rm = TRUE),
    balanced_sd = stats::sd(.data$female_lfpr, na.rm = TRUE),
    balanced_q1 = stats::quantile(.data$female_lfpr, 0.25, na.rm = TRUE, names = FALSE),
    balanced_median = stats::median(.data$female_lfpr, na.rm = TRUE),
    balanced_q3 = stats::quantile(.data$female_lfpr, 0.75, na.rm = TRUE, names = FALSE),
    .groups = "drop"
  )

series_table <- tibble::tibble(Year = expected_years) |>
  dplyr::left_join(available_stats, by = "Year") |>
  dplyr::left_join(balanced_stats, by = "Year") |>
  dplyr::mutate(
    female_lfpr_europe_mean = if (balanced_sample_used) .data$balanced_mean else .data$available_mean,
    n_countries = if (balanced_sample_used) .data$n_balanced else .data$n_available,
    sd_between_countries = if (balanced_sample_used) .data$balanced_sd else .data$available_sd,
    q1 = if (balanced_sample_used) .data$balanced_q1 else .data$available_q1,
    median = if (balanced_sample_used) .data$balanced_median else .data$available_median,
    q3 = if (balanced_sample_used) .data$balanced_q3 else .data$available_q3,
    sample_used = ifelse(balanced_sample_used, "Eșantion echilibrat", "Eșantion disponibil anual")
  )

series_model_data <- series_table |>
  dplyr::filter(!is.na(.data$female_lfpr_europe_mean)) |>
  dplyr::mutate(t_index = .data$Year - min(.data$Year) + 1L)

if (nrow(series_model_data) < 8) {
  stop("Seria anuală are prea puține observații valide pentru analiza solicitată.", call. = FALSE)
}

series_ts <- stats::ts(series_model_data$female_lfpr_europe_mean, start = min(series_model_data$Year), frequency = 1)
lag_max <- min(8L, floor(length(series_ts) / 3L))

write_csv_output(series_table, "europe_time_series.csv")

trend_model <- stats::lm(female_lfpr_europe_mean ~ t_index, data = series_model_data)
trend_summary <- summary(trend_model)
trend_beta <- unname(stats::coef(trend_model)[["t_index"]])
trend_p <- unname(stats::coef(trend_summary)[["t_index", "Pr(>|t|)"]])
trend_r2 <- unname(trend_summary$r.squared)
trend_direction <- ifelse(trend_beta > 0, "Trend crescător", ifelse(trend_beta < 0, "Trend descrescător", "Trend aproximativ plat"))
trend_table <- tibble::tibble(
  Model = "Y_t = alpha + beta * t + epsilon_t",
  Alpha = unname(stats::coef(trend_model)[["(Intercept)"]]),
  Beta = trend_beta,
  `p-value beta` = trend_p,
  `R²` = trend_r2,
  `Direcția trendului` = trend_direction,
  `Interpretare prudentă` = paste0(trend_direction, "; coeficientul anual estimat este ", round(trend_beta, 4), " puncte procentuale.")
)
write_csv_output(trend_table, "trend_model.csv")

diff_values <- diff(series_model_data$female_lfpr_europe_mean)
diff_years <- series_model_data$Year[-1]
diff_table <- tibble::tibble(
  Year = diff_years,
  Delta = as.numeric(diff_values)
)
shock_rows <- dplyr::bind_rows(
  diff_table |>
    dplyr::arrange(dplyr::desc(.data$Delta)) |>
    dplyr::slice_head(n = 3) |>
    dplyr::mutate(Tip = "Șoc pozitiv"),
  diff_table |>
    dplyr::arrange(.data$Delta) |>
    dplyr::slice_head(n = 3) |>
    dplyr::mutate(Tip = "Șoc negativ")
) |>
  dplyr::select("Tip", "Year", "Delta")
write_csv_output(shock_rows, "diff_shocks.csv")

largest_change <- diff_table |>
  dplyr::mutate(abs_delta = abs(.data$Delta)) |>
  dplyr::arrange(dplyr::desc(.data$abs_delta)) |>
  dplyr::slice_head(n = 1)
visual_conclusion <- paste0(
  "Seria are un ", tolower(trend_direction), ". Variația anuală maximă în valoare absolută apare în ",
  largest_change$Year[[1]], " și este de ", round(largest_change$Delta[[1]], 3), " puncte procentuale."
)
if (largest_change$Year[[1]] == 2020L) {
  visual_conclusion <- paste0(visual_conclusion, " Anul 2020 coincide cu pandemia COVID-19 și prezintă o modificare neobișnuită a seriei.")
}

event_data <- tibble::tibble(
  x = c(2008.5, 2020, 2022.5),
  label = c("Criza financiară", "Pandemia COVID-19", "Șoc inflaționist și energetic"),
  color = c(BURGUNDY, PLUM, GOLD)
)

period_mean <- mean(series_model_data$female_lfpr_europe_mean, na.rm = TRUE)
plot_series <- ggplot2::ggplot(series_model_data, ggplot2::aes(x = .data$Year, y = .data$female_lfpr_europe_mean)) +
  ggplot2::geom_ribbon(ggplot2::aes(ymin = .data$q1, ymax = .data$q3), fill = PLUM, alpha = 0.11, na.rm = TRUE) +
  ggplot2::geom_hline(yintercept = period_mean, color = GOLD, linewidth = 0.75, linetype = "longdash") +
  ggplot2::geom_line(color = TEAL, linewidth = 1.15) +
  ggplot2::geom_point(color = NAVY, fill = TEAL, shape = 21, size = 2.5, stroke = 0.8) +
  ggplot2::geom_smooth(method = "lm", se = FALSE, color = BURGUNDY, linewidth = 0.95) +
  ggplot2::geom_vline(data = event_data, ggplot2::aes(xintercept = .data$x), color = event_data$color, linetype = "dotted", linewidth = 0.55) +
  ggplot2::annotate("label", x = event_data$x, y = max(series_model_data$female_lfpr_europe_mean, na.rm = TRUE) + 0.45, label = event_data$label, size = 3, color = NAVY, fill = "white") +
  ggplot2::scale_x_continuous(breaks = seq(START_YEAR, END_YEAR, by = 2)) +
  ggplot2::labs(
    title = "Evoluția participării feminine în timp",
    subtitle = "Media neponderată a ratelor naționale; banda indică intervalul Q1–Q3 între țări.",
    x = "An",
    y = "Participare feminină (%)"
  ) +
  theme_project()
save_plot(plot_series, "flfp_europe_evolution.png", width = 11, height = 6.2)

series_summary <- tibble::tibble(
  Indicator = c("Ani", "Media perioadei", "Minim", "Maxim", "Trend anual estimat"),
  Valoare = c(
    length(series_ts),
    period_mean,
    min(series_model_data$female_lfpr_europe_mean, na.rm = TRUE),
    max(series_model_data$female_lfpr_europe_mean, na.rm = TRUE),
    trend_beta
  ),
  Detaliu = c(
    paste0(min(series_model_data$Year), "–", max(series_model_data$Year)),
    "Media neponderată a ratelor naționale",
    "Valoarea minimă a seriei agregate",
    "Valoarea maximă a seriei agregate",
    trend_direction
  )
)
write_csv_output(series_summary, "series_summary.csv")

trend_df <- series_model_data |>
  dplyr::mutate(Trend = as.numeric(stats::fitted(trend_model)))
plot_trend <- ggplot2::ggplot(trend_df, ggplot2::aes(x = .data$Year)) +
  ggplot2::geom_line(ggplot2::aes(y = .data$female_lfpr_europe_mean, color = "Serie observată"), linewidth = 1.1) +
  ggplot2::geom_point(ggplot2::aes(y = .data$female_lfpr_europe_mean), color = TEAL, size = 2.2) +
  ggplot2::geom_line(ggplot2::aes(y = .data$Trend, color = "Trend liniar estimat"), linewidth = 1, linetype = "longdash") +
  ggplot2::scale_color_manual(values = c("Serie observată" = TEAL, "Trend liniar estimat" = BURGUNDY), name = NULL) +
  ggplot2::scale_x_continuous(breaks = seq(START_YEAR, END_YEAR, by = 2)) +
  ggplot2::labs(title = "Seria observată și trendul liniar estimat", x = "An", y = "Participare feminină (%)") +
  theme_project()
save_plot(plot_trend, "flfp_trend_linear.png", width = 10.5, height = 5.8)

save_plot(correlogram_plot(series_ts, "acf", "ACF pentru seria originală"), "acf_level.png", width = 7.2, height = 4.8)
save_plot(correlogram_plot(series_ts, "pacf", "PACF pentru seria originală"), "pacf_level.png", width = 7.2, height = 4.8)

stationarity_level <- stationarity_tests(as.numeric(series_ts), "Nivel")
write_csv_output(stationarity_level, "stationarity_level.csv")
level_vote <- stationarity_vote(stationarity_level)

diff_ts <- stats::ts(diff_values, start = min(diff_years), frequency = 1)
stationarity_diff1 <- stationarity_tests(as.numeric(diff_ts), "Diferența I")
write_csv_output(stationarity_diff1, "stationarity_diff1.csv")
diff_vote <- stationarity_vote(stationarity_diff1)

estimated_integration_order <- if (level_vote$status == "compatibilă cu staționaritatea") {
  "I(0)"
} else if (diff_vote$status == "compatibilă cu staționaritatea") {
  "I(1)"
} else {
  "Neclar după diferențierea de ordinul I"
}
if (estimated_integration_order != "I(0)") {
  add_warning("Testele de rădăcină unitară au putere redusă în eșantioane scurte; rezultatele ADF, PP și KPSS sunt analizate împreună cu structura ACF/PACF și comportamentul grafic al seriei.")
}

integration_summary <- tibble::tibble(
  Test = c("ADF", "PP", "KPSS", "Concluzie comună"),
  Nivel = c(
    paste0(sum(stationarity_level$Test == "ADF" & stationarity_level[["Susține staționaritatea"]], na.rm = TRUE), " specificații susțin staționaritatea"),
    paste0(sum(stationarity_level$Test == "PP" & stationarity_level[["Susține staționaritatea"]], na.rm = TRUE), " specificații susțin staționaritatea"),
    paste0(sum(stationarity_level$Test == "KPSS" & stationarity_level[["Susține staționaritatea"]], na.rm = TRUE), " specificații susțin staționaritatea"),
    level_vote$status
  ),
  `Diferența I` = c(
    paste0(sum(stationarity_diff1$Test == "ADF" & stationarity_diff1[["Susține staționaritatea"]], na.rm = TRUE), " specificații susțin staționaritatea"),
    paste0(sum(stationarity_diff1$Test == "PP" & stationarity_diff1[["Susține staționaritatea"]], na.rm = TRUE), " specificații susțin staționaritatea"),
    paste0(sum(stationarity_diff1$Test == "KPSS" & stationarity_diff1[["Susține staționaritatea"]], na.rm = TRUE), " specificații susțin staționaritatea"),
    diff_vote$status
  ),
  `Concluzie ordin de integrare` = c(rep(estimated_integration_order, 3), estimated_integration_order)
)
write_csv_output(integration_summary, "integration_summary.csv")

plot_diff <- ggplot2::ggplot(diff_table, ggplot2::aes(x = .data$Year, y = .data$Delta)) +
  ggplot2::geom_hline(yintercept = 0, color = GOLD, linewidth = 0.7) +
  ggplot2::geom_line(color = TEAL, linewidth = 1) +
  ggplot2::geom_point(color = NAVY, fill = TEAL, shape = 21, size = 2.4) +
  ggplot2::geom_point(data = shock_rows, ggplot2::aes(x = .data$Year, y = .data$Delta), color = BURGUNDY, fill = GOLD, shape = 21, size = 3, inherit.aes = FALSE) +
  ggplot2::labs(title = "Seria diferențiată de ordinul I", x = "An", y = "Delta participare feminină") +
  theme_project()
save_plot(plot_diff, "flfp_diff1.png", width = 10.5, height = 5.4)
save_plot(correlogram_plot(diff_ts, "acf", "ACF pentru seria diferențiată"), "acf_diff1.png", width = 7.2, height = 4.8)
save_plot(correlogram_plot(diff_ts, "pacf", "PACF pentru seria diferențiată"), "pacf_diff1.png", width = 7.2, height = 4.8)

d_order <- ifelse(estimated_integration_order == "I(0)", 0L, 1L)
candidate_orders <- tibble::tribble(
  ~p, ~q,
  0L, 0L,
  1L, 0L,
  0L, 1L,
  1L, 1L,
  2L, 0L,
  0L, 2L
)

arma_rows <- list()
arma_series <- if (d_order == 1L) diff_ts else series_ts
for (index in seq_len(nrow(candidate_orders))) {
  p <- candidate_orders$p[[index]]
  q <- candidate_orders$q[[index]]
  fit <- fit_arima_safe(arma_series, c(p, 0L, q), include_drift = FALSE, include_mean = TRUE)
  if (!is.null(fit)) {
    arma_rows[[length(arma_rows) + 1]] <- tibble::tibble(
      Model = paste0("ARMA(", p, ",", q, ")"),
      p = p,
      q = q,
      AIC = stats::AIC(fit),
      AICc = safe_aicc(fit),
      BIC = stats::BIC(fit),
      logLik = as.numeric(stats::logLik(fit)),
      sigma2 = fit$sigma2
    )
  }
}
arma_candidates <- dplyr::bind_rows(arma_rows) |>
  dplyr::arrange(.data$AICc)
write_csv_output(arma_candidates, "arma_candidates.csv")

arima_rows <- list()
arima_fits <- list()
for (index in seq_len(nrow(candidate_orders))) {
  p <- candidate_orders$p[[index]]
  q <- candidate_orders$q[[index]]
  fit <- fit_arima_safe(series_ts, c(p, d_order, q), include_drift = d_order == 1L, include_mean = d_order == 0L)
  if (!is.null(fit)) {
    key <- paste0(p, "_", d_order, "_", q)
    arima_fits[[key]] <- fit
    arima_rows[[length(arima_rows) + 1]] <- tibble::tibble(
      Model = paste0("ARIMA(", p, ",", d_order, ",", q, ")"),
      p = p,
      d = d_order,
      q = q,
      AIC = stats::AIC(fit),
      AICc = safe_aicc(fit),
      BIC = stats::BIC(fit),
      logLik = as.numeric(stats::logLik(fit)),
      sigma2 = fit$sigma2
    )
  }
}
arima_candidates <- dplyr::bind_rows(arima_rows) |>
  dplyr::arrange(.data$AICc)
if (nrow(arima_candidates) == 0) {
  stop("Niciun model ARIMA candidat nu a putut fi estimat stabil.", call. = FALSE)
}
write_csv_output(arima_candidates, "arima_candidates.csv")

selected_order <- as.integer(arima_candidates[1, c("p", "d", "q")])
selected_arima_name <- arima_candidates$Model[[1]]
selected_fit <- arima_fits[[paste0(selected_order[1], "_", selected_order[2], "_", selected_order[3])]]

auto_fit <- tryCatch(
  forecast::auto.arima(
    series_ts,
    seasonal = FALSE,
    max.p = 2,
    max.q = 2,
    max.d = max(1L, d_order),
    stepwise = FALSE,
    approximation = FALSE
  ),
  error = function(e) NULL
)
auto_arima_label <- if (is.null(auto_fit)) "Nedisponibil" else paste0("ARIMA", paste0("(", paste(forecast::arimaorder(auto_fit), collapse = ","), ")"))

coef_values <- stats::coef(selected_fit)
vcov_matrix <- tryCatch(stats::vcov(selected_fit), error = function(e) NULL)
se_values <- if (!is.null(vcov_matrix) && length(coef_values) > 0) sqrt(diag(vcov_matrix)) else rep(NA_real_, length(coef_values))
z_values <- coef_values / se_values
p_values <- 2 * stats::pnorm(abs(z_values), lower.tail = FALSE)
selected_coefficients <- tibble::tibble(
  Parametru = names(coef_values),
  Coeficient = as.numeric(coef_values),
  `Standard error` = as.numeric(se_values),
  Statistică = as.numeric(z_values),
  `p-value aproximativ` = as.numeric(p_values),
  AICc = safe_aicc(selected_fit),
  BIC = stats::BIC(selected_fit)
)
write_csv_output(selected_coefficients, "selected_arima_coefficients.csv")

residuals_arima <- as.numeric(stats::residuals(selected_fit))
residuals_arima <- residuals_arima[is.finite(residuals_arima)]
lb_lag <- min(6L, max(1L, floor(length(residuals_arima) / 3L)))
fit_df <- sum(selected_order[c(1, 3)])
if (lb_lag <= fit_df) lb_lag <- min(length(residuals_arima) - 1L, fit_df + 1L)
ljung <- tryCatch(stats::Box.test(residuals_arima, lag = lb_lag, type = "Ljung-Box", fitdf = fit_df), error = function(e) NULL)
jb <- tryCatch(tseries::jarque.bera.test(residuals_arima), error = function(e) NULL)
shapiro <- tryCatch(stats::shapiro.test(residuals_arima), error = function(e) NULL)
white_noise <- if (!is.null(ljung) && is.finite(ljung$p.value) && ljung$p.value >= 0.05 && abs(mean(residuals_arima, na.rm = TRUE)) < 0.25) {
  "Reziduurile sunt compatibile cu un comportament apropiat de zgomot alb, în limitele eșantionului scurt."
} else {
  "Diagnosticarea sugerează prudență: reziduurile pot păstra structură sau testele sunt instabile în eșantion scurt."
}
arima_diagnostics <- tibble::tibble(
  Indicator = c("Ljung-Box", "Jarque-Bera", "Shapiro-Wilk", "Skewness", "Kurtosis", "Media reziduurilor", "Concluzie zgomot alb"),
  Statistică = c(
    ifelse(is.null(ljung), NA_real_, as.numeric(ljung$statistic)),
    ifelse(is.null(jb), NA_real_, as.numeric(jb$statistic)),
    ifelse(is.null(shapiro), NA_real_, as.numeric(shapiro$statistic)),
    moments::skewness(residuals_arima),
    moments::kurtosis(residuals_arima),
    mean(residuals_arima, na.rm = TRUE),
    NA_real_
  ),
  `p-value` = c(
    ifelse(is.null(ljung), NA_real_, ljung$p.value),
    ifelse(is.null(jb), NA_real_, jb$p.value),
    ifelse(is.null(shapiro), NA_real_, shapiro$p.value),
    NA_real_, NA_real_, NA_real_, NA_real_
  ),
  Concluzie = c(
    ifelse(is.null(ljung), "Test indisponibil.", format_decision(ljung$p.value < 0.05, "Reziduurile prezintă autocorelare semnificativă.", "Nu există dovezi suficiente de autocorelare reziduală.")),
    "Normalitatea reziduurilor este un diagnostic complementar.",
    "Normalitatea reziduurilor este un diagnostic complementar.",
    "Asimetria reziduurilor.",
    "Aplatizarea reziduurilor.",
    "Media reziduurilor trebuie să fie apropiată de zero.",
    white_noise
  )
)
write_csv_output(arima_diagnostics, "arima_diagnostics.csv")

ic_plot_data <- arima_candidates |>
  dplyr::select("Model", "AICc", "BIC") |>
  tidyr::pivot_longer(cols = c("AICc", "BIC"), names_to = "Criteriu", values_to = "Valoare")
plot_ic <- ggplot2::ggplot(ic_plot_data, ggplot2::aes(x = stats::reorder(.data$Model, .data$Valoare), y = .data$Valoare, fill = .data$Criteriu)) +
  ggplot2::geom_col(position = "dodge", width = 0.68) +
  ggplot2::coord_flip() +
  ggplot2::scale_fill_manual(values = c("AICc" = TEAL, "BIC" = BURGUNDY)) +
  ggplot2::labs(title = "Comparația modelelor ARIMA candidate", x = NULL, y = "Valoare criteriu informațional") +
  theme_project()
save_plot(plot_ic, "arima_information_criteria.png", width = 9, height = 5.2)

residual_df <- tibble::tibble(
  Year = series_model_data$Year[seq_along(stats::residuals(selected_fit))],
  Reziduu = as.numeric(stats::residuals(selected_fit))
) |>
  dplyr::filter(is.finite(.data$Reziduu))
fitted_df <- tibble::tibble(
  Year = series_model_data$Year,
  Observat = as.numeric(series_ts),
  Fitted = as.numeric(stats::fitted(selected_fit))
)
plot_res_time <- ggplot2::ggplot(residual_df, ggplot2::aes(x = .data$Year, y = .data$Reziduu)) +
  ggplot2::geom_hline(yintercept = 0, color = GOLD, linewidth = 0.7) +
  ggplot2::geom_line(color = TEAL, linewidth = 0.9) +
  ggplot2::geom_point(color = NAVY, size = 2) +
  ggplot2::labs(title = "Reziduuri ARIMA în timp", x = "An", y = "Reziduu") +
  theme_project()
save_plot(plot_res_time, "arima_residuals_time.png", width = 8, height = 4.8)
save_plot(correlogram_plot(residuals_arima, "acf", "ACF reziduuri ARIMA"), "arima_residuals_acf.png", width = 7, height = 4.6)
plot_qq <- ggplot2::ggplot(residual_df, ggplot2::aes(sample = .data$Reziduu)) +
  ggplot2::stat_qq(color = TEAL, size = 2) +
  ggplot2::stat_qq_line(color = BURGUNDY, linewidth = 0.8) +
  ggplot2::labs(title = "Q-Q Plot reziduuri ARIMA", x = "Cuantile teoretice", y = "Cuantile observate") +
  theme_project()
save_plot(plot_qq, "arima_residuals_qq.png", width = 7, height = 4.8)
plot_hist <- ggplot2::ggplot(residual_df, ggplot2::aes(x = .data$Reziduu)) +
  ggplot2::geom_histogram(ggplot2::aes(y = after_stat(density)), bins = 8, fill = TEAL, alpha = 0.75, color = "white") +
  ggplot2::geom_density(color = BURGUNDY, linewidth = 0.9) +
  ggplot2::labs(title = "Histogramă și densitate pentru reziduuri", x = "Reziduu", y = "Densitate") +
  theme_project()
save_plot(plot_hist, "arima_residuals_histogram.png", width = 7, height = 4.8)
plot_obs_fit <- ggplot2::ggplot(fitted_df, ggplot2::aes(x = .data$Year)) +
  ggplot2::geom_line(ggplot2::aes(y = .data$Observat, color = "Observat"), linewidth = 1) +
  ggplot2::geom_line(ggplot2::aes(y = .data$Fitted, color = "Fitted"), linewidth = 1, linetype = "longdash") +
  ggplot2::scale_color_manual(values = c("Observat" = NAVY, "Fitted" = GOLD), name = NULL) +
  ggplot2::labs(title = "Observed vs Fitted ARIMA", x = "An", y = "Participare feminină (%)") +
  theme_project()
save_plot(plot_obs_fit, "arima_observed_fitted.png", width = 8.5, height = 4.8)
plot_box <- ggplot2::ggplot(residual_df, ggplot2::aes(x = "Reziduuri", y = .data$Reziduu)) +
  ggplot2::geom_boxplot(fill = PLUM, alpha = 0.22, color = BURGUNDY, width = 0.35) +
  ggplot2::geom_jitter(color = TEAL, width = 0.06, size = 2, alpha = 0.8) +
  ggplot2::labs(title = "Box plot reziduuri ARIMA", x = NULL, y = "Reziduu") +
  theme_project()
save_plot(plot_box, "arima_residuals_boxplot.png", width = 6.4, height = 4.8)

ses_fit <- forecast::ses(series_ts, h = 1)
holt_fit <- forecast::holt(series_ts, h = 1, damped = FALSE)
ets_model <- forecast::ets(series_ts, model = "ZZN")
ets_fit <- forecast::forecast(ets_model, h = 1)

smoothing_models <- tibble::tibble(
  Model = c("SES", "Holt", "ETS"),
  Alpha = c(ses_fit$model$par["alpha"], holt_fit$model$par["alpha"], ets_model$par["alpha"]),
  Beta = c(NA_real_, holt_fit$model$par["beta"], ifelse("beta" %in% names(ets_model$par), ets_model$par["beta"], NA_real_)),
  Phi = c(NA_real_, ifelse("phi" %in% names(holt_fit$model$par), holt_fit$model$par["phi"], NA_real_), ifelse("phi" %in% names(ets_model$par), ets_model$par["phi"], NA_real_)),
  AICc = c(safe_aicc(ses_fit), safe_aicc(holt_fit), safe_aicc(ets_model)),
  SSE = c(safe_sse(ses_fit), safe_sse(holt_fit), ets_model$mse * length(series_ts)),
  `Model ETS ales` = c(NA_character_, NA_character_, as.character(ets_model))
)
write_csv_output(smoothing_models, "smoothing_models.csv")

smoothing_df <- tibble::tibble(
  Year = series_model_data$Year,
  Real = as.numeric(series_ts),
  SES = as.numeric(stats::fitted(ses_fit)),
  Holt = as.numeric(stats::fitted(holt_fit)),
  ETS = as.numeric(stats::fitted(ets_model))
)
save_smoothing_plot <- function(column, filename, title, color) {
  plot <- ggplot2::ggplot(smoothing_df, ggplot2::aes(x = .data$Year)) +
    ggplot2::geom_line(ggplot2::aes(y = .data$Real, color = "Serie observată"), linewidth = 1) +
    ggplot2::geom_line(ggplot2::aes(y = .data[[column]], color = column), linewidth = 1, linetype = "longdash") +
    ggplot2::scale_color_manual(values = c("Serie observată" = NAVY, column = color), name = NULL) +
    ggplot2::labs(title = title, x = "An", y = "Participare feminină (%)") +
    theme_project()
  save_plot(plot, filename, width = 8.5, height = 4.8)
}
save_smoothing_plot("SES", "smoothing_ses.png", "Seria observată și netezirea exponențială simplă", TEAL)
save_smoothing_plot("Holt", "smoothing_holt.png", "Seria observată și modelul Holt", BURGUNDY)
save_smoothing_plot("ETS", "smoothing_ets.png", "Seria observată și modelul ETS", GOLD)
smooth_long <- smoothing_df |>
  tidyr::pivot_longer(cols = c("Real", "SES", "Holt", "ETS"), names_to = "Serie", values_to = "Valoare")
plot_smooth_compare <- ggplot2::ggplot(smooth_long, ggplot2::aes(x = .data$Year, y = .data$Valoare, color = .data$Serie)) +
  ggplot2::geom_line(linewidth = 1) +
  ggplot2::scale_color_manual(values = c("Real" = NAVY, "SES" = TEAL, "Holt" = BURGUNDY, "ETS" = GOLD), name = NULL) +
  ggplot2::labs(title = "Comparație SES, Holt și ETS", x = "An", y = "Participare feminină (%)") +
  theme_project()
save_plot(plot_smooth_compare, "smoothing_comparison.png", width = 10, height = 5.5)

train_end <- min(2018L, max(series_model_data$Year) - 5L)
test_start <- train_end + 1L
train_values <- series_model_data |>
  dplyr::filter(.data$Year <= train_end) |>
  dplyr::pull(.data$female_lfpr_europe_mean)
test_data <- series_model_data |>
  dplyr::filter(.data$Year >= test_start)
test_years <- test_data$Year
test_values <- test_data$female_lfpr_europe_mean
train_ts <- stats::ts(train_values, start = min(series_model_data$Year), frequency = 1)
horizon <- length(test_values)
if (horizon < 3) {
  add_warning("Setul de test are mai puțin de trei observații; interpretarea metricilor predictive este foarte limitată.")
}

model_names <- c("Naive", "Drift", "SES", "Holt", "ETS", "ARIMA selectat")
forecast_objects <- stats::setNames(vector("list", length(model_names)), model_names)
for (model_name in model_names) {
  forecast_objects[[model_name]] <- fit_forecast_model(model_name, train_ts, horizon, selected_order)
}

holdout_rows <- dplyr::bind_rows(lapply(model_names, function(model_name) {
  forecast_rows(model_name, forecast_objects[[model_name]], test_years, test_values)
}))
write_csv_output(holdout_rows, "forecast_holdout.csv")
naive_pred <- holdout_rows |>
  dplyr::filter(.data$Model == "Naive") |>
  dplyr::pull(.data$Prognoză)
accuracy_rows <- dplyr::bind_rows(lapply(model_names, function(model_name) {
  rows <- holdout_rows |>
    dplyr::filter(.data$Model == model_name)
  forecast_accuracy(
    model_name,
    test_values,
    rows$Prognoză,
    train_ts,
    naive_pred,
    extract_forecast_aicc(forecast_objects[[model_name]])
  )
})) |>
  dplyr::arrange(.data$RMSE)
write_csv_output(accuracy_rows, "forecast_accuracy.csv")

best_forecast_model <- accuracy_rows |>
  dplyr::filter(is.finite(.data$RMSE)) |>
  dplyr::slice_min(order_by = .data$RMSE, n = 1, with_ties = FALSE) |>
  dplyr::pull(.data$Model)
if (length(best_forecast_model) == 0) best_forecast_model <- "ARIMA selectat"

arima_holdout <- forecast_objects[["ARIMA selectat"]]
if (!is.null(arima_holdout)) {
  arima_fc_df <- forecast_rows("ARIMA selectat", arima_holdout, test_years, test_values)
  train_plot <- tibble::tibble(An = series_model_data$Year[series_model_data$Year <= train_end], Valoare = train_values, Segment = "Training")
  test_plot <- tibble::tibble(An = test_years, Valoare = test_values, Segment = "Test real")
  plot_forecast_arima <- ggplot2::ggplot() +
    ggplot2::geom_line(data = train_plot, ggplot2::aes(x = .data$An, y = .data$Valoare, color = .data$Segment), linewidth = 1) +
    ggplot2::geom_line(data = test_plot, ggplot2::aes(x = .data$An, y = .data$Valoare, color = .data$Segment), linewidth = 1) +
    ggplot2::geom_ribbon(data = arima_fc_df, ggplot2::aes(x = .data$An, ymin = .data[["Lower 95%"]], ymax = .data[["Upper 95%"]]), fill = PLUM, alpha = 0.10) +
    ggplot2::geom_ribbon(data = arima_fc_df, ggplot2::aes(x = .data$An, ymin = .data[["Lower 80%"]], ymax = .data[["Upper 80%"]]), fill = TEAL, alpha = 0.16) +
    ggplot2::geom_line(data = arima_fc_df, ggplot2::aes(x = .data$An, y = .data$Prognoză, color = "Prognoză ARIMA"), linewidth = 1, linetype = "longdash") +
    ggplot2::geom_vline(xintercept = test_start - 0.5, color = GOLD, linetype = "dotted", linewidth = 0.8) +
    ggplot2::scale_color_manual(values = c("Training" = NAVY, "Test real" = TEAL, "Prognoză ARIMA" = BURGUNDY), name = NULL) +
    ggplot2::labs(title = "Valori reale și prognoza modelului ARIMA", x = "An", y = "Participare feminină (%)") +
    theme_project()
  save_plot(plot_forecast_arima, "forecast_arima_holdout.png", width = 10.5, height = 5.8)
}

compare_models <- c("Naive", "Holt", "ETS", "ARIMA selectat")
compare_df <- holdout_rows |>
  dplyr::filter(.data$Model %in% compare_models) |>
  dplyr::select("Model", "An", "Prognoză") |>
  tidyr::pivot_wider(names_from = "Model", values_from = "Prognoză") |>
  dplyr::left_join(tibble::tibble(An = test_years, Real = test_values), by = "An") |>
  tidyr::pivot_longer(cols = -An, names_to = "Serie", values_to = "Valoare")
plot_compare_forecasts <- ggplot2::ggplot(compare_df, ggplot2::aes(x = .data$An, y = .data$Valoare, color = .data$Serie)) +
  ggplot2::geom_line(linewidth = 1) +
  ggplot2::geom_point(size = 2) +
  ggplot2::scale_color_manual(values = c("Real" = NAVY, "Naive" = MUTED, "Holt" = BURGUNDY, "ETS" = GOLD, "ARIMA selectat" = TEAL), name = NULL) +
  ggplot2::labs(title = "Compararea prognozelor pe setul de test", x = "An", y = "Participare feminină (%)") +
  theme_project()
save_plot(plot_compare_forecasts, "forecast_model_comparison.png", width = 10, height = 5.5)

plot_rmse <- ggplot2::ggplot(accuracy_rows, ggplot2::aes(x = stats::reorder(.data$Model, -.data$RMSE), y = .data$RMSE, fill = .data$Model)) +
  ggplot2::geom_col(width = 0.68, show.legend = FALSE) +
  ggplot2::coord_flip() +
  ggplot2::scale_fill_manual(values = rep(c(TEAL, GOLD, BURGUNDY, PLUM, NAVY, MUTED), length.out = nrow(accuracy_rows))) +
  ggplot2::labs(title = "Performanța modelelor pe setul de test: RMSE", x = NULL, y = "RMSE") +
  theme_project()
save_plot(plot_rmse, "forecast_rmse_comparison.png", width = 8.5, height = 4.8)
plot_mae <- ggplot2::ggplot(accuracy_rows, ggplot2::aes(x = stats::reorder(.data$Model, -.data$MAE), y = .data$MAE, fill = .data$Model)) +
  ggplot2::geom_col(width = 0.68, show.legend = FALSE) +
  ggplot2::coord_flip() +
  ggplot2::scale_fill_manual(values = rep(c(GOLD, TEAL, BURGUNDY, PLUM, NAVY, MUTED), length.out = nrow(accuracy_rows))) +
  ggplot2::labs(title = "Performanța modelelor pe setul de test: MAE", x = NULL, y = "MAE") +
  theme_project()
save_plot(plot_mae, "forecast_mae_comparison.png", width = 8.5, height = 4.8)

rolling_models <- c("Naive", "ETS", "ARIMA selectat")
rolling_rows <- lapply(rolling_models, function(model_name) {
  errors <- tryCatch(
    {
      if (model_name == "Naive") {
        forecast::tsCV(series_ts, forecastfunction = function(y, h) forecast::naive(y, h = h), h = 1)
      } else if (model_name == "ETS") {
        forecast::tsCV(series_ts, forecastfunction = function(y, h) forecast::forecast(forecast::ets(y, model = "ZZN"), h = h), h = 1)
      } else {
        forecast::tsCV(series_ts, forecastfunction = function(y, h) {
          fit <- forecast::Arima(y, order = selected_order, include.drift = selected_order[2] == 1, include.mean = selected_order[2] == 0, method = "ML")
          forecast::forecast(fit, h = h)
        }, h = 1)
      }
    },
    error = function(e) NULL
  )
  rmse <- if (is.null(errors)) NA_real_ else sqrt(mean(errors^2, na.rm = TRUE))
  tibble::tibble(Model = model_name, `RMSE rolling-origin` = rmse, Observație = ifelse(is.na(rmse), "Validarea rolling-origin nu a fost stabilă.", "Validare h=1 prin origine mobilă."))
})
rolling_accuracy <- dplyr::bind_rows(rolling_rows)
write_csv_output(rolling_accuracy, "rolling_origin_accuracy.csv")

future_horizon <- FORECAST_END_YEAR - END_YEAR
future_years <- (END_YEAR + 1L):FORECAST_END_YEAR
future_fc <- fit_forecast_model(best_forecast_model, series_ts, future_horizon, selected_order)
if (is.null(future_fc)) {
  future_fc <- fit_forecast_model("ARIMA selectat", series_ts, future_horizon, selected_order)
  best_forecast_model <- "ARIMA selectat"
}
future_forecast <- forecast_rows(best_forecast_model, future_fc, future_years)
write_csv_output(future_forecast, "future_forecast.csv")
plot_future <- ggplot2::ggplot() +
  ggplot2::geom_line(data = series_model_data, ggplot2::aes(x = .data$Year, y = .data$female_lfpr_europe_mean, color = "Serie observată"), linewidth = 1) +
  ggplot2::geom_ribbon(data = future_forecast, ggplot2::aes(x = .data$An, ymin = .data[["Lower 95%"]], ymax = .data[["Upper 95%"]]), fill = PLUM, alpha = 0.10) +
  ggplot2::geom_ribbon(data = future_forecast, ggplot2::aes(x = .data$An, ymin = .data[["Lower 80%"]], ymax = .data[["Upper 80%"]]), fill = TEAL, alpha = 0.16) +
  ggplot2::geom_line(data = future_forecast, ggplot2::aes(x = .data$An, y = .data$Prognoză, color = "Prognoză 2024–2026"), linewidth = 1, linetype = "longdash") +
  ggplot2::geom_point(data = future_forecast, ggplot2::aes(x = .data$An, y = .data$Prognoză), color = BURGUNDY, size = 2.4) +
  ggplot2::scale_color_manual(values = c("Serie observată" = NAVY, "Prognoză 2024–2026" = BURGUNDY), name = NULL) +
  ggplot2::labs(title = "Prognoză finală exploratorie 2024–2026", x = "An", y = "Participare feminină (%)") +
  theme_project()
save_plot(plot_future, "forecast_future_2024_2026.png", width = 10, height = 5.5)

multivar_yearly <- analysis_data |>
  dplyr::group_by(.data$Year) |>
  dplyr::summarise(
    FLFP = mean(.data$female_lfpr, na.rm = TRUE),
    Inflation = mean(.data$inflation, na.rm = TRUE),
    GDPGrowth = mean(.data$gdp_growth, na.rm = TRUE),
    FemaleUnemployment = mean(.data$female_unemployment, na.rm = TRUE),
    n_flfp = sum(!is.na(.data$female_lfpr)),
    n_inflation = sum(!is.na(.data$inflation)),
    n_gdp_growth = sum(!is.na(.data$gdp_growth)),
    n_unemployment = sum(!is.na(.data$female_unemployment)),
    .groups = "drop"
  ) |>
  dplyr::filter(.data$Year >= START_YEAR, .data$Year <= END_YEAR) |>
  dplyr::mutate(dplyr::across(c("FLFP", "Inflation", "GDPGrowth", "FemaleUnemployment"), ~ ifelse(is.nan(.x), NA_real_, .x)))
write_csv_output(multivar_yearly, "multivariate_series.csv")

standardized <- multivar_yearly |>
  dplyr::select("Year", "FLFP", "Inflation", "GDPGrowth", "FemaleUnemployment") |>
  tidyr::drop_na() |>
  dplyr::mutate(dplyr::across(-"Year", ~ as.numeric(scale(.x)))) |>
  tidyr::pivot_longer(cols = -"Year", names_to = "Serie", values_to = "Valoare")
plot_multivar <- ggplot2::ggplot(standardized, ggplot2::aes(x = .data$Year, y = .data$Valoare, color = .data$Serie)) +
  ggplot2::geom_hline(yintercept = 0, color = GOLD, linewidth = 0.45) +
  ggplot2::geom_line(linewidth = 0.95) +
  ggplot2::scale_color_manual(values = c("FLFP" = NAVY, "Inflation" = BURGUNDY, "GDPGrowth" = GOLD, "FemaleUnemployment" = TEAL), name = NULL) +
  ggplot2::labs(title = "Evoluția standardizată a principalilor indicatori", x = "An", y = "Z-score") +
  theme_project()
save_plot(plot_multivar, "multivariate_standardized.png", width = 10.5, height = 5.8)

series_stationarity <- lapply(c("FLFP", "Inflation", "GDPGrowth", "FemaleUnemployment"), function(series_name) {
  values <- multivar_yearly[[series_name]]
  values <- values[is.finite(values)]
  if (length(values) < 8) {
    return(tibble::tibble(Serie = series_name, ADF = "Indisponibil", PP = "Indisponibil", KPSS = "Indisponibil", `Ordin de integrare estimat` = "Date insuficiente"))
  }
  level <- stationarity_tests(values, "Nivel")
  diffed <- stationarity_tests(diff(values), "Diferența I")
  level_v <- stationarity_vote(level)
  diff_v <- stationarity_vote(diffed)
  order <- if (level_v$status == "compatibilă cu staționaritatea") "I(0)" else if (diff_v$status == "compatibilă cu staționaritatea") "I(1)" else "Neclar"
  test_summary <- level |>
    dplyr::group_by(.data$Test) |>
    dplyr::summarise(Suport = ifelse(any(.data[["Susține staționaritatea"]], na.rm = TRUE), "staționaritate", "nestaționaritate / mixt"), .groups = "drop")
  tibble::tibble(
    Serie = series_name,
    ADF = test_summary$Suport[match("ADF", test_summary$Test)],
    PP = test_summary$Suport[match("PP", test_summary$Test)],
    KPSS = test_summary$Suport[match("KPSS", test_summary$Test)],
    `Ordin de integrare estimat` = order
  )
})
multivariate_stationarity <- dplyr::bind_rows(series_stationarity)
write_csv_output(multivariate_stationarity, "multivariate_stationarity.csv")

bivar_data <- multivar_yearly |>
  dplyr::select("Year", "FLFP", "Inflation") |>
  tidyr::drop_na()
flfp_order <- multivariate_stationarity$`Ordin de integrare estimat`[multivariate_stationarity$Serie == "FLFP"]
inflation_order <- multivariate_stationarity$`Ordin de integrare estimat`[multivariate_stationarity$Serie == "Inflation"]
use_diff_bivar <- !(flfp_order == "I(0)" && inflation_order == "I(0)")
bivar_model_data <- if (use_diff_bivar) {
  bivar_data |>
    dplyr::mutate(FLFP = .data$FLFP - dplyr::lag(.data$FLFP), Inflation = .data$Inflation - dplyr::lag(.data$Inflation)) |>
    tidyr::drop_na()
} else {
  bivar_data
}
bivar_label <- ifelse(use_diff_bivar, "diferențe de ordinul I", "niveluri")

ccf_object <- tryCatch(stats::ccf(bivar_model_data$FLFP, bivar_model_data$Inflation, lag.max = min(6L, floor(nrow(bivar_model_data) / 3L)), plot = FALSE), error = function(e) NULL)
if (!is.null(ccf_object)) {
  ccf_df <- tibble::tibble(Lag = as.numeric(ccf_object$lag), Corelație = as.numeric(ccf_object$acf))
  plot_ccf <- ggplot2::ggplot(ccf_df, ggplot2::aes(x = .data$Lag, y = .data$Corelație)) +
    ggplot2::geom_hline(yintercept = 0, color = MUTED, linewidth = 0.35) +
    ggplot2::geom_col(fill = TEAL, width = 0.55) +
    ggplot2::labs(title = "Corelație încrucișată FLFP–Inflație", subtitle = paste("Serii utilizate în", bivar_label), x = "Lag", y = "Corelație") +
    theme_project()
  save_plot(plot_ccf, "multivariate_ccf_flfp_inflation.png", width = 8, height = 4.8)
}

scatter_temporal <- ggplot2::ggplot(bivar_data, ggplot2::aes(x = .data$Inflation, y = .data$FLFP, color = .data$Year)) +
  ggplot2::geom_path(color = MUTED, linewidth = 0.45, alpha = 0.55) +
  ggplot2::geom_point(size = 2.6) +
  ggplot2::scale_color_gradient(low = TEAL, high = BURGUNDY) +
  ggplot2::labs(title = "Traiectorie temporală FLFP–Inflație", x = "Inflație medie neponderată (%)", y = "Participare feminină medie neponderată (%)", color = "An") +
  theme_project()
save_plot(scatter_temporal, "multivariate_scatter_flfp_inflation.png", width = 8, height = 5.2)

common_standardized <- bivar_data |>
  dplyr::mutate(FLFP = as.numeric(scale(.data$FLFP)), Inflation = as.numeric(scale(.data$Inflation))) |>
  tidyr::pivot_longer(cols = c("FLFP", "Inflation"), names_to = "Serie", values_to = "Valoare")
plot_common <- ggplot2::ggplot(common_standardized, ggplot2::aes(x = .data$Year, y = .data$Valoare, color = .data$Serie)) +
  ggplot2::geom_hline(yintercept = 0, color = GOLD, linewidth = 0.45) +
  ggplot2::geom_line(linewidth = 1) +
  ggplot2::scale_color_manual(values = c("FLFP" = NAVY, "Inflation" = BURGUNDY), name = NULL) +
  ggplot2::labs(title = "Evoluție standardizată comună: FLFP și inflație", x = "An", y = "Z-score") +
  theme_project()
save_plot(plot_common, "multivariate_flfp_inflation_standardized.png", width = 9, height = 5.2)

multivariate_method_used <- "Nicio estimare multivariată"
granger_results <- tibble::tibble(
  Direcție = character(),
  `Lag VAR` = integer(),
  Statistică = numeric(),
  `p-value` = numeric(),
  Concluzie = character()
)
var_summary <- tibble::tibble(
  Componentă = character(),
  Indicator = character(),
  Valoare = character()
)
ardl_summary <- tibble::tibble(
  Termen = character(),
  Coeficient = numeric(),
  `p-value` = numeric(),
  Concluzie = character()
)

if (nrow(bivar_model_data) >= 12) {
  var_input <- bivar_model_data |>
    dplyr::select("FLFP", "Inflation")
  lag_choice <- 1L
  var_model <- tryCatch(vars::VAR(var_input, p = lag_choice, type = "const"), error = function(e) NULL)
  if (!is.null(var_model)) {
    roots <- tryCatch(vars::roots(var_model), error = function(e) numeric())
    stable <- length(roots) > 0 && all(Mod(roots) < 1)
    serial <- tryCatch(vars::serial.test(var_model, lags.pt = min(4L, floor(nrow(var_input) / 3L)), type = "PT.asymptotic"), error = function(e) NULL)
    normality <- tryCatch(vars::normality.test(var_model), error = function(e) NULL)
    granger_inf <- tryCatch(vars::causality(var_model, cause = "Inflation"), error = function(e) NULL)
    granger_flfp <- tryCatch(vars::causality(var_model, cause = "FLFP"), error = function(e) NULL)
    granger_results <- tibble::tibble(
      Direcție = c("Inflație → FLFP", "FLFP → Inflație"),
      `Lag VAR` = lag_choice,
      Statistică = c(
        ifelse(is.null(granger_inf), NA_real_, as.numeric(granger_inf$Granger$statistic)),
        ifelse(is.null(granger_flfp), NA_real_, as.numeric(granger_flfp$Granger$statistic))
      ),
      `p-value` = c(
        ifelse(is.null(granger_inf), NA_real_, as.numeric(granger_inf$Granger$p.value)),
        ifelse(is.null(granger_flfp), NA_real_, as.numeric(granger_flfp$Granger$p.value))
      ),
      Concluzie = c(
        ifelse(!is.null(granger_inf) && granger_inf$Granger$p.value < 0.05, "Respingem H0: inflația are valoare predictivă incrementală pentru FLFP.", "Nu respingem H0: nu apar dovezi suficiente de valoare predictivă incrementală."),
        ifelse(!is.null(granger_flfp) && granger_flfp$Granger$p.value < 0.05, "Respingem H0: FLFP are valoare predictivă incrementală pentru inflație.", "Nu respingem H0: nu apar dovezi suficiente de valoare predictivă incrementală.")
      )
    )
    var_summary <- tibble::tibble(
      Componentă = c("Specificație", "Serii utilizate", "Lag ales", "Stabilitate", "Autocorelare reziduuri", "Normalitate reziduuri"),
      Indicator = c("Model", "Transformare", "p", "Roots", "Portmanteau p-value", "Jarque-Bera multivariat p-value"),
      Valoare = c(
        "VAR bivariabil FLFP–Inflație",
        bivar_label,
        as.character(lag_choice),
        ifelse(stable, "Stabil", "Instabil / de interpretat cu prudență"),
        ifelse(is.null(serial), NA_character_, formatC(serial$serial$p.value, format = "f", digits = 4)),
        ifelse(is.null(normality), NA_character_, formatC(normality$jb.mul$JB$p.value, format = "f", digits = 4))
      )
    )
    multivariate_method_used <- "VAR bivariabil parcimonios"
    if (stable) {
      irf_object <- tryCatch(vars::irf(var_model, impulse = "Inflation", response = "FLFP", n.ahead = min(5L, nrow(var_input) - 3L), boot = TRUE, runs = 200), error = function(e) NULL)
      if (!is.null(irf_object)) {
        horizon_irf <- seq_along(irf_object$irf$Inflation[, "FLFP"]) - 1L
        irf_df <- tibble::tibble(
          Orizont = horizon_irf,
          Răspuns = as.numeric(irf_object$irf$Inflation[, "FLFP"]),
          Inferior = as.numeric(irf_object$Lower$Inflation[, "FLFP"]),
          Superior = as.numeric(irf_object$Upper$Inflation[, "FLFP"])
        )
        plot_irf <- ggplot2::ggplot(irf_df, ggplot2::aes(x = .data$Orizont, y = .data$Răspuns)) +
          ggplot2::geom_hline(yintercept = 0, color = GOLD, linewidth = 0.5) +
          ggplot2::geom_ribbon(ggplot2::aes(ymin = .data$Inferior, ymax = .data$Superior), fill = TEAL, alpha = 0.16) +
          ggplot2::geom_line(color = BURGUNDY, linewidth = 1) +
          ggplot2::geom_point(color = NAVY, size = 2) +
          ggplot2::labs(title = "Răspunsul participării feminine la un șoc al inflației", x = "Orizont", y = "Răspuns FLFP") +
          theme_project()
        save_plot(plot_irf, "multivariate_irf_inflation_to_flfp.png", width = 8, height = 4.8)
      }
    } else {
      add_warning("VAR-ul bivariabil este estimabil, dar stabilitatea nu este confirmată; IRF nu este interpretată ca rezultat principal.")
    }
  } else {
    add_warning("VAR-ul bivariabil nu a putut fi estimat stabil; se încearcă o alternativă ARDL parcimonioasă.")
  }
}

if (multivariate_method_used == "Nicio estimare multivariată" && nrow(bivar_data) >= 12 && !("Neclar după diferențierea de ordinul I" %in% c(flfp_order, inflation_order))) {
  ardl_data <- bivar_data |>
    dplyr::mutate(
      FLFP_lag1 = dplyr::lag(.data$FLFP),
      Inflation_lag1 = dplyr::lag(.data$Inflation)
    ) |>
    tidyr::drop_na()
  ardl_fit <- tryCatch(stats::lm(FLFP ~ FLFP_lag1 + Inflation + Inflation_lag1, data = ardl_data), error = function(e) NULL)
  if (!is.null(ardl_fit)) {
    ardl_summary <- broom::tidy(ardl_fit) |>
      dplyr::transmute(
        Termen = .data$term,
        Coeficient = .data$estimate,
        `p-value` = .data$p.value,
        Concluzie = "Model ARDL bivariabil cu lag maxim 1; rezultatele sunt exploratorii."
      )
    multivariate_method_used <- "ARDL bivariabil parcimonios"
  }
}

if (nrow(granger_results) == 0) {
  granger_results <- tibble::tibble(
    Direcție = c("Inflație → FLFP", "FLFP → Inflație"),
    `Lag VAR` = NA_integer_,
    Statistică = NA_real_,
    `p-value` = NA_real_,
    Concluzie = "Testul Granger nu a fost estimat deoarece condițiile pentru VAR staționar și stabil nu au fost îndeplinite."
  )
}
if (nrow(var_summary) == 0) {
  var_summary <- tibble::tibble(
    Componentă = "VAR",
    Indicator = "Status",
    Valoare = "VAR nu a fost estimat stabil pentru seriile disponibile."
  )
}
if (nrow(ardl_summary) == 0) {
  ardl_summary <- tibble::tibble(
    Termen = "ARDL",
    Coeficient = NA_real_,
    `p-value` = NA_real_,
    Concluzie = ifelse(multivariate_method_used == "ARDL bivariabil parcimonios", "Model estimat.", "ARDL nu a fost utilizat deoarece VAR-ul bivariabil a fost estimat sau condițiile nu au justificat alternativa.")
  )
}
write_csv_output(granger_results, "granger_results.csv")
write_csv_output(var_summary, "var_summary.csv")
write_csv_output(ardl_summary, "ardl_summary.csv")

valid_granger <- granger_results |>
  dplyr::filter(is.finite(.data[["p-value"]]))
granger_detected <- nrow(valid_granger) > 0 && any(valid_granger[["p-value"]] < 0.05)
multivariate_conclusion <- if (multivariate_method_used == "VAR bivariabil parcimonios") {
  if (granger_detected) {
    "Modelul VAR bivariabil indică cel puțin o relație de valoare predictivă temporală între FLFP și inflație, interpretată strict exploratoriu în contextul eșantionului anual scurt."
  } else {
    "Modelul VAR bivariabil nu oferă dovezi suficiente pentru o relație Granger robustă între FLFP și inflație în eșantionul anual analizat."
  }
} else if (multivariate_method_used == "ARDL bivariabil parcimonios") {
  "A fost utilizată o alternativă ARDL bivariabilă foarte parcimonioasă; rezultatele descriu asocieri dinamice exploratorii, nu efecte cauzale structurale."
} else {
  "Condițiile empirice nu justifică estimarea unui model multivariat robust; analiza rămâne descriptivă și corelațională."
}

add_warning("Modelele SARIMA nu sunt utilizate în analiza principală deoarece seria are frecvență anuală și nu conține o structură sezonieră intra-anuală observabilă.")
add_warning("Prognoza 2024–2026 reprezintă o extensie statistică a seriei și nu include explicit șocuri economice viitoare sau modificări structurale.")
add_warning("Seria anuală conține numai 23 de observații, astfel încât prognozele au un grad ridicat de incertitudine și trebuie interpretate ca exercițiu econometric, nu ca predicții structurale ferme.")
add_warning("Dimensiunea eșantionului este redusă pentru Johansen/VECM; aceste modele nu sunt forțate în analiza principală.")

metadata <- list(
  start_year = START_YEAR,
  end_year = END_YEAR,
  frequency = "Anuală",
  number_observations = length(series_ts),
  aggregation_method = "Media neponderată a ratelor naționale de participare feminină pentru economiile europene din eșantion.",
  balanced_sample_used = balanced_sample_used,
  balanced_country_count = length(balanced_countries),
  available_country_count_min = min(country_counts$n_countries),
  available_country_count_max = max(country_counts$n_countries),
  estimated_integration_order = estimated_integration_order,
  selected_arima = selected_arima_name,
  auto_arima_check = auto_arima_label,
  selected_forecast_model = best_forecast_model,
  train_start = min(series_model_data$Year),
  train_end = train_end,
  test_start = test_start,
  test_end = max(test_years),
  multivariate_method_used = multivariate_method_used,
  multivariate_conclusion = multivariate_conclusion,
  visual_conclusion = visual_conclusion,
  sarima_note = "Modelele SARIMA nu sunt utilizate în analiza principală deoarece seria are frecvență anuală și nu conține o structură sezonieră intra-anuală observabilă.",
  granger_note = "Cauzalitatea Granger indică valoare predictivă temporală incrementală și nu reprezintă cauzalitate economică structurală.",
  irf_note = "IRF descrie dinamica modelului VAR estimat și nu trebuie interpretată automat ca efect cauzal structural.",
  warnings = warnings_list,
  timestamp = format(Sys.time(), "%Y-%m-%d %H:%M:%S %z")
)
jsonlite::write_json(metadata, file.path(META_DIR, "time_series_analysis.json"), pretty = TRUE, auto_unbox = TRUE, na = "null")

saveRDS(selected_fit, file.path(MODEL_DIR, "selected_arima.rds"))
saveRDS(list(ses = ses_fit, holt = holt_fit, ets = ets_model), file.path(MODEL_DIR, "smoothing_models.rds"))

message("Analiza seriilor de timp a fost finalizată: ", OUTPUT_ROOT)

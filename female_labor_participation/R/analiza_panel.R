#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  required_packages <- c(
    "readxl", "dplyr", "tidyr", "ggplot2", "plm", "lmtest", "sandwich",
    "broom", "car", "jsonlite", "scales", "patchwork", "moments",
    "corrplot", "ggrepel"
  )
  missing_packages <- required_packages[!vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)]
  if (length(missing_packages) > 0) {
    stop(
      paste0(
        "Pachete R lipsă: ", paste(missing_packages, collapse = ", "),
        ". Instalează-le manual înainte de rularea analizei panel."
      ),
      call. = FALSE
    )
  }
  invisible(lapply(required_packages, require, character.only = TRUE))
})

START_YEAR <- 2001L
END_YEAR <- 2023L

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
    script <- file.path(candidate, "R", "analiza_panel.R")
    if (file.exists(dataset) && file.exists(script)) {
      return(normalizePath(candidate, winslash = "/", mustWork = TRUE))
    }
  }
  stop("Nu pot identifica rădăcina proiectului pentru analiza panel.", call. = FALSE)
}

ROOT <- find_project_root()
DATA_PATH <- file.path(ROOT, "data", "P_Data_Extract_From_World_Development_Indicators.xlsx")
COUNTRY_CONFIG_PATH <- file.path(ROOT, "config", "europe_countries.csv")
OUTPUT_ROOT <- file.path(ROOT, "outputs", "panel")
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
  ~slug, ~code, ~label, ~short_label,
  "female_lfpr", "SL.TLF.ACTI.FE.ZS", "Participarea femeilor la forța de muncă, 15–64 ani", "Participarea femeilor",
  "gdp_pc_ppp", "NY.GDP.PCAP.PP.KD", "PIB real pe locuitor, PPP", "PIB/locuitor PPP",
  "gdp_growth", "NY.GDP.MKTP.KD.ZG", "Creșterea PIB real", "Creștere PIB",
  "inflation", "FP.CPI.TOTL.ZG", "Inflația, prețurile de consum", "Inflație",
  "female_unemployment", "SL.UEM.TOTL.FE.ZS", "Șomajul femeilor", "Șomaj feminin",
  "fertility", "SP.DYN.TFRT.IN", "Rata totală a fertilității", "Fertilitate",
  "urbanization", "SP.URB.TOTL.IN.ZS", "Urbanizarea", "Urbanizare",
  "female_services", "SL.SRV.EMPL.FE.ZS", "Femeile ocupate în servicii", "Femei în servicii",
  "child_dependency", "SP.POP.DPND.YG", "Rata de dependență a copiilor", "Dependența copiilor",
  "female_vulnerable", "SL.EMP.VULN.FE.ZS", "Ocuparea vulnerabilă a femeilor", "Ocupare vulnerabilă",
  "internet", "IT.NET.USER.ZS", "Utilizarea internetului", "Internet",
  "women_parliament", "SG.GEN.PARL.ZS", "Femeile în parlamentele naționale", "Femei în parlament",
  "female_tertiary", "SE.TER.ENRR.FE", "Educația terțiară a femeilor", "Educație terțiară",
  "control_corruption", "GOV_WGI_CC_EST", "Controlul corupției", "Controlul corupției"
)

label_for <- function(term) {
  if (term == "(Intercept)") return("Intercept")
  if (term == "LogGDP") return("Log PIB/locuitor PPP")
  if (grepl(":", term, fixed = TRUE)) {
    return(paste(vapply(strsplit(term, ":", fixed = TRUE)[[1]], label_for, character(1)), collapse = " × "))
  }
  match <- variables[variables$slug == term, ]
  if (nrow(match) == 1) return(match$short_label[[1]])
  term
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

if (!file.exists(DATA_PATH)) stop(paste0("Fișierul Excel nu există: ", DATA_PATH), call. = FALSE)
if (!file.exists(COUNTRY_CONFIG_PATH)) stop(paste0("Fișierul config/europe_countries.csv nu există: ", COUNTRY_CONFIG_PATH), call. = FALSE)

raw_data <- suppressWarnings(readxl::read_excel(DATA_PATH, na = c("", "NA", "..")))
europe_countries <- read.csv(COUNTRY_CONFIG_PATH, stringsAsFactors = FALSE, fileEncoding = "UTF-8")
column_names <- names(raw_data)
mapped_columns <- stats::setNames(vapply(variables$code, function(code) find_column_by_code(column_names, code), character(1)), variables$slug)

analysis_data <- data.frame(
  ISO3 = as.character(raw_data[["Country Code"]]),
  country_original = as.character(raw_data[["Country Name"]]),
  Year = as.integer(coerce_numeric(raw_data[["Time"]])),
  stringsAsFactors = FALSE
)
for (slug in variables$slug) {
  analysis_data[[slug]] <- coerce_numeric(raw_data[[mapped_columns[[slug]]]])
}
analysis_data <- analysis_data |>
  dplyr::filter(!is.na(.data$Year), .data$Year >= START_YEAR, .data$Year <= END_YEAR, .data$ISO3 %in% europe_countries$iso3) |>
  dplyr::left_join(europe_countries, by = c("ISO3" = "iso3")) |>
  dplyr::mutate(
    country = dplyr::coalesce(.data$country, .data$country_original),
    LogGDP = dplyr::if_else(!is.na(.data$gdp_pc_ppp) & .data$gdp_pc_ppp > 0, log(.data$gdp_pc_ppp), NA_real_)
  ) |>
  dplyr::select("country", "ISO3", "Year", dplyr::everything(), -"country_original")

candidate_predictors <- c(
  "LogGDP", "gdp_growth", "inflation", "female_unemployment", "fertility",
  "urbanization", "female_services", "child_dependency", "female_vulnerable",
  "internet", "women_parliament", "female_tertiary", "control_corruption"
)
main_formula <- stats::as.formula(paste("female_lfpr ~", paste(candidate_predictors, collapse = " + ")))

complete_model_data <- function(dataframe, variables_required) {
  dataframe |>
    dplyr::select("country", "ISO3", "Year", dplyr::all_of(variables_required)) |>
    dplyr::filter(dplyr::if_all(dplyr::all_of(variables_required), ~ !is.na(.x))) |>
    dplyr::arrange(.data$ISO3, .data$Year)
}

fit_panel <- function(formula, dataframe, model, effect = "individual") {
  pdata <- plm::pdata.frame(dataframe, index = c("ISO3", "Year"), drop.index = FALSE, row.names = FALSE)
  tryCatch(plm::plm(formula, data = pdata, model = model, effect = effect), error = function(e) NULL)
}

fit_lm_aux <- function(formula, dataframe) {
  tryCatch(stats::lm(formula, data = dataframe), error = function(e) NULL)
}

extract_r2 <- function(model) {
  if (is.null(model)) return(c(rsq = NA_real_, adjrsq = NA_real_))
  values <- tryCatch(summary(model)$r.squared, error = function(e) c(rsq = NA_real_, adjrsq = NA_real_))
  c(rsq = unname(values[["rsq"]]), adjrsq = unname(values[["adjrsq"]]))
}

coefficient_table <- function(model, vcov_matrix = NULL) {
  if (is.null(model)) return(tibble::tibble())
  if (is.null(vcov_matrix)) {
    matrix <- tryCatch(summary(model)$coefficients, error = function(e) NULL)
  } else {
    matrix <- tryCatch(as.matrix(lmtest::coeftest(model, vcov. = vcov_matrix)), error = function(e) NULL)
  }
  if (is.null(matrix) || nrow(matrix) == 0) return(tibble::tibble())
  estimate <- matrix[, 1]
  se <- matrix[, 2]
  statistic <- matrix[, 3]
  pvalue <- matrix[, 4]
  critical <- stats::qnorm(0.975)
  tibble::tibble(
    Termen = vapply(rownames(matrix), label_for, character(1)),
    term_raw = rownames(matrix),
    Coeficient = as.numeric(estimate),
    `Standard error` = as.numeric(se),
    Statistică = as.numeric(statistic),
    `p-value` = as.numeric(pvalue),
    `IC 95% inferior` = as.numeric(estimate - critical * se),
    `IC 95% superior` = as.numeric(estimate + critical * se)
  )
}

model_metrics <- function(model, dataframe, predictors, label) {
  r2 <- extract_r2(model)
  tibble::tibble(
    Model = label,
    N = nrow(dataframe),
    Țări = dplyr::n_distinct(dataframe$ISO3),
    Ani = dplyr::n_distinct(dataframe$Year),
    Predictori = length(predictors),
    `Grade libertate reziduale` = if (is.null(model)) NA_integer_ else stats::df.residual(model),
    `R² relevant` = r2[["rsq"]],
    `R² ajustat` = r2[["adjrsq"]],
    AIC = tryCatch(stats::AIC(model), error = function(e) NA_real_),
    BIC = tryCatch(stats::BIC(model), error = function(e) NA_real_)
  )
}

format_decision <- function(p_value, issue_text, no_issue_text) {
  if (is.na(p_value) || !is.finite(p_value)) return("Testul nu a putut fi calculat stabil pentru eșantionul disponibil.")
  if (p_value < 0.05) {
    paste("Respingem ipoteza nulă la pragul de 5%.", issue_text)
  } else {
    paste("Nu respingem ipoteza nulă la pragul de 5%.", no_issue_text)
  }
}

test_to_row <- function(result, category, test, h0, issue_text, no_issue_text) {
  statistic <- if (is.null(result)) NA_real_ else suppressWarnings(as.numeric(result$statistic[[1]]))
  p_value <- if (is.null(result)) NA_real_ else suppressWarnings(as.numeric(result$p.value))
  parameter <- if (is.null(result) || is.null(result$parameter)) NA_character_ else paste(names(result$parameter), as.numeric(result$parameter), collapse = "; ")
  tibble::tibble(
    Categorie = category,
    Test = test,
    `Ipoteza nulă` = h0,
    Statistică = statistic,
    df = parameter,
    `p-value` = p_value,
    Decizie = dplyr::case_when(
      is.na(p_value) ~ "Nu poate fi calculat",
      p_value < 0.05 ~ "Respingem H0",
      TRUE ~ "Nu respingem H0"
    ),
    Interpretare = format_decision(p_value, issue_text, no_issue_text)
  )
}

actual_fitted_frame <- function(model, dataframe) {
  indexes <- plm::index(model)
  fitted_values <- as.numeric(stats::fitted(model))
  residual_values <- as.numeric(stats::residuals(model))
  frame <- data.frame(
    ISO3 = as.character(indexes[[1]]),
    Year = as.integer(as.character(indexes[[2]])),
    fitted = fitted_values,
    residual = residual_values,
    stringsAsFactors = FALSE
  )
  dataframe |>
    dplyr::select("country", "ISO3", "Year", "female_lfpr") |>
    dplyr::inner_join(frame, by = c("ISO3", "Year")) |>
    dplyr::mutate(std_residual = as.numeric(scale(.data$residual)))
}

panel_pdata <- plm::pdata.frame(analysis_data, index = c("ISO3", "Year"), drop.index = FALSE, row.names = FALSE)
panel_index_counts <- analysis_data |>
  dplyr::count(.data$ISO3, name = "T") |>
  dplyr::summarise(Tmin = min(.data$T), Tmax = max(.data$T), balanced = dplyr::n_distinct(.data$T) == 1)
balanced_panel <- isTRUE(panel_index_counts$balanced[[1]])
panel_structure <- tibble::tibble(
  Indicator = c("Număr de țări", "Perioadă", "Număr observații country-year", "Număr variabile candidate", "Panel echilibrat / neechilibrat", "Tmin", "Tmax"),
  Valoare = c(
    dplyr::n_distinct(analysis_data$ISO3),
    paste0(START_YEAR, "–", END_YEAR),
    nrow(analysis_data),
    length(candidate_predictors),
    if (balanced_panel) "Panel echilibrat" else "Panel neechilibrat",
    panel_index_counts$Tmin[[1]],
    panel_index_counts$Tmax[[1]]
  )
)
write_csv_output(panel_structure, "panel_structure.csv")

coverage <- analysis_data |>
  dplyr::mutate(disponibil = !is.na(.data$female_lfpr)) |>
  dplyr::select("country", "Year", "disponibil")
coverage$country <- factor(coverage$country, levels = sort(unique(coverage$country)))
save_plot(
  ggplot2::ggplot(coverage, ggplot2::aes(x = .data$Year, y = .data$country, fill = .data$disponibil)) +
    ggplot2::geom_tile(color = "white", linewidth = 0.15) +
    ggplot2::scale_fill_manual(values = c("FALSE" = "#EFE4DB", "TRUE" = TEAL), labels = c("Absentă", "Disponibilă"), name = "") +
    ggplot2::labs(title = "Acoperirea panelului pe țări și ani", x = "An", y = "") +
    theme_project(base_size = 9) +
    ggplot2::theme(axis.text.y = ggplot2::element_text(size = 7), legend.position = "bottom"),
  "panel_coverage_heatmap.png",
  width = 11,
  height = 8.5
)

variation_variables <- c("female_lfpr", candidate_predictors)
within_between_variation <- lapply(variation_variables, function(variable) {
  data_var <- analysis_data |>
    dplyr::select("ISO3", dplyr::all_of(variable)) |>
    dplyr::filter(!is.na(.data[[variable]]))
  country_means <- data_var |>
    dplyr::group_by(.data$ISO3) |>
    dplyr::summarise(country_mean = mean(.data[[variable]], na.rm = TRUE), .groups = "drop")
  joined <- data_var |>
    dplyr::left_join(country_means, by = "ISO3") |>
    dplyr::mutate(within_component = .data[[variable]] - .data$country_mean)
  tibble::tibble(
    Variabilă = label_for(variable),
    `Media globală` = mean(data_var[[variable]], na.rm = TRUE),
    `Overall SD` = stats::sd(data_var[[variable]], na.rm = TRUE),
    `Between SD` = stats::sd(country_means$country_mean, na.rm = TRUE),
    `Within SD` = stats::sd(joined$within_component, na.rm = TRUE)
  )
}) |>
  dplyr::bind_rows()
write_csv_output(within_between_variation, "within_between_variation.csv")

variation_long <- within_between_variation |>
  dplyr::select("Variabilă", "Between SD", "Within SD") |>
  tidyr::pivot_longer(cols = c("Between SD", "Within SD"), names_to = "Tip variație", values_to = "SD")
save_plot(
  ggplot2::ggplot(variation_long, ggplot2::aes(x = stats::reorder(.data$Variabilă, .data$SD), y = .data$SD, fill = .data$`Tip variație`)) +
    ggplot2::geom_col(position = "dodge", alpha = 0.88) +
    ggplot2::coord_flip() +
    ggplot2::scale_fill_manual(values = c("Between SD" = GOLD, "Within SD" = TEAL)) +
    ggplot2::labs(title = "Variația within și between a principalilor indicatori", x = "", y = "Deviație standard") +
    theme_project(),
  "within_between_variation.png",
  width = 10,
  height = 7
)

country_heterogeneity <- analysis_data |>
  dplyr::filter(!is.na(.data$female_lfpr)) |>
  dplyr::group_by(.data$country) |>
  dplyr::summarise(
    Media = mean(.data$female_lfpr),
    N = dplyr::n(),
    SD = stats::sd(.data$female_lfpr),
    SE = .data$SD / sqrt(.data$N),
    `IC 95% inferior` = .data$Media - stats::qt(0.975, pmax(.data$N - 1, 1)) * .data$SE,
    `IC 95% superior` = .data$Media + stats::qt(0.975, pmax(.data$N - 1, 1)) * .data$SE,
    .groups = "drop"
  )
europe_mean <- mean(analysis_data$female_lfpr, na.rm = TRUE)
save_plot(
  ggplot2::ggplot(country_heterogeneity, ggplot2::aes(x = .data$Media, y = stats::reorder(.data$country, .data$Media))) +
    ggplot2::geom_vline(xintercept = europe_mean, color = GOLD, linewidth = 0.9) +
    ggplot2::geom_errorbar(ggplot2::aes(xmin = .data$`IC 95% inferior`, xmax = .data$`IC 95% superior`), width = 0.15, color = NAVY, alpha = 0.75) +
    ggplot2::geom_point(color = TEAL, size = 2.4) +
    ggplot2::labs(title = "Participarea medie a femeilor pe țări", x = "Medie 2001–2023 (%)", y = "") +
    theme_project(base_size = 9),
  "country_mean_ci.png",
  width = 10,
  height = 8.5
)

year_heterogeneity <- analysis_data |>
  dplyr::filter(!is.na(.data$female_lfpr)) |>
  dplyr::group_by(.data$Year) |>
  dplyr::summarise(
    Media = mean(.data$female_lfpr),
    N = dplyr::n(),
    SD = stats::sd(.data$female_lfpr),
    SE = .data$SD / sqrt(.data$N),
    `IC 95% inferior` = .data$Media - stats::qt(0.975, pmax(.data$N - 1, 1)) * .data$SE,
    `IC 95% superior` = .data$Media + stats::qt(0.975, pmax(.data$N - 1, 1)) * .data$SE,
    .groups = "drop"
  )
save_plot(
  ggplot2::ggplot(year_heterogeneity, ggplot2::aes(x = .data$Year, y = .data$Media)) +
    ggplot2::annotate("rect", xmin = 2008, xmax = 2009, ymin = -Inf, ymax = Inf, fill = BURGUNDY, alpha = 0.08) +
    ggplot2::annotate("rect", xmin = 2020, xmax = 2020, ymin = -Inf, ymax = Inf, fill = PLUM, alpha = 0.1) +
    ggplot2::annotate("rect", xmin = 2022, xmax = 2023, ymin = -Inf, ymax = Inf, fill = GOLD, alpha = 0.12) +
    ggplot2::geom_ribbon(ggplot2::aes(ymin = .data$`IC 95% inferior`, ymax = .data$`IC 95% superior`), fill = TEAL, alpha = 0.12) +
    ggplot2::geom_line(color = TEAL, linewidth = 1.1) +
    ggplot2::geom_point(color = NAVY, size = 2) +
    ggplot2::labs(title = "Evoluția mediei europene a participării feminine", x = "An", y = "Media europeană (%)") +
    theme_project(),
  "year_mean_ci.png",
  width = 10,
  height = 5.8
)

spaghetti_data <- analysis_data |> dplyr::filter(!is.na(.data$female_lfpr))
save_plot(
  ggplot2::ggplot(spaghetti_data, ggplot2::aes(x = .data$Year, y = .data$female_lfpr, group = .data$ISO3)) +
    ggplot2::geom_line(color = TEAL, alpha = 0.18, linewidth = 0.45) +
    ggplot2::geom_line(data = year_heterogeneity, ggplot2::aes(x = .data$Year, y = .data$Media, group = 1), color = GOLD, linewidth = 1.35, inherit.aes = FALSE) +
    ggplot2::labs(title = "Evoluția participării feminine în economiile europene", x = "An", y = "Participarea femeilor (%)") +
    theme_project(),
  "female_lfpr_spaghetti.png",
  width = 10,
  height = 5.8
)

key_years <- c(2001, 2007, 2009, 2019, 2020, 2023)
distribution_data <- analysis_data |>
  dplyr::filter(.data$Year %in% key_years, !is.na(.data$female_lfpr)) |>
  dplyr::mutate(An = factor(.data$Year, levels = key_years))
save_plot(
  ggplot2::ggplot(distribution_data, ggplot2::aes(x = .data$An, y = .data$female_lfpr)) +
    ggplot2::geom_violin(fill = TEAL, alpha = 0.22, color = TEAL) +
    ggplot2::geom_boxplot(width = 0.22, fill = IVORY, color = NAVY, outlier.color = BURGUNDY) +
    ggplot2::labs(title = "Distribuția participării feminine în ani reprezentativi", x = "An", y = "Participarea femeilor (%)") +
    theme_project(),
  "representative_years_distribution.png",
  width = 9,
  height = 5.8
)

main_data <- complete_model_data(analysis_data, c("female_lfpr", candidate_predictors))
pooled_model <- fit_panel(main_formula, main_data, "pooling")
fe_model <- fit_panel(main_formula, main_data, "within", "individual")
re_model <- fit_panel(main_formula, main_data, "random", "individual")
twoway_model <- fit_panel(main_formula, main_data, "within", "twoways")

write_csv_output(coefficient_table(pooled_model), "pooled_coefficients.csv")
write_csv_output(coefficient_table(fe_model), "fe_coefficients.csv")
write_csv_output(coefficient_table(re_model), "re_coefficients.csv")
write_csv_output(coefficient_table(twoway_model), "twoway_coefficients.csv")
saveRDS(pooled_model, file.path(MODEL_DIR, "pooled_model.rds"))
saveRDS(fe_model, file.path(MODEL_DIR, "fe_model.rds"))
saveRDS(re_model, file.path(MODEL_DIR, "re_model.rds"))
saveRDS(twoway_model, file.path(MODEL_DIR, "twoway_model.rds"))

metrics_main <- dplyr::bind_rows(
  model_metrics(pooled_model, main_data, candidate_predictors, "Pooled OLS"),
  model_metrics(fe_model, main_data, candidate_predictors, "Fixed Effects"),
  model_metrics(re_model, main_data, candidate_predictors, "Random Effects"),
  model_metrics(twoway_model, main_data, candidate_predictors, "Two-Way Fixed Effects")
)
write_csv_output(metrics_main, "panel_model_metrics.csv")

comparison_terms <- unique(c(
  coefficient_table(pooled_model)$Termen,
  coefficient_table(fe_model)$Termen,
  coefficient_table(re_model)$Termen
))
comparison_terms <- comparison_terms[comparison_terms != "Intercept"]
model_comparison <- tibble::tibble(Predictor = comparison_terms)
for (entry in list(Pooled = pooled_model, FE = fe_model, RE = re_model)) {
  model_name <- names(entry)
}
add_model_cols <- function(base, model, prefix) {
  table <- coefficient_table(model) |> dplyr::select("Termen", "Coeficient", "p-value")
  names(table) <- c("Predictor", paste0(prefix, " β"), paste0(prefix, " p"))
  dplyr::left_join(base, table, by = "Predictor")
}
model_comparison <- model_comparison |>
  add_model_cols(pooled_model, "Pooled") |>
  add_model_cols(fe_model, "FE") |>
  add_model_cols(re_model, "RE")
write_csv_output(model_comparison, "model_comparison.csv")

plot_coefficients <- dplyr::bind_rows(
  coefficient_table(pooled_model) |> dplyr::mutate(Model = "Pooled OLS"),
  coefficient_table(fe_model) |> dplyr::mutate(Model = "Fixed Effects"),
  coefficient_table(re_model) |> dplyr::mutate(Model = "Random Effects")
) |>
  dplyr::filter(.data$Termen != "Intercept")
save_plot(
  ggplot2::ggplot(plot_coefficients, ggplot2::aes(x = .data$Coeficient, y = stats::reorder(.data$Termen, .data$Coeficient), color = .data$Model)) +
    ggplot2::geom_vline(xintercept = 0, color = NAVY, linewidth = 0.6, alpha = 0.55) +
    ggplot2::geom_errorbar(ggplot2::aes(xmin = .data$`IC 95% inferior`, xmax = .data$`IC 95% superior`), position = ggplot2::position_dodge(width = 0.62), width = 0.18) +
    ggplot2::geom_point(position = ggplot2::position_dodge(width = 0.62), size = 2.4) +
    ggplot2::scale_color_manual(values = c("Pooled OLS" = GOLD, "Fixed Effects" = TEAL, "Random Effects" = BURGUNDY)) +
    ggplot2::labs(title = "Coeficienții estimați: Pooled OLS vs FE vs RE", x = "Coeficient", y = "") +
    theme_project(),
  "pooled_fe_re_coefficient_plot.png",
  width = 10.5,
  height = 7
)

observed_fitted_plot <- function(model, dataframe, title, filename) {
  frame <- actual_fitted_frame(model, dataframe)
  save_plot(
    ggplot2::ggplot(frame, ggplot2::aes(x = .data$fitted, y = .data$female_lfpr)) +
      ggplot2::geom_abline(slope = 1, intercept = 0, color = GOLD, linewidth = 0.8) +
      ggplot2::geom_point(color = TEAL, alpha = 0.58, size = 1.8) +
      ggplot2::labs(title = title, x = "Valori estimate", y = "Valori observate") +
      theme_project(),
    filename,
    width = 7.5,
    height = 5.2
  )
}
observed_fitted_plot(pooled_model, main_data, "Observed vs Fitted – Pooled", "observed_fitted_pooled.png")
observed_fitted_plot(fe_model, main_data, "Observed vs Fitted – FE", "observed_fitted_fe.png")
observed_fitted_plot(re_model, main_data, "Observed vs Fitted – RE", "observed_fitted_re.png")

safe_test <- function(expr) tryCatch(expr, error = function(e) NULL)
f_test <- safe_test(plm::pFtest(fe_model, pooled_model))
lm_test <- safe_test(plm::plmtest(pooled_model, type = "bp"))
hausman_test <- safe_test(plm::phtest(fe_model, re_model))
time_effects_test <- safe_test(plm::pFtest(twoway_model, fe_model))

f_test_table <- test_to_row(
  f_test,
  "Selecția modelului",
  "F test: FE vs Pooled",
  "Efectele individuale nu sunt necesare; modelul pooled este adecvat.",
  "Există dovezi privind heterogenitatea individuală, ceea ce favorizează modelul cu efecte fixe față de pooled OLS.",
  "Testul nu oferă suficiente dovezi pentru a prefera efectele fixe față de pooled OLS."
)
lm_test_table <- test_to_row(
  lm_test,
  "Selecția modelului",
  "Breusch-Pagan LM: RE vs Pooled",
  "Varianța efectului individual este zero; pooled OLS este suficient.",
  "Există heterogenitate panel semnificativă, ceea ce favorizează RE față de pooled OLS.",
  "Testul nu oferă suficiente dovezi pentru a prefera RE față de pooled OLS."
)
hausman_test_table <- test_to_row(
  hausman_test,
  "Selecția modelului",
  "Hausman: FE vs RE",
  "Estimatorul RE este consistent; diferențele dintre FE și RE nu sunt sistematice.",
  "Modelul FE este preferat modelului RE.",
  "Testul nu oferă suficiente dovezi pentru a prefera FE în detrimentul RE."
)
time_effects_test_table <- test_to_row(
  time_effects_test,
  "Selecția modelului",
  "Test efecte temporale: FE one-way vs two-way",
  "Efectele temporale comune nu aduc informație suplimentară.",
  "Efectele temporale comune aduc informație suplimentară pentru specificația panel.",
  "Testul nu oferă suficiente dovezi pentru includerea efectelor temporale comune."
)
write_csv_output(f_test_table, "f_test.csv")
write_csv_output(lm_test_table, "lm_test.csv")
write_csv_output(hausman_test_table, "hausman_test.csv")
write_csv_output(time_effects_test_table, "time_effects_test.csv")
write_csv_output(dplyr::bind_rows(f_test_table, lm_test_table, hausman_test_table, time_effects_test_table), "selection_tests.csv")

f_p <- f_test_table$`p-value`[[1]]
lm_p <- lm_test_table$`p-value`[[1]]
hausman_p <- hausman_test_table$`p-value`[[1]]
time_p <- time_effects_test_table$`p-value`[[1]]
selected_model_name <- "Pooled OLS"
selected_model_object <- pooled_model
if (!is.na(f_p) && f_p < 0.05) {
  selected_model_name <- "Fixed Effects"
  selected_model_object <- fe_model
}
if (!is.na(lm_p) && lm_p < 0.05 && (is.na(hausman_p) || hausman_p >= 0.05)) {
  selected_model_name <- "Random Effects"
  selected_model_object <- re_model
}
if (!is.na(hausman_p) && hausman_p < 0.05) {
  selected_model_name <- "Fixed Effects"
  selected_model_object <- fe_model
}
if (selected_model_name == "Fixed Effects" && !is.na(time_p) && time_p < 0.05) {
  selected_model_name <- "Two-Way Fixed Effects"
  selected_model_object <- twoway_model
}
selected_model_note <- paste0(
  "Specificația favorizată de testele panel este ", selected_model_name,
  ". Specificația este favorizată de testele aplicate și va fi supusă diagnosticului reziduurilor și verificărilor de robustețe."
)
write_csv_output(tibble::tibble(Componentă = "Specificația favorizată", Valoare = selected_model_name, Interpretare = selected_model_note), "selected_model.csv")

time_effects <- tryCatch(plm::fixef(twoway_model, effect = "time"), error = function(e) NULL)
if (is.null(time_effects)) {
  time_effects_table <- tibble::tibble(An = integer(), `Efect temporal` = numeric())
} else {
  time_effects_table <- tibble::tibble(An = as.integer(names(time_effects)), `Efect temporal` = as.numeric(time_effects))
}
write_csv_output(time_effects_table, "time_fixed_effects.csv")
if (nrow(time_effects_table) > 0) {
  save_plot(
    ggplot2::ggplot(time_effects_table, ggplot2::aes(x = .data$An, y = .data$`Efect temporal`)) +
      ggplot2::annotate("rect", xmin = 2008, xmax = 2009, ymin = -Inf, ymax = Inf, fill = BURGUNDY, alpha = 0.08) +
      ggplot2::annotate("rect", xmin = 2020, xmax = 2020, ymin = -Inf, ymax = Inf, fill = PLUM, alpha = 0.1) +
      ggplot2::annotate("rect", xmin = 2022, xmax = 2023, ymin = -Inf, ymax = Inf, fill = GOLD, alpha = 0.12) +
      ggplot2::geom_hline(yintercept = 0, color = NAVY, linewidth = 0.6) +
      ggplot2::geom_line(color = TEAL, linewidth = 1) +
      ggplot2::geom_point(color = BURGUNDY, size = 2) +
      ggplot2::labs(title = "Efectele temporale comune estimate", x = "An", y = "Efect temporal") +
      theme_project(),
    "time_fixed_effects.png",
    width = 9.5,
    height = 5.5
  )
}

selected_diag <- actual_fitted_frame(selected_model_object, main_data)
bp_diag <- safe_test(lmtest::bptest(selected_model_object))
pbg_diag <- safe_test(plm::pbgtest(selected_model_object))
wooldridge_diag <- safe_test(plm::pwartest(selected_model_object))
pesaran_diag <- safe_test(plm::pcdtest(selected_model_object, test = "cd"))
jb_diag <- safe_test(tseries::jarque.bera.test(selected_diag$residual))
shapiro_diag <- if (nrow(selected_diag) <= 5000) safe_test(stats::shapiro.test(selected_diag$residual)) else NULL

panel_diagnostics <- dplyr::bind_rows(
  test_to_row(bp_diag, "Heteroscedasticitate", "Breusch-Pagan", "Varianța erorilor este constantă.", "Există dovezi privind heteroscedasticitatea reziduurilor.", "Testul nu oferă suficiente dovezi privind heteroscedasticitatea."),
  test_to_row(pbg_diag, "Autocorelare", "Breusch-Godfrey panel", "Reziduurile nu prezintă autocorelare serială.", "Există dovezi privind autocorelarea serială în reziduuri.", "Testul nu oferă suficiente dovezi privind autocorelarea serială."),
  test_to_row(wooldridge_diag, "Autocorelare", "Wooldridge", "Reziduurile nu prezintă autocorelare serială în panel.", "Există dovezi privind autocorelarea serială în panel.", "Testul nu oferă suficiente dovezi privind autocorelarea serială în panel."),
  test_to_row(pesaran_diag, "Dependență transversală", "Pesaran CD", "Reziduurile sunt independente între țări.", "Există dovezi privind dependența transversală între economiile europene. Șocurile comune și integrarea economică pot determina corelarea reziduurilor între țări.", "Testul nu oferă suficiente dovezi privind dependența transversală între țări."),
  test_to_row(jb_diag, "Normalitate", "Jarque-Bera", "Reziduurile urmează o distribuție normală.", "Testul indică abatere de la normalitate.", "Testul nu oferă suficiente dovezi privind abaterea de la normalitate."),
  test_to_row(shapiro_diag, "Normalitate", "Shapiro-Wilk", "Reziduurile urmează o distribuție normală.", "Testul indică abatere de la normalitate.", "Testul nu oferă suficiente dovezi privind abaterea de la normalitate."),
  tibble::tibble(
    Categorie = "Normalitate",
    Test = c("Skewness", "Kurtosis"),
    `Ipoteza nulă` = c("Indicator descriptiv", "Indicator descriptiv"),
    Statistică = c(moments::skewness(selected_diag$residual), moments::kurtosis(selected_diag$residual)),
    df = NA_character_,
    `p-value` = NA_real_,
    Decizie = "Indicator descriptiv",
    Interpretare = c("Skewness descrie asimetria reziduurilor.", "Kurtosis este raportată în convenția Pearson.")
  )
)
write_csv_output(panel_diagnostics, "panel_diagnostics.csv")

vif_lm <- fit_lm_aux(main_formula, main_data)
vif_values <- tryCatch(car::vif(vif_lm), error = function(e) NULL)
if (is.null(vif_values)) {
  vif_panel <- tibble::tibble(Predictor = "VIF indisponibil", VIF = NA_real_, Interpretare = "Modelul auxiliar nu a permis calculul VIF.")
} else {
  if (is.matrix(vif_values)) vif_values <- vif_values[, 1]
  vif_panel <- tibble::tibble(
    Predictor = vapply(names(vif_values), label_for, character(1)),
    VIF = as.numeric(vif_values),
    Interpretare = dplyr::case_when(
      VIF < 5 ~ "risc redus/moderat",
      VIF < 10 ~ "atenție",
      TRUE ~ "risc sever"
    )
  )
}
write_csv_output(vif_panel, "vif_panel.csv")
save_plot(
  ggplot2::ggplot(vif_panel |> dplyr::filter(!is.na(.data$VIF)), ggplot2::aes(x = stats::reorder(.data$Predictor, .data$VIF), y = .data$VIF, fill = .data$Interpretare)) +
    ggplot2::geom_col(alpha = 0.9) +
    ggplot2::coord_flip() +
    ggplot2::scale_fill_manual(values = c("risc redus/moderat" = TEAL, "atenție" = GOLD, "risc sever" = BURGUNDY)) +
    ggplot2::labs(title = "VIF pentru predictorii modelului panel", x = "", y = "VIF") +
    theme_project(),
  "vif_panel_bar.png",
  width = 9.5,
  height = 6
)

save_residual_plot <- function(plot, filename, width = 9, height = 5.5) save_plot(plot, filename, width, height)
extreme_countries <- selected_diag |>
  dplyr::group_by(.data$country) |>
  dplyr::summarise(median_resid = stats::median(.data$residual, na.rm = TRUE), .groups = "drop") |>
  dplyr::arrange(dplyr::desc(abs(.data$median_resid))) |>
  dplyr::slice_head(n = 8)

save_residual_plot(
  ggplot2::ggplot(selected_diag, ggplot2::aes(x = .data$fitted, y = .data$residual)) +
    ggplot2::geom_hline(yintercept = 0, color = GOLD, linewidth = 0.8) +
    ggplot2::geom_point(color = TEAL, alpha = 0.55, size = 1.7) +
    ggplot2::geom_smooth(method = "loess", formula = y ~ x, se = FALSE, color = BURGUNDY, linewidth = 0.9) +
    ggplot2::labs(title = "Residuals vs Fitted", x = "Valori estimate", y = "Reziduuri") +
    theme_project(),
  "panel_residuals_fitted.png"
)
save_residual_plot(
  ggplot2::ggplot(selected_diag, ggplot2::aes(sample = .data$std_residual)) +
    ggplot2::stat_qq(color = TEAL, alpha = 0.75, size = 1.8) +
    ggplot2::stat_qq_line(color = BURGUNDY, linewidth = 0.9) +
    ggplot2::labs(title = "Q-Q Plot", x = "Cuantile teoretice", y = "Cuantile eșantion") +
    theme_project(),
  "panel_qq_plot.png"
)
save_residual_plot(
  ggplot2::ggplot(selected_diag, ggplot2::aes(x = .data$residual)) +
    ggplot2::geom_histogram(ggplot2::aes(y = ggplot2::after_stat(density)), bins = 28, fill = TEAL, alpha = 0.72, color = "white") +
    ggplot2::geom_density(color = BURGUNDY, linewidth = 1) +
    ggplot2::labs(title = "Histogramă reziduuri", x = "Reziduuri", y = "Densitate") +
    theme_project(),
  "panel_residual_histogram.png"
)
save_residual_plot(
  ggplot2::ggplot(selected_diag, ggplot2::aes(x = "", y = .data$residual)) +
    ggplot2::geom_boxplot(fill = GOLD, color = NAVY, alpha = 0.55, width = 0.32) +
    ggplot2::geom_jitter(color = TEAL, width = 0.08, alpha = 0.45, size = 1.5) +
    ggplot2::labs(title = "Box plot reziduuri", x = "", y = "Reziduuri") +
    theme_project(),
  "panel_residual_boxplot.png",
  width = 7,
  height = 5
)
save_residual_plot(
  ggplot2::ggplot(selected_diag, ggplot2::aes(x = .data$Year, y = .data$residual, group = .data$ISO3)) +
    ggplot2::geom_line(color = TEAL, alpha = 0.16, linewidth = 0.35) +
    ggplot2::stat_summary(ggplot2::aes(group = 1), fun = mean, geom = "line", color = GOLD, linewidth = 1.25) +
    ggplot2::geom_hline(yintercept = 0, color = NAVY, linewidth = 0.5) +
    ggplot2::labs(title = "Reziduuri în timp", x = "An", y = "Reziduuri") +
    theme_project(),
  "panel_residuals_time.png"
)
country_resid_order <- selected_diag |>
  dplyr::group_by(.data$country) |>
  dplyr::summarise(median_resid = median(.data$residual, na.rm = TRUE), .groups = "drop")
selected_diag <- selected_diag |>
  dplyr::left_join(country_resid_order, by = "country")
save_residual_plot(
  ggplot2::ggplot(selected_diag, ggplot2::aes(x = stats::reorder(.data$country, .data$median_resid), y = .data$residual)) +
    ggplot2::geom_boxplot(fill = IVORY, color = NAVY, outlier.color = BURGUNDY, alpha = 0.92) +
    ggplot2::coord_flip() +
    ggplot2::labs(title = "Reziduuri pe țări", x = "", y = "Reziduuri") +
    theme_project(base_size = 8),
  "panel_residuals_by_country.png",
  width = 9.5,
  height = 8.2
)
year_residuals <- selected_diag |>
  dplyr::group_by(.data$Year) |>
  dplyr::summarise(`Media reziduurilor` = mean(.data$residual, na.rm = TRUE), .groups = "drop")
save_residual_plot(
  ggplot2::ggplot(year_residuals, ggplot2::aes(x = .data$Year, y = .data$`Media reziduurilor`)) +
    ggplot2::geom_hline(yintercept = 0, color = NAVY, linewidth = 0.5) +
    ggplot2::geom_line(color = TEAL, linewidth = 1.1) +
    ggplot2::geom_point(color = BURGUNDY, size = 2) +
    ggplot2::labs(title = "Media reziduurilor pe an", x = "An", y = "Media reziduurilor") +
    theme_project(),
  "panel_residuals_year_mean.png"
)
save_residual_plot(
  ggplot2::ggplot(selected_diag, ggplot2::aes(x = .data$fitted, y = .data$female_lfpr)) +
    ggplot2::geom_abline(slope = 1, intercept = 0, color = GOLD, linewidth = 0.8) +
    ggplot2::geom_point(color = TEAL, alpha = 0.55, size = 1.7) +
    ggplot2::labs(title = "Observed vs Fitted", x = "Valori estimate", y = "Valori observate") +
    theme_project(),
  "panel_observed_fitted_final.png"
)

cluster_vcov <- tryCatch(plm::vcovHC(selected_model_object, type = "HC1", cluster = "group"), error = function(e) NULL)
dk_vcov <- tryCatch(plm::vcovSCC(selected_model_object, type = "HC1", maxlag = 3), error = function(e) NULL)
cluster_coefficients <- coefficient_table(selected_model_object, cluster_vcov)
dk_coefficients <- coefficient_table(selected_model_object, dk_vcov)
write_csv_output(cluster_coefficients, "robust_cluster_coefficients.csv")
write_csv_output(dk_coefficients, "driscoll_kraay_coefficients.csv")

conventional_selected <- coefficient_table(selected_model_object)
robust_inference <- conventional_selected |>
  dplyr::select("Termen", "term_raw", β = "Coeficient", `SE convențional` = "Standard error", `p convențional` = "p-value") |>
  dplyr::left_join(cluster_coefficients |> dplyr::select("term_raw", `SE cluster` = "Standard error", `p cluster` = "p-value"), by = "term_raw") |>
  dplyr::left_join(dk_coefficients |> dplyr::select("term_raw", `SE Driscoll–Kraay` = "Standard error", `p Driscoll–Kraay` = "p-value"), by = "term_raw") |>
  dplyr::select(-"term_raw")
write_csv_output(robust_inference, "robust_inference_comparison.csv")

serial_detected <- any(panel_diagnostics$Categorie == "Autocorelare" & !is.na(panel_diagnostics$`p-value`) & panel_diagnostics$`p-value` < 0.05)
cross_detected <- any(panel_diagnostics$Categorie == "Dependență transversală" & !is.na(panel_diagnostics$`p-value`) & panel_diagnostics$`p-value` < 0.05)
heteroskedasticity_detected <- any(panel_diagnostics$Categorie == "Heteroscedasticitate" & !is.na(panel_diagnostics$`p-value`) & panel_diagnostics$`p-value` < 0.05)
robust_method_used <- if (!is.null(dk_vcov) && (serial_detected || cross_detected)) "Driscoll–Kraay" else "Cluster pe țară"
robust_for_plot <- if (robust_method_used == "Driscoll–Kraay" && nrow(dk_coefficients) > 0) dk_coefficients else cluster_coefficients
save_plot(
  ggplot2::ggplot(robust_for_plot |> dplyr::filter(.data$Termen != "Intercept"), ggplot2::aes(x = .data$Coeficient, y = stats::reorder(.data$Termen, .data$Coeficient))) +
    ggplot2::geom_vline(xintercept = 0, color = BURGUNDY, linewidth = 0.8) +
    ggplot2::geom_errorbar(ggplot2::aes(xmin = .data$`IC 95% inferior`, xmax = .data$`IC 95% superior`), width = 0.18, color = NAVY) +
    ggplot2::geom_point(color = TEAL, size = 2.7) +
    ggplot2::labs(title = "Coeficienții modelului final și intervalele de încredere robuste", subtitle = robust_method_used, x = "Coeficient", y = "") +
    theme_project(),
  "robust_coefficient_plot.png",
  width = 10,
  height = 7
)

country_effects <- tryCatch(plm::fixef(fe_model, effect = "individual"), error = function(e) NULL)
if (is.null(country_effects)) {
  country_fixed_effects <- tibble::tibble(ISO3 = character(), `Efect fix` = numeric())
} else {
  country_fixed_effects <- tibble::tibble(ISO3 = names(country_effects), `Efect fix` = as.numeric(country_effects)) |>
    dplyr::left_join(europe_countries, by = c("ISO3" = "iso3")) |>
    dplyr::select(Țară = "country", "ISO3", "Efect fix")
}
write_csv_output(country_fixed_effects, "country_fixed_effects.csv")
if (nrow(country_fixed_effects) > 0) {
  save_plot(
    ggplot2::ggplot(country_fixed_effects, ggplot2::aes(x = .data$`Efect fix`, y = stats::reorder(.data$Țară, .data$`Efect fix`))) +
      ggplot2::geom_vline(xintercept = 0, color = GOLD, linewidth = 0.8) +
      ggplot2::geom_point(color = TEAL, size = 2.3) +
      ggplot2::labs(title = "Efectele fixe estimate pe țări", x = "Efect fix", y = "") +
      theme_project(base_size = 9),
    "country_fixed_effects.png",
    width = 9.5,
    height = 8
  )
}

thematic_specs <- list(
  "Model A – Macroeconomic" = c("LogGDP", "gdp_growth", "inflation", "female_unemployment"),
  "Model B – Demografie și capital uman" = c("LogGDP", "fertility", "urbanization", "child_dependency", "female_tertiary"),
  "Model C – Structura pieței muncii și digitalizare" = c("LogGDP", "female_services", "female_vulnerable", "internet"),
  "Model D – Instituții" = c("LogGDP", "women_parliament", "control_corruption"),
  "Model E – Model complet candidat" = candidate_predictors
)
thematic_results <- list()
thematic_summary <- list()
for (spec_name in names(thematic_specs)) {
  predictors <- thematic_specs[[spec_name]]
  spec_data <- complete_model_data(analysis_data, c("female_lfpr", predictors))
  spec_formula <- stats::as.formula(paste("female_lfpr ~", paste(predictors, collapse = " + ")))
  spec_model <- fit_panel(spec_formula, spec_data, "within", "individual")
  thematic_results[[spec_name]] <- coefficient_table(spec_model) |> dplyr::mutate(Model = spec_name)
  thematic_summary[[spec_name]] <- model_metrics(spec_model, spec_data, predictors, spec_name) |>
    dplyr::mutate(Predictori_text = paste(vapply(predictors, label_for, character(1)), collapse = ", "))
}
thematic_models_summary <- dplyr::bind_rows(thematic_summary) |>
  dplyr::select("Model", Predictori = "Predictori_text", "N", "Țări", "R² relevant", "AIC", "BIC")
write_csv_output(thematic_models_summary, "thematic_models_summary.csv")

stability_data <- dplyr::bind_rows(thematic_results) |>
  dplyr::filter(.data$Termen != "Intercept")
save_plot(
  ggplot2::ggplot(stability_data, ggplot2::aes(x = .data$Coeficient, y = .data$Termen, color = .data$Model)) +
    ggplot2::geom_vline(xintercept = 0, color = NAVY, alpha = 0.45) +
    ggplot2::geom_point(size = 2.1, alpha = 0.9, position = ggplot2::position_dodge(width = 0.55)) +
    ggplot2::scale_color_manual(values = c(TEAL, GOLD, BURGUNDY, PLUM, NAVY)) +
    ggplot2::labs(title = "Stabilitatea coeficienților între specificații", x = "Coeficient FE", y = "") +
    theme_project(base_size = 10) +
    ggplot2::theme(legend.position = "bottom"),
  "coefficient_stability_specs.png",
  width = 11,
  height = 7.5
)

subperiods <- list("2001–2007" = c(2001, 2007), "2008–2019" = c(2008, 2019), "2020–2023" = c(2020, 2023))
subperiod_rows <- list()
subperiod_coef <- list()
for (period_name in names(subperiods)) {
  years <- subperiods[[period_name]]
  period_raw <- analysis_data |> dplyr::filter(.data$Year >= years[[1]], .data$Year <= years[[2]])
  period_data <- complete_model_data(period_raw, c("female_lfpr", candidate_predictors))
  period_model <- if (dplyr::n_distinct(period_data$Year) >= 4 && nrow(period_data) > length(candidate_predictors) + dplyr::n_distinct(period_data$ISO3)) {
    fit_panel(main_formula, period_data, "within", "individual")
  } else {
    NULL
  }
  if (is.null(period_model)) {
    subperiod_rows[[period_name]] <- tibble::tibble(Perioadă = period_name, N = nrow(period_data), Țări = dplyr::n_distinct(period_data$ISO3), Ani = dplyr::n_distinct(period_data$Year), Status = "Specificația nu este estimabilă stabil pentru această subperioadă.")
  } else {
    r2 <- extract_r2(period_model)
    subperiod_rows[[period_name]] <- tibble::tibble(Perioadă = period_name, N = nrow(period_data), Țări = dplyr::n_distinct(period_data$ISO3), Ani = dplyr::n_distinct(period_data$Year), Status = "Estimat", `R² relevant` = r2[["rsq"]])
    subperiod_coef[[period_name]] <- coefficient_table(period_model) |> dplyr::mutate(Perioadă = period_name)
  }
}
subperiod_models_summary <- dplyr::bind_rows(subperiod_rows)
write_csv_output(subperiod_models_summary, "subperiod_models_summary.csv")
subperiod_plot_data <- dplyr::bind_rows(subperiod_coef) |> dplyr::filter(.data$Termen != "Intercept")
if (nrow(subperiod_plot_data) > 0) {
  save_plot(
    ggplot2::ggplot(subperiod_plot_data, ggplot2::aes(x = .data$Coeficient, y = .data$Termen, color = .data$Perioadă)) +
      ggplot2::geom_vline(xintercept = 0, color = NAVY, alpha = 0.45) +
      ggplot2::geom_errorbar(ggplot2::aes(xmin = .data$`IC 95% inferior`, xmax = .data$`IC 95% superior`), width = 0.16, position = ggplot2::position_dodge(width = 0.55)) +
      ggplot2::geom_point(size = 2, position = ggplot2::position_dodge(width = 0.55)) +
      ggplot2::scale_color_manual(values = c(TEAL, GOLD, BURGUNDY)) +
      ggplot2::labs(title = "Stabilitatea coeficienților în subperioade", x = "Coeficient FE", y = "") +
      theme_project(base_size = 10),
    "coefficient_stability_subperiods.png",
    width = 11,
    height = 7.5
  )
}

panel_summary_text <- paste0(
  "Panelul este ", if (balanced_panel) "echilibrat" else "neechilibrat",
  ", cu ", dplyr::n_distinct(analysis_data$ISO3), " economii și ", nrow(analysis_data), " observații country-year. ",
  "Testele de selecție favorizează specificația ", selected_model_name, ". ",
  if (!is.na(time_p) && time_p < 0.05) "Efectele temporale comune sunt susținute de testele aplicate. " else "Testele nu indică necesitatea clară a efectelor temporale comune. ",
  if (heteroskedasticity_detected) "Diagnosticul indică heteroscedasticitate. " else "Diagnosticul nu oferă dovezi clare de heteroscedasticitate. ",
  if (serial_detected) "Există dovezi privind autocorelarea serială. " else "Nu apar dovezi clare privind autocorelarea serială. ",
  if (cross_detected) "Există dovezi privind dependența transversală. " else "Nu apar dovezi clare privind dependența transversală. ",
  "Inferența robustă utilizată pentru interpretarea finală este: ", robust_method_used,
  ". Rezultatele descriu asocieri statistice în date panel și nu demonstrează automat relații cauzale."
)
write_csv_output(tibble::tibble(Secțiune = "Concluzii ale analizei panel", Text = panel_summary_text), "panel_final_summary.csv")

interpretation_rows <- robust_for_plot |>
  dplyr::filter(.data$Termen != "Intercept") |>
  dplyr::mutate(
    Interpretare = dplyr::case_when(
      .data$term_raw == "LogGDP" ~ paste0("În interiorul aceleiași economii, o creștere aproximativă cu 1% a PIB/locuitor este asociată cu o modificare estimată de ", round(.data$Coeficient / 100, 3), " puncte procentuale a participării feminine, după controlul celorlalte variabile și al efectelor fixe incluse."),
      TRUE ~ paste0("În interiorul aceleiași economii, o creștere cu o unitate a variabilei ", .data$Termen, " este asociată cu o modificare estimată de ", round(.data$Coeficient, 3), " puncte procentuale a participării feminine, celelalte condiții incluse în model fiind menținute constante.")
    )
  ) |>
  dplyr::select("Termen", "Coeficient", "p-value", "Interpretare")
write_csv_output(interpretation_rows, "panel_coefficient_interpretations.csv")

metadata <- list(
  start_year = START_YEAR,
  end_year = END_YEAR,
  number_countries = dplyr::n_distinct(analysis_data$ISO3),
  number_observations = nrow(analysis_data),
  balanced = balanced_panel,
  Tmin = panel_index_counts$Tmin[[1]],
  Tmax = panel_index_counts$Tmax[[1]],
  selected_model = selected_model_name,
  selected_model_note = selected_model_note,
  time_effects_required = !is.na(time_p) && time_p < 0.05,
  heteroskedasticity_detected = heteroskedasticity_detected,
  serial_correlation_detected = serial_detected,
  cross_sectional_dependence_detected = cross_detected,
  robust_method_used = robust_method_used,
  timestamp = format(Sys.time(), "%Y-%m-%d %H:%M:%S %Z"),
  conclusion = panel_summary_text,
  notes = list(
    panel_method = "Spre deosebire de analiza transversală, modelele panel utilizează atât dimensiunea spațială, cât și dimensiunea temporală a datelor.",
    fe = "Modelul FE utilizează variația în interiorul fiecărei economii și controlează caracteristicile constante în timp care diferă între țări.",
    re = "Modelul RE presupune că efectul specific fiecărei țări nu este corelat cu predictorii incluși.",
    robust = "Erorile standard robuste modifică inferența statistică – erorile standard, statisticile și valorile p – nu coeficienții estimați ai modelului.",
    normality = "În panelurile cu un număr mare de observații, inferența modelelor FE/RE se bazează mai mult pe specificarea corectă și pe erori standard robuste decât pe normalitatea strictă a reziduurilor.",
    fixed_effects = "Efectele fixe surprind caracteristici neobservate ale economiilor care sunt relativ constante în timp și nu trebuie interpretate drept performanță sau clasament."
  )
)
jsonlite::write_json(metadata, file.path(META_DIR, "panel_analysis.json"), pretty = TRUE, auto_unbox = TRUE, na = "null")

message("Analiza panel R s-a încheiat cu succes.")

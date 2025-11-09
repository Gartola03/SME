################################################################################
# Script: Data Preprocessing, Metrics, and Association Analysis
# ******************************************************************************
# Descripción:
#   Este script proporciona funciones para:
#     1. Discretizar variables numéricas (intervalos iguales y frecuencias iguales)
#     2. Calcular métricas del conjunto de datos (entropía, varianza, AUC)
#     3. Normalizar y estandarizar datos numéricos
#     4. Filtrar variables según métricas
#     5. Calcular matrices de asociación por pares
#     6. Representar gráficamente métricas y matrices de asociación
#
# Autor: Garikoitz Artola Obando (gartola008@ikasle.ehu.eus)
# Fecha: 09/11/2025
################################################################################


# LOAD LIBRARIES ---------------------------------------------------------------
library(ggplot2)
library(reshape2)



# AUXILIAR FUNCTIONS -----------------------------------------------------------
discretizeEW <- function(x, num.bins) {
  if (!is.numeric(x)) stop("discretizeEW: x must be numeric")
  if (!is.numeric(num.bins) || length(num.bins) != 1) stop("discretizeEW: num.bins must be single numeric")
  if (num.bins < 1) stop("discretizeEW: num.bins must be >= 1")
  if (all(is.na(x))) return(list(factor = factor(x), cut.points = numeric(0)))
  finite.x <- x[is.finite(x)]
  if (length(finite.x) == 0) return(list(factor = factor(x), cut.points = numeric(0)))
  minx <- min(finite.x)
  maxx <- max(finite.x)

  if (minx == maxx) { # all same value
    cut.points <- numeric(0)
    x.discretized <- factor(rep("single_value", length(x)), levels = "single_value")
    return(list(factor = x.discretized, cut.points = cut.points))
  }

  # create breakpoints (num.bins equal width)
  breaks <- seq(from = minx, to = maxx, length.out = num.bins + 1)
  # internal cut points excludes endpoints
  cut.points <- sort(unique(breaks))[ - c(1, length(breaks)) ]
  intervalos <- c(-Inf, cut.points, Inf)
  x.discretized <- cut(x, breaks = intervalos, include.lowest = TRUE, right = FALSE)
  return(list(factor = x.discretized, cut.points = cut.points))
}


discretizeEF <- function(x, num.bins) {
  if (!is.numeric(x)) stop("discretizeEF: x must be numeric")
  if (!is.numeric(num.bins) || length(num.bins) != 1) stop("discretizeEF: num.bins must be single numeric")
  if (num.bins < 1) stop("discretizeEF: num.bins must be >= 1")
  if (all(is.na(x))) return(list(factor = factor(x), cut.points = numeric(0)))
  n <- sum(!is.na(x))
  if (n == 0) return(list(factor = factor(x), cut.points = numeric(0)))

  # compute number per bin (some bins may have one additional element)
  num.perBin <- n %/% num.bins
  rest.perBin <- n %% num.bins

  # sizes distribution from smallest to largest is fine but we will build cutpoints from sorted x
  bin.sizes <- c(rep(num.perBin + 1, rest.perBin), rep(num.perBin, num.bins - rest.perBin))
  sorted_idx <- order(x, na.last = TRUE)
  sorted_x <- x[sorted_idx]

  # cumulative sizes to pick cut values (end of each bin except last)
  cum.sizes <- cumsum(bin.sizes)
  cut.indices <- cum.sizes[1:(length(cum.sizes) - 1)]
  # if there are ties that span cut indices, it's okay: we pick the observed value at that index
  cut.points <- unique(sorted_x[cut.indices])
  intervalos <- c(-Inf, cut.points, Inf)
  x.discretized <- cut(x, breaks = intervalos, include.lowest = TRUE, right = FALSE)
  return(list(factor = x.discretized, cut.points = cut.points))
}


entropy <- function(x, base = 2, na.rm = TRUE) {
  if (na.rm) x <- x[!is.na(x)]
  if (length(x) == 0) return(0)
  probs <- table(x) / length(x)
  probs <- probs[probs > 0]
  H <- -sum(probs * log(probs, base = base))
  return(as.numeric(H))
}


# joint entropy for two discrete variables
joint_entropy <- function(x, y, base = 2, na.rm = TRUE) {
  if (na.rm) {
    ok <- !(is.na(x) | is.na(y))
    x <- x[ok]; y <- y[ok]
  }
  if (length(x) == 0) return(0)
  tab <- table(x, y)
  probs <- tab / sum(tab)
  probs <- probs[probs > 0]
  H <- -sum(probs * log(probs, base = base))
  return(as.numeric(H))
}


# mutual information I(X;Y) = H(X)+H(Y)-H(X,Y)
mutual_information <- function(x, y, base = 2, na.rm = TRUE) {
  Hx <- entropy(x, base = base, na.rm = na.rm)
  Hy <- entropy(y, base = base, na.rm = na.rm)
  Hxy <- joint_entropy(x, y, base = base, na.rm = na.rm)
  MI <- Hx + Hy - Hxy
  return(as.numeric(MI))
}


compute_auc_numeric <- function(x, class) {
  # class must be TRUE/FALSE or convert to logical
  if (is.numeric(class) || is.factor(class))
    class <- as.logical(as.numeric(as.factor(class)) - 1)

  # Remove NA
  ok <- !(is.na(x) | is.na(class))
  x <- x[ok]
  class <- class[ok]

  # Sort by predictor value
  o <- order(x)
  x <- x[o]
  class <- class[o]

  # total positives / negatives
  P <- sum(class)
  N <- sum(!class)

  if (P == 0 || N == 0)
    stop("compute_auc_numeric: class must have both TRUE and FALSE")

  # All possible cutpoints = each unique value
  cuts <- unique(x)
  TPR <- numeric(length(cuts))
  FPR <- numeric(length(cuts))
  for (i in seq_along(cuts)) {
    cutoff <- cuts[i]

    # prediction: x >= cutoff is TRUE
    pred <- x >= cutoff

    TP <- sum(pred &  class)
    FP <- sum(pred & !class)
    FN <- sum(!pred &  class)
    TN <- sum(!pred & !class)

    TPR[i] <- TP / (TP + FN)
    FPR[i] <- FP / (FP + TN)
  }

  # Add starting point (0,0) and ending point (1,1)
  FPR <- c(0, FPR, 1)
  TPR <- c(0, TPR, 1)

  # Compute area using trapezoids
  auc <- sum( diff(FPR) * (head(TPR, -1) + tail(TPR, -1)) / 2 )
  return(auc)
}



# FUNCTIONS --------------------------------------------------------------------

# 1. DISCRETIZATION ------------------------------------------------------------
dis.attribute <- function(attribute, num.bin, mode = FALSE) {
  if (!is.numeric(attribute)) stop("Error: attribute is not numeric")
  if (!is.numeric(num.bin) || length(num.bin) != 1) stop("Error: num.bin must be a single numeric")
  if (!is.logical(mode) || length(mode) != 1) stop("Error: mode must be boolean (logical)")
  if (mode == FALSE) {
    return(discretizeEW(attribute, num.bin))
  } else {
    return(discretizeEF(attribute, num.bin))
  }
}


dis.dataset <- function(dataset, num.bin, mode = FALSE) {
  # --- Error checking ---
  if (!is.data.frame(dataset)) stop("Error: dataset must be a data.frame")
  if (!is.numeric(num.bin) || length(num.bin) != 1) stop("Error: num.bin must be a single numeric")
  if (!is.logical(mode) || length(mode) != 1) stop("Error: mode must be boolean (logical)")

  # Copy to avoid modifying in place
  res.df <- dataset
  # Select numeric columns
  numeric.cols <- sapply(dataset, is.numeric)

  # Loop over numeric columns and discretize
  for (colname in names(dataset)[numeric.cols]) {
    if (mode == FALSE) {
      d <- discretizeEW(dataset[[colname]], num.bin)
    } else {
      d <- discretizeEF(dataset[[colname]], num.bin)
    }
    res.df[[colname]] <- d$factor   # replace column with discretized factor
  }
  return(res.df)
}



# 2. METRICS -------------------------------------------------------------------
metrics_dataset <- function(dataset, class.var = NULL, discretize_for_auc = FALSE, bins = 10, verbose = FALSE) {
  if (!is.data.frame(dataset)) stop("dataset must be a data.frame")

  if (!is.null(class.var) && !(class.var %in% names(dataset))) stop("metrics_dataset: class.var not found")

  class.vec <- if (!is.null(class.var)) dataset[[class.var]] else NULL

  result <- data.frame(
    variable = character(0),
    type = character(0),
    variance = numeric(0),
    entropy = numeric(0),
    auc = numeric(0),
    stringsAsFactors = FALSE
  )

  for (nm in names(dataset)) {
    if (!is.null(class.var) && nm == class.var) next

    col <- dataset[[nm]]

    if (is.numeric(col)) {
      varv <- ifelse(all(is.na(col)), NA_real_, var(col, na.rm = TRUE))
      ent <- NA_real_
      aucv <- NA_real_

      if (!is.null(class.vec)) {
        tmp <- try({
          aucv <- compute_auc_numeric(col, class.vec)
        }, silent = TRUE)

        if (inherits(tmp, "try-error") && verbose)
          message("AUC failed for ", nm)
      }

      typ <- "numeric"

    } else {
      varv <- NA_real_
      ent <- entropy(col)
      aucv <- NA_real_
      typ <- "categorical"
    }

    result <- rbind(
      result,
      data.frame(variable = nm, type = typ, variance = varv, entropy = ent, auc = aucv)
    )
  }
  rownames(result) <- NULL
  return(result)
}



# 3. NORMALIZATION AND STANDARDIZATION -----------------------------------------
normalize_vector <- function(v, na.rm = TRUE) {
  if (!is.numeric(v)) stop("normalize_vector: v must be numeric")
  if (na.rm) v2 <- v[is.finite(v)] else v2 <- v
  if (length(v2) == 0) return(rep(NA_real_, length(v)))
  minv <- min(v2)
  maxv <- max(v2)
  denom <- (maxv - minv)
  if (denom == 0) return(rep(0, length(v))) # all equal -> zero vector
  out <- (v - minv) / denom
  return(out)
}


standardize_vector <- function(v, na.rm = TRUE) {
  if (!is.numeric(v)) stop("standardize_vector: v must be numeric")
  if (na.rm) v2 <- v[is.finite(v)] else v2 <- v
  if (length(v2) == 0) return(rep(NA_real_, length(v)))
  m <- mean(v2)
  s <- sd(v2)
  if (s == 0) return(rep(0, length(v)))
  out <- (v - m) / s
  return(out)
}


normalize_dataset <- function(df, na.rm = TRUE) {
  if (!is.data.frame(df)) stop("normalize_dataset: df must be data.frame")
  out <- df
  for (nm in names(df)) {
    if (is.numeric(df[[nm]])) out[[nm]] <- normalize_vector(df[[nm]], na.rm = na.rm)
  }
  return(out)
}


standardize_dataset <- function(df, na.rm = TRUE) {
  if (!is.data.frame(df)) stop("standardize_dataset: df must be data.frame")
  out <- df
  for (nm in names(df)) {
    if (is.numeric(df[[nm]])) out[[nm]] <- standardize_vector(df[[nm]], na.rm = na.rm)
  }
  return(out)
}



# 4. FILTERING -----------------------------------------------------------------
filter_by_metric <- function(dataset, metric = c("entropy", "variance", "auc"), threshold, class.var = NULL, bins = 10) {
  metric <- match.arg(metric)
  if (!is.data.frame(dataset)) stop("filter_by_metric: dataset must be a data.frame")
  if (!is.numeric(threshold) || length(threshold) != 1) stop("filter_by_metric: threshold must be single numeric")
  # compute metrics
  met.df <- metrics_dataset(dataset, class.var = class.var)
  keep.vars <- character(0)
  if (metric == "entropy") {
    # entropy available for categorical only
    keep.vars <- met.df$variable[!is.na(met.df$entropy) & met.df$entropy >= threshold]
  } else if (metric == "variance") {
    keep.vars <- met.df$variable[!is.na(met.df$variance) & met.df$variance >= threshold]
  } else if (metric == "auc") {
    if (is.null(class.var)) stop("filter_by_metric: class.var must be provided to compute AUC")
    keep.vars <- met.df$variable[!is.na(met.df$auc) & met.df$auc >= threshold]
  }
  # return dataset that contains class.var (if present) + kept variables
  final.vars <- intersect(c(keep.vars, class.var), names(dataset))
  if (length(final.vars) == 0) {
    warning("filter_by_metric: no variables meet the threshold; returning empty data.frame with zero columns")
    return(dataset[ , FALSE, drop = FALSE])
  }
  return(dataset[, final.vars, drop = FALSE])
}



# 5. PAIRWISE ASSOCIATION MATRIX -----------------------------------------------
pairwise_assoc_matrix <- function(dataset, num.bin = 10, normalize_mi = TRUE, mode = FALSE, na.rm = TRUE) {
  if (!is.data.frame(dataset)) stop("pairwise_assoc_matrix: dataset must be data.frame")

  n <- ncol(dataset)
  names_cols <- names(dataset)
  mat <- matrix(NA_real_, nrow = n, ncol = n, dimnames = list(names_cols, names_cols))

  # Determine column types
  types <- sapply(dataset, function(x) if (is.numeric(x)) "numeric" else "categorical")

  # Discretize numeric columns once using existing functions
  dataset_discretized <- dis.dataset(dataset, num.bin = num.bin, mode = mode)

  # Loop over all pairs
  for (i in seq_len(n)) {
    for (j in seq_len(n)) {
      xi <- dataset[[i]]
      xj <- dataset[[j]]

      if (i == j) {
        mat[i, j] <- 1
        next
      }

      # Numeric-Numeric: Pearson correlation
      if (types[i] == "numeric" && types[j] == "numeric") {
        mat[i, j] <- suppressWarnings(cor(xi, xj, use = "pairwise.complete.obs"))
      } else {
        # At least one categorical: use mutual information on factors
        fi <- if (types[i] == "numeric") dataset_discretized[[i]] else as.factor(xi)
        fj <- if (types[j] == "numeric") dataset_discretized[[j]] else as.factor(xj)

        mi <- mutual_information(fi, fj, na.rm = na.rm)  # your MI function

        if (normalize_mi) {
          Hx <- entropy(fi, na.rm = na.rm)
          Hy <- entropy(fj, na.rm = na.rm)
          denom <- sqrt(Hx * Hy)
          mat[i, j] <- if (denom == 0) 0 else mi / denom
        } else {
          mat[i, j] <- mi
        }
      }
    }
  }
  return(mat)
}



# 6. PLOTS ---------------------------------------------------------------------
plot_auc <- function(metrics_df) {
  # --- checks ---
  if (!("variable" %in% names(metrics_df)) || !("auc" %in% names(metrics_df))) {
    stop("plot_auc: metrics_df must contain columns 'variable' and 'auc'")
  }

  aucs <- metrics_df$auc
  vars <- metrics_df$variable
  valid <- !is.na(aucs)

  if (sum(valid) == 0) {
    warning("plot_auc: no AUC values to plot")
    return(NULL)
  }

  aucs <- aucs[valid]
  vars <- vars[valid]

  # Sort by descending AUC
  ord <- order(aucs, decreasing = TRUE)
  aucs <- aucs[ord]
  vars <- vars[ord]

  # --- Plot ---
  bar_positions <- barplot(
    height = aucs,
    names.arg = vars,
    las = 2,
    ylim = c(0, 1),
    col = "skyblue",
    border = "gray40",
    main = "AUC per numeric attribute",
    ylab = "AUC"
  )

  abline(h = 0.5, col = "gray50", lty = 2)

  invisible(NULL)
}


plot_assoc_matrix <- function(mat, main = "Association matrix") {
  if (!is.matrix(mat))
    stop("plot_assoc_matrix: mat must be a matrix")

  # Convert matrix to long format for ggplot
  df <- melt(mat)
  colnames(df) <- c("Row", "Col", "Value")

  # Reverse y-axis order (so top matches your image())
  df$Row <- factor(df$Row, levels = rev(rownames(mat)))
  df$Col <- factor(df$Col, levels = colnames(mat))

  # bwr palette (blue → white → red)
  palette_colors <- colorRampPalette(c("blue", "white", "red"))(200)

  ggplot(df, aes(x = Col, y = Row, fill = Value)) +
    geom_tile(color = NA) +
    scale_fill_gradientn(
      colours = palette_colors,
      limits = c(-1, 1),
      name = "Value"
    ) +
    labs(title = main, x = "", y = "") +
    theme_minimal(base_size = 14) +
    theme(
      axis.text.x  = element_text(angle = 90, vjust = 0.5, hjust = 1),
      panel.grid   = element_blank(),
      plot.title   = element_text(hjust = 0.5)
    )
}




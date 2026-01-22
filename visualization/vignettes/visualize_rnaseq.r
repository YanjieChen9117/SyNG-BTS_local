# ==============================================================================
# RNA-seq 数据增强评估结果可视化脚本 - 本地化版本 (Customized CVAE 单模型)
# ==============================================================================
#
# 说明:
# - 读取 visualization/evaluation_results/ 下的 *-RNA_evaluation_results.csv
# - 图片保存到 report/2026-01-23/
# - 支持单模型 CVAE_customized，无 model_list
#
# 运行方式: 在 ** 项目根目录 ** 下执行，例如:
#   setwd("/path/to/SyNG-BTS_local")
#   source("visualization/vignettes/visualize_rnaseq.r")
# ==============================================================================

library(tidyverse)
library(ggplot2)

# 项目根目录（若在 visualization/vignettes 下运行则自动上溯）
repo_root <- getwd()
if (basename(repo_root) == "vignettes" && basename(dirname(repo_root)) == "visualization") {
  repo_root <- normalizePath(file.path(repo_root, "..", ".."), winslash = "/")
}

cancer_list <- c("PAAD-RNA", "READ-RNA", "COAD-RNA", "LAML-RNA", "SKCM-RNA")
metrics <- c("ARI", "ccc_pos", "ccc_log2FC", "ccc_log10pvalue")

eval_dir <- file.path(repo_root, "visualization", "evaluation_results")
plot_dir <- file.path(repo_root, "report", "2026-01-23")
dir.create(plot_dir, showWarnings = FALSE, recursive = TRUE)

for (cancer in cancer_list) {
  cat("正在绘制:", cancer, "\n")

  file_path <- file.path(eval_dir, paste0(cancer, "_evaluation_results.csv"))
  if (!file.exists(file_path)) {
    cat("未找到文件:", file_path, "，跳过\n")
    next
  }

  cat("读取文件:", file_path, "\n")
  all_results <- read.csv(file_path)
  cat("数据维度:", dim(all_results), "\n")

  required_cols <- c(metrics, "norm", "model")
  missing_cols <- setdiff(required_cols, colnames(all_results))
  if (length(missing_cols) > 0) {
    cat("缺少必需的列:", paste(missing_cols, collapse = ", "), "，跳过\n")
    next
  }

  df_long <- all_results %>%
    pivot_longer(cols = all_of(metrics), names_to = "metric", values_to = "value")

  df_long$norm <- as.factor(df_long$norm)
  levels(df_long$norm)[levels(df_long$norm) == ""] <- "Raw"

  # 单模型 CVAE_customized：按实际出现的 model 取 levels
  df_long$model <- factor(df_long$model, levels = unique(df_long$model))
  df_long$metric <- factor(df_long$metric, levels = metrics)

  df_box <- df_long %>%
    group_by(model, norm, metric) %>%
    mutate(
      n_valid = sum(!is.na(value)),
      sd_value = ifelse(n_valid > 1, sd(value, na.rm = TRUE), NA_real_),
      is_constant = !is.na(sd_value) & sd_value == 0,
      all_na = (n_valid == 0)
    ) %>%
    ungroup() %>%
    filter(!is_constant & !all_na)

  if (nrow(df_box) == 0) {
    cat("没有可绘制的数据，跳过\n")
    next
  }

  dodge_w <- 0.7
  p <- ggplot(df_box, aes(x = norm, y = value, fill = model)) +
    geom_boxplot(
      width = 0.6, outlier.size = 0.5, color = "black",
      position = position_dodge(width = dodge_w)
    ) +
    facet_grid(metric ~ ., scales = "free_y") +
    theme_classic(base_size = 16) +
    theme(
      strip.text       = element_text(size = 12),
      strip.background = element_blank(),
      axis.title.x     = element_blank(),
      axis.title.y     = element_text(size = 14),
      axis.text.x      = element_text(angle = 45, hjust = 1),
      legend.position  = "right",
      panel.border     = element_rect(colour = "black", fill = NA, linewidth = 0.8),
      panel.spacing.y  = unit(0.6, "lines")
    ) +
    labs(x = NULL, y = "Evaluation Score", fill = "Model")

  save_path <- file.path(plot_dir, paste0(cancer, "_evaluation_boxplot.png"))
  cat("保存到:", save_path, "\n")
  ggsave(save_path, plot = p, width = 8, height = 12, dpi = 300)

  if (file.exists(save_path)) {
    cat("已保存:", save_path, "\n")
  } else {
    cat("保存失败\n")
  }
}

cat("绘图完成。\n")

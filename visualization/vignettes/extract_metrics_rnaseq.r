# ==============================================================================
# RNA-seq 数据增强评估脚本 - 本地化版本 (Customized CVAE 单模型)
# ==============================================================================
#
# 数据结构说明:
# - 数据路径: RNA-seq_augmentation_data/main/{癌症类型}_5-2/{标准化方法}/batch_{n}/
# - 文件命名: {Cancer}Positive_5-2_{norm}_test.csv, {Cancer}Positive_5-2_{norm}_train_CVAE_augmented.csv
# - 仅支持单个 customized CVAE 模型，无 model_list
#
# 输出: visualization/evaluation_results/{癌症类型}-RNA_evaluation_results.csv
#
# 运行方式: 在 ** 项目根目录 ** 下执行，例如:
#   setwd("/path/to/SyNG-BTS_local")
#   source("visualization/vignettes/extract_metrics_rnaseq.r")
# ==============================================================================

# 项目根目录（若在 visualization/vignettes 下运行则自动上溯）
repo_root <- getwd()
if (basename(repo_root) == "vignettes" && basename(dirname(repo_root)) == "visualization") {
  repo_root <- normalizePath(file.path(repo_root, "..", ".."), winslash = "/")
}

# 加载本仓库的评估函数（不依赖其他 repository）
eval_r_dir <- file.path(repo_root, "visualization", "R")
source(file.path(eval_r_dir, "evaluations_functions.r"))
source(file.path(eval_r_dir, "get_evaluation.r"))

# 坐标文件等可选；本仓库若无则置 NULL
coords <- NULL

# ==============================================================================
# 定义参数 - 适配 RNA-seq_augmentation_data/main + customized CVAE
# ==============================================================================
cancer_list <- c("COAD", "LAML", "PAAD", "READ", "SKCM")
norm_list <- c("DESeq", "TC", "raw")
batch_list <- 1:20
base_path <- file.path(repo_root, "RNA-seq_augmentation_data", "main")
output_dir <- file.path(repo_root, "visualization", "evaluation_results")

if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

# 按癌症类型汇总并写结果（与原先 per-cancer 输出格式一致）
for (cancer in cancer_list) {
  output_cancer_name <- paste0(cancer, "-RNA")
  cat("正在处理癌症类型:", cancer, " (输出名称:", output_cancer_name, ")\n")

  result_df <- get_eval_all_configs_customized_cvae(
    cancer = cancer,
    norm_list = norm_list,
    batch_list = batch_list,
    base_path = base_path,
    coords = coords,
    draw = 1L,
    group_col = "groups",
    plot_first = FALSE
  )

  if (nrow(result_df) > 0) {
    result_df$cancer <- output_cancer_name
  }

  output_file <- file.path(output_dir, paste0(output_cancer_name, "_evaluation_results.csv"))
  write.csv(result_df, output_file, row.names = FALSE)
  cat("结果已保存到:", output_file, "\n")
  cat("处理了", nrow(result_df), "个配置组合\n\n")
}

cat("所有癌症类型的评估已完成！\n")

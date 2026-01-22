get_eval <- function(real_df, generated_df,
                     model_name = "Model",
                     log = TRUE,
                     failure = c("replace", "remove"),
                     poly = FALSE,
                     coords = NULL,
                     draw = 5,
                     group_col = "groups",
                     plot_first = FALSE) {
    failure <- match.arg(failure)

    df_index_store <- data.frame(
        model = rep(model_name, draw),
        draw = 1:draw,
        ccc_log10pvalue = NA,
        ccc_log2FC = NA,
        ARI = NA,
        fail_prop = NA,
        ccc_pos = NA,
        ks_mean = NA,
        ks_sd = NA,
        ks_zero = NA,
        ks_cv = NA
    )

    # 提取表达列名（不包含 group 列）
    cols_to_use <- setdiff(colnames(real_df), group_col)

    # 提取 real 的表达数据和 groups
    dat_real <- as.matrix(real_df[, cols_to_use])
    groups_real <- if (!is.null(group_col) && group_col %in% colnames(real_df)) {
        real_df[[group_col]]
    } else {
        NULL
    }

    for (i in 1:draw) {
        # 从 generated 中分层抽样
        sampled_generated <- stratified_sample(
            real_df = real_df,
            generated_df = generated_df,
            group_col = if (group_col %in% colnames(real_df)) group_col else NULL,
            replace = FALSE,
            seed = 123 + i
        )

        # 提取 generated 的表达数据和 groups
        dat_generated <- as.matrix(sampled_generated[, cols_to_use])
        groups_generated <- if (!is.null(group_col) && group_col %in% colnames(sampled_generated)) {
            sampled_generated[[group_col]]
        } else {
            NULL
        }

        # 可视化（仅第一次）
        if (i == 1 && plot_first) {
            try(print(heatmap_eval(real_df, sampled_generated, group_col = group_col, log = log)), silent = TRUE)
            try(
                {
                    umap_out <- UMAP_eval(real_df, sampled_generated, group_col = group_col, log = log, failure = failure)
                    print(umap_out$p_umap)
                },
                silent = TRUE
            )
        }

        # DEA
        try(
            {
                de_res <- DEA_eval(
                    dat_real, dat_generated,
                    groups_real = groups_real,
                    groups_generated = groups_generated,
                    log = log, failure = failure
                )
                df_index_store[i, c("ccc_log10pvalue", "ccc_log2FC")] <- de_res
            },
            silent = TRUE
        )

        # Clustering
        try(
            {
                df_index_store[i, "ARI"] <- cluster_eval(
                    dat_real, dat_generated,
                    groups_real = groups_real,
                    groups_generated = groups_generated,
                    log = log, failure = failure
                )
            },
            silent = TRUE
        )

        # Failure features
        try(
            {
                df_index_store[i, "fail_prop"] <- fail_features_eval(dat_real, dat_generated)
            },
            silent = TRUE
        )

        # ccpos
        try(
            {
                df_index_store[i, "ccc_pos"] <- ccpos_eval(
                    dat_real, dat_generated,
                    failure = failure, log = log,
                    coords = coords, poly = poly, thres = 32
                )
            },
            silent = TRUE
        )

        # Summary
        try(
            {
                summary_out <- summary_eval(dat_real, dat_generated, log = log, failure = failure)
                df_index_store[i, c("ks_mean", "ks_sd", "ks_zero", "ks_cv")] <- unlist(summary_out)
            },
            silent = TRUE
        )
    }

    return(df_index_store)
}

get_eval_all_configs <- function(cancer, subtype_list, epoch_list, batch_list, model_list, norm_list,
                                 base_path_real, base_path_generated,
                                 log = TRUE,
                                 failure = "replace",
                                 poly = TRUE,
                                 coords = NULL,
                                 draw = 5,
                                 group_col = "groups",
                                 plot_first = FALSE) {
    all_results <- list()

    for (subtype in subtype_list) {
        for (epoch in epoch_list) {
            for (batch in batch_list) {
                for (model in model_list) {
                    for (norm in norm_list) {
                        # 构造文件路径
                        # 处理模型名称映射
                        model_mapping <- list(
                            "CVAE1-10" = "CVAE1-10",
                            "AE-CVAE1-10" = "AEhead_CVAE1-10"
                        )
                        mapped_model <- model_mapping[[model]]
                        
                        # 构建批次目录路径
                        batch_dir <- file.path(base_path_real, cancer, norm, paste0("batch_", batch))
                        
                        # 构建文件路径
                        real_file <- file.path(batch_dir, paste0(cancer, "_", norm, "_batch_", batch, "_test.csv"))
                        
                        # 根据模型类型构建生成文件路径
                        if (model == "AE-CVAE1-10") {
                            generated_file <- file.path(batch_dir, paste0(cancer, "_", norm, "_batch_", batch, "_train_AEhead_epoch", epoch, "_batch01_CVAE1-10_generated.csv"))
                        } else {
                            generated_file <- file.path(batch_dir, paste0(cancer, "_", norm, "_batch_", batch, "_train_epoch", epoch, "_batch01_", mapped_model, "_generated.csv"))
                        }

                        cat("Processing:", basename(real_file), " + ", basename(generated_file), "\n")

                        # 检查文件存在
                        if (!file.exists(real_file) || !file.exists(generated_file)) {
                            warning("Missing file: ", real_file, " or ", generated_file)
                            next
                        }

                        # 读取真实数据
                        real_df <- read.csv(real_file, check.names = FALSE)
                        if ("samples" %in% colnames(real_df)) {
                            real_df$samples <- NULL
                        }
                        
                        # 将groups列统一为字符格式的1/0
                        if (group_col %in% colnames(real_df)) {
                            if (all(real_df[[group_col]] %in% c("YES", "NO"))) {
                                # 从YES/NO映射为1/0
                                real_df[[group_col]] <- ifelse(real_df[[group_col]] == "YES", "1", "0")
                            } else {
                                # 已经是数字格式，转换为字符
                                real_df[[group_col]] <- as.character(real_df[[group_col]])
                            }
                        }
                        
                        cols_to_log <- setdiff(colnames(real_df), group_col)
                        real_df[cols_to_log] <- log2(real_df[cols_to_log] + 1)

                        # 读取生成数据
                        generated_df <- read.csv(generated_file, header = FALSE, check.names = FALSE)
                        
                        # 处理列数不匹配的情况
                        ############################################################
                        expr_cols <- setdiff(colnames(real_df), group_col)
                        
                        if (ncol(generated_df) == length(expr_cols)) {
                            # 生成数据只有表达数据，需要添加groups列
                            colnames(generated_df) <- expr_cols
                            generated_df[[group_col]] <- sample(real_df[[group_col]], nrow(generated_df), replace = TRUE)
                        } else if (ncol(generated_df) == ncol(real_df)) {
                            # 生成数据列数与真实数据相同
                            colnames(generated_df) <- colnames(real_df)
                        } else if (ncol(generated_df) >= length(expr_cols)) {
                            # 生成数据列数大于等于表达数据列数，取前N列
                            generated_df <- generated_df[, 1:length(expr_cols)]
                            colnames(generated_df) <- expr_cols
                            generated_df[[group_col]] <- sample(real_df[[group_col]], nrow(generated_df), replace = TRUE)
                        } else {
                            warning("生成数据列数不匹配，跳过此配置")
                            next
                        }
                        ############################################################

                        
                        # 调用核心评估函数
                        result <- get_eval(
                            real_df = real_df,
                            generated_df = generated_df,
                            model_name = model,
                            log = log,
                            failure = failure,
                            poly = poly,
                            coords = coords,
                            draw = draw,
                            group_col = group_col,
                            plot_first = plot_first
                        )

                        # 添加元信息
                        result$cancer <- cancer
                        result$subtype <- subtype
                        result$epoch <- epoch
                        result$batch <- batch
                        result$norm <- norm

                        all_results[[length(all_results) + 1]] <- result
                    }
                }
            }
        }
    }

    final_result <- do.call(rbind, all_results)
    return(final_result)
}

# 新的函数适配 optimal_config_offline_aug_test 数据格式
get_eval_all_configs_optimal <- function(cancer, subtype_list, epoch_list, batch_list, model_list, norm_list,
                                        base_path_real, base_path_generated,
                                        log = TRUE,
                                        failure = "replace",
                                        poly = TRUE,
                                        coords = NULL,
                                        draw = 5,
                                        group_col = "groups",
                                        plot_first = FALSE) {
    all_results <- list()

    for (subtype in subtype_list) {
        for (epoch in epoch_list) {
            for (batch in batch_list) {
                for (model in model_list) {
                    for (norm in norm_list) {
                        # 构建批次目录路径 - 适配新的文件夹结构
                        batch_dir <- file.path(base_path_real, cancer, norm, paste0("batch_", batch))
                        
                        # 构建文件路径
                        real_file <- file.path(batch_dir, paste0(cancer, "_", norm, "_batch_", batch, "_test.csv"))
                        
                        # 根据模型类型构建生成文件路径 - 适配新的文件命名格式
                        if (model == "AEhead_CVAE1-10") {
                            generated_file <- file.path(batch_dir, paste0(cancer, "_", norm, "_batch_", batch, "_train_AEhead_epoch", epoch, "_batch01_CVAE1-10_generated.csv"))
                        } else if (model == "Gaussianhead_CVAE1-10") {
                            generated_file <- file.path(batch_dir, paste0(cancer, "_", norm, "_batch_", batch, "_train_Gaussianhead_epoch", epoch, "_batch01_CVAE1-10_generated.csv"))
                        } else {
                            # CVAE1-10 基础版本
                            generated_file <- file.path(batch_dir, paste0(cancer, "_", norm, "_batch_", batch, "_train_epoch", epoch, "_batch01_CVAE1-10_generated.csv"))
                        }

                        cat("Processing:", basename(real_file), " + ", basename(generated_file), "\n")

                        # 检查文件存在
                        if (!file.exists(real_file) || !file.exists(generated_file)) {
                            warning("Missing file: ", real_file, " or ", generated_file)
                            next
                        }

                        # 读取真实数据
                        real_df <- read.csv(real_file, check.names = FALSE)
                        if ("samples" %in% colnames(real_df)) {
                            real_df$samples <- NULL
                        }
                        
                        # 将groups列统一为字符格式的1/0
                        if (group_col %in% colnames(real_df)) {
                            if (all(real_df[[group_col]] %in% c("YES", "NO"))) {
                                # 从YES/NO映射为1/0
                                real_df[[group_col]] <- ifelse(real_df[[group_col]] == "YES", "1", "0")
                            } else {
                                # 已经是数字格式，转换为字符
                                real_df[[group_col]] <- as.character(real_df[[group_col]])
                            }
                        }
                        
                        cols_to_log <- setdiff(colnames(real_df), group_col)
                        real_df[cols_to_log] <- log2(real_df[cols_to_log] + 1)

                        # 读取生成数据
                        generated_df <- read.csv(generated_file, header = FALSE, check.names = FALSE)
                        
                        # 处理列数不匹配的情况
                        ############################################################
                        expr_cols <- setdiff(colnames(real_df), group_col)
                        
                        if (ncol(generated_df) == length(expr_cols)) {
                            # 生成数据只有表达数据，需要添加groups列
                            colnames(generated_df) <- expr_cols
                            generated_df[[group_col]] <- sample(real_df[[group_col]], nrow(generated_df), replace = TRUE)
                        } else if (ncol(generated_df) == ncol(real_df)) {
                            # 生成数据列数与真实数据相同
                            colnames(generated_df) <- colnames(real_df)
                        } else if (ncol(generated_df) >= length(expr_cols)) {
                            # 生成数据列数大于等于表达数据列数，取前N列
                            generated_df <- generated_df[, 1:length(expr_cols)]
                            colnames(generated_df) <- expr_cols
                            generated_df[[group_col]] <- sample(real_df[[group_col]], nrow(generated_df), replace = TRUE)
                        } else {
                            warning("生成数据列数不匹配，跳过此配置")
                            next
                        }
                        ############################################################

                        
                        # 调用核心评估函数
                        result <- get_eval(
                            real_df = real_df,
                            generated_df = generated_df,
                            model_name = model,
                            log = log,
                            failure = failure,
                            poly = poly,
                            coords = coords,
                            draw = draw,
                            group_col = group_col,
                            plot_first = plot_first
                        )
                        
                        # 添加配置信息
                        result$cancer <- cancer
                        result$subtype <- subtype
                        result$epoch <- epoch
                        result$batch <- batch
                        result$norm <- norm
                        
                        all_results[[length(all_results) + 1]] <- result
                    }
                }
            }
        }
    }

    final_result <- do.call(rbind, all_results)
    return(final_result)
}

# 新的函数适配 RNAseq_augmentation_data 数据格式
get_eval_all_configs_rnaseq <- function(cancer, subtype_list, epoch_list, batch_list, model_list, norm_list,
                                        base_path_real, base_path_generated,
                                        log = TRUE,
                                        failure = "replace",
                                        poly = TRUE,
                                        coords = NULL,
                                        draw = 5,
                                        group_col = "groups",
                                        plot_first = FALSE) {
    all_results <- list()

    for (subtype in subtype_list) {
        for (epoch in epoch_list) {
            for (batch in batch_list) {
                for (model in model_list) {
                    for (norm in norm_list) {
                        # 构建目录路径 - 适配 RNAseq_augmentation_data 新的文件夹结构
                        # 格式: RNAseq_augmentation_data/{癌症类型}_5-2/{标准化方法}/batch_{数字}/{模型}/
                        model_dir <- file.path(base_path_real, paste0(cancer, "_5-2"), norm, paste0("batch_", batch), model)
                        
                        # 构建文件路径
                        # 测试文件: {癌症类型}_5-2_{标准化方法}_batch_{数字}_test.csv
                        real_file <- file.path(model_dir, paste0(cancer, "_5-2_", norm, "_batch_", batch, "_test.csv"))
                        
                        # 生成文件: {癌症类型}_5-2_{标准化方法}_batch_{数字}_train_epoch{epoch}_batch01_{模型}_generated.csv
                        generated_file <- file.path(model_dir, paste0(cancer, "_5-2_", norm, "_batch_", batch, "_train_epoch", epoch, "_batch01_", model, "_generated.csv"))

                        cat("Processing:", basename(real_file), " + ", basename(generated_file), "\n")

                        # 检查文件存在
                        if (!file.exists(real_file) || !file.exists(generated_file)) {
                            warning("Missing file: ", real_file, " or ", generated_file)
                            next
                        }

                        # 读取真实数据
                        real_df <- read.csv(real_file, check.names = FALSE)
                        if ("samples" %in% colnames(real_df)) {
                            real_df$samples <- NULL
                        }
                        
                        # 将groups列统一为字符格式的1/0
                        if (group_col %in% colnames(real_df)) {
                            if (all(real_df[[group_col]] %in% c("YES", "NO"))) {
                                # 从YES/NO映射为1/0
                                real_df[[group_col]] <- ifelse(real_df[[group_col]] == "YES", "1", "0")
                            } else {
                                # 已经是数字格式，转换为字符
                                real_df[[group_col]] <- as.character(real_df[[group_col]])
                            }
                        }
                        
                        cols_to_log <- setdiff(colnames(real_df), group_col)
                        real_df[cols_to_log] <- log2(real_df[cols_to_log] + 1)

                        # 读取生成数据
                        generated_df <- read.csv(generated_file, header = FALSE, check.names = FALSE)
                        
                        # 处理列数不匹配的情况
                        ############################################################
                        expr_cols <- setdiff(colnames(real_df), group_col)
                        
                        if (ncol(generated_df) == length(expr_cols)) {
                            # 生成数据只有表达数据，需要添加groups列
                            colnames(generated_df) <- expr_cols
                            generated_df[[group_col]] <- sample(real_df[[group_col]], nrow(generated_df), replace = TRUE)
                        } else if (ncol(generated_df) == ncol(real_df)) {
                            # 生成数据列数与真实数据相同
                            colnames(generated_df) <- colnames(real_df)
                        } else if (ncol(generated_df) >= length(expr_cols)) {
                            # 生成数据列数大于等于表达数据列数，取前N列
                            generated_df <- generated_df[, 1:length(expr_cols)]
                            colnames(generated_df) <- expr_cols
                            generated_df[[group_col]] <- sample(real_df[[group_col]], nrow(generated_df), replace = TRUE)
                        } else {
                            warning("生成数据列数不匹配，跳过此配置")
                            next
                        }
                        ############################################################

                        
                        # 调用核心评估函数
                        result <- get_eval(
                            real_df = real_df,
                            generated_df = generated_df,
                            model_name = model,
                            log = log,
                            failure = failure,
                            poly = poly,
                            coords = coords,
                            draw = draw,
                            group_col = group_col,
                            plot_first = plot_first
                        )
                        
                        # 添加配置信息
                        result$cancer <- cancer
                        result$subtype <- subtype
                        result$epoch <- epoch
                        result$batch <- batch
                        result$norm <- norm
                        
                        all_results[[length(all_results) + 1]] <- result
                    }
                }
            }
        }
    }

    final_result <- do.call(rbind, all_results)
    return(final_result)
}

# 适配 SyNG-BTS 本地 customized CVAE 数据格式 (RNA-seq_augmentation_data/main/)
# 无 model_list，单模型 CVAE_customized；文件命名见 RNA-seq_batch_augmentation.py
get_eval_all_configs_customized_cvae <- function(cancer, norm_list, batch_list,
                                                 base_path,
                                                 log = TRUE,
                                                 failure = "replace",
                                                 poly = TRUE,
                                                 coords = NULL,
                                                 draw = 5,
                                                 group_col = "groups",
                                                 plot_first = FALSE) {
    all_results <- list()
    model_name <- "CVAE_customized"

    for (norm in norm_list) {
            for (batch in batch_list) {
                # 路径: base_path / {cancer}_5-2 / {norm} / batch_{batch}
                batch_dir <- file.path(base_path, paste0(cancer, "_5-2"), norm, paste0("batch_", batch))
                # 文件前缀: raw -> {cancer}Positive_5-2, 否则 {cancer}Positive_5-2_{norm}
                prefix <- if (norm == "raw") {
                    paste0(cancer, "Positive_5-2")
                } else {
                    paste0(cancer, "Positive_5-2_", norm)
                }
                real_file <- file.path(batch_dir, paste0(prefix, "_test.csv"))
                generated_file <- file.path(batch_dir, paste0(prefix, "_train_CVAE_augmented.csv"))

                cat("Processing:", basename(real_file), " + ", basename(generated_file), "\n")

                if (!file.exists(real_file) || !file.exists(generated_file)) {
                    warning("Missing file: ", real_file, " or ", generated_file)
                    next
                }

                real_df <- read.csv(real_file, check.names = FALSE)
                if ("samples" %in% colnames(real_df)) {
                    real_df$samples <- NULL
                }
                if (group_col %in% colnames(real_df)) {
                    if (all(real_df[[group_col]] %in% c("YES", "NO"))) {
                        real_df[[group_col]] <- ifelse(real_df[[group_col]] == "YES", "1", "0")
                    } else {
                        real_df[[group_col]] <- as.character(real_df[[group_col]])
                    }
                }
                cols_to_log <- setdiff(colnames(real_df), group_col)
                real_df[cols_to_log] <- log2(real_df[cols_to_log] + 1)

                # customized CVAE 输出带 header
                generated_df <- read.csv(generated_file, check.names = FALSE)
                expr_cols <- setdiff(colnames(real_df), group_col)

                if (ncol(generated_df) == length(expr_cols)) {
                    colnames(generated_df) <- expr_cols
                    generated_df[[group_col]] <- sample(real_df[[group_col]], nrow(generated_df), replace = TRUE)
                } else if (ncol(generated_df) == ncol(real_df)) {
                    colnames(generated_df) <- colnames(real_df)
                } else if (ncol(generated_df) >= length(expr_cols)) {
                    generated_df <- generated_df[, 1:length(expr_cols)]
                    colnames(generated_df) <- expr_cols
                    generated_df[[group_col]] <- sample(real_df[[group_col]], nrow(generated_df), replace = TRUE)
                } else {
                    warning("生成数据列数不匹配，跳过此配置")
                    next
                }

                result <- get_eval(
                    real_df = real_df,
                    generated_df = generated_df,
                    model_name = model_name,
                    log = log,
                    failure = failure,
                    poly = poly,
                    coords = coords,
                    draw = draw,
                    group_col = group_col,
                    plot_first = plot_first
                )
                result$cancer <- cancer
                result$subtype <- ""
                result$epoch <- NA
                result$batch <- batch
                result$norm <- norm
                all_results[[length(all_results) + 1]] <- result
            }
        }

    final_result <- do.call(rbind, all_results)
    return(final_result)
}

from semi.ssl import train_advSemiSeg

ssl_algorithm={'AdvSemiSeg':train_advSemiSeg}

def train_ssl(
            cfg,
            model,
            train_dataset,
            label_ratio,
            split_ids,
            ssl_method,
            val_dataset,
            optimizer,
            save_dir,
            iters,
            batch_size,
            resume_model,
            save_interval,
            log_iters,
            num_workers,
            use_vdl,
            losses,
            keep_checkpoint_max,
            test_config,
            fp16,
            profiler_options,
            to_static_training):
    ssl_algorithm[ssl_method](
            cfg,
            model,
            train_dataset,
            label_ratio,
            split_ids,
            val_dataset,
            optimizer,
            save_dir,
            iters,
            batch_size,
            resume_model,
            save_interval,
            log_iters,
            num_workers,
            use_vdl,
            losses,
            keep_checkpoint_max,
            test_config,
            fp16,
            profiler_options,
            to_static_training)




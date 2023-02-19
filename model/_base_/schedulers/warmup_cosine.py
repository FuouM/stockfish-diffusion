from scheduler_stuff.warmup_cosine_scheduler import LambdaWarmUpCosineScheduler

lambda_func = LambdaWarmUpCosineScheduler(
    warm_up_steps=1000,
    lr_min=1e-4,
    lr_max=8e-4,
    lr_start=1e-5,
    max_decay_steps=150000,
)

optimizer = dict(
    type="AdamW",
    lr=1.0,
    weight_decay=1e-2,
    betas=(0.9, 0.98),
    eps=1e-9,
)

scheduler = dict(
    type="LambdaLR",
    lr_lambda=lambda_func,
)

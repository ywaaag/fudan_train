"""Apply reviewed post-resume optimizer settings explicitly."""
def enforce_optimizer_overrides(runner, manifest: dict) -> None:
    optimizer = manifest.get("optimizer")
    if optimizer is None:
        return
    actor_lr = optimizer["learning_rate"]
    runner.alg.learning_rate = actor_lr
    runner.alg.schedule = optimizer["schedule"]
    if "entropy_coef" in optimizer:
        runner.alg.entropy_coef = optimizer["entropy_coef"]
    for group in runner.alg.optimizer.param_groups:
        group["lr"] = actor_lr
    if runner.alg.extra_optimizer is not None:
        for group in runner.alg.extra_optimizer.param_groups:
            group["lr"] = optimizer["extra_learning_rate"]


def get_loss(trainer_history):
    # extract loss metrics
    train_loss = [metric["loss"] for metric in trainer_history if "loss" in metric.keys()]

    val_loss = [metric["eval_loss"] for metric in trainer_history if "eval_loss" in metric.keys()]

    return get_loss 
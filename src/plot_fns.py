def get_loss(trainer_history):
    eval_losses = {}
    train_losses = {}

    for item in trainer_history:
        if 'eval_loss' in item:
            epoch = item['epoch']
            eval_losses[epoch] = item['eval_loss']
        if 'loss' in item:
            epoch = item['epoch']
            train_losses[epoch] = item['loss']

    return train_losses, eval_losses
def _get_filename(dataset_name, config):
    """Get a filename based on the configuration used for model and dataset"""
    return dataset_name + '_{e}_epochs_{h}_{nl}_layers_{t}_timesteps'.format(
        e=config['epochs'], h=config['model_config']['hidden_size'],
        nl=config['model_config']['n_layers'], t=config['sequence_length']
    )


text8config = {
    'batch_size': 128,
    'sequence_length': 100,
    'epochs': 51,
    'initial_lr': 0.001,
    'lr_schedule': {
        25: 0.0005,
        39: 0.0002
    },
    'model_config': {
        'embedding_size': 64,
        'hidden_size': 512,
        'n_layers': 2
    },
    'is_bytes': False
}
text8config['filename'] = _get_filename('text8', text8config)

ptbconfig = {
    'batch_size': 128,
    'sequence_length': 100,
    'epochs': 60,
    'initial_lr': 0.001,
    'lr_schedule': {
        40: 0.0001,
        50: 0.00002
    },
    'model_config': {
        'embedding_size': 16,
        'hidden_size': 512,
        'n_layers': 2
    },
    'is_bytes': False
}
ptbconfig['filename'] = _get_filename('ptb', ptbconfig)

hutter_prize_config = {
    'batch_size': 128,
    'sequence_length': 100,
    'epochs': 51,
    'initial_lr': 0.001,
    'lr_schedule': {
        25: 0.0005,
        39: 0.0002
    },
    'model_config': {
        'embedding_size': 64,
        'hidden_size': 512,
        'n_layers': 2
    },
    'is_bytes': True
}
hutter_prize_config['filename'] = _get_filename('hutter_prize', hutter_prize_config)

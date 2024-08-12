from __future__ import annotations

from core.agents.stablebaselines3.models import DRLAgent as DRLAgent_sb3
from core.data_process.process import DataProcess
from env.CryptoEnv.env_multiple_crypto import CryptoEnv

if __name__ == "__main__":
    technical_indicator_list = ['macd', 'rsi', 'cci', 'dx']
    path = 'D:\\Lab\\quant\\code\\finbot\\data\\btcusd_2022-01-02_to_2022-07-03.csv'
    process = DataProcess(path, 'BTCUSDT', technical_indicator_list)
    price_array, tech_array = process.run()
    data_config = {
        'price_array': price_array,
        'tech_array': tech_array
    }

    # build environment using processed data
    env = CryptoEnv(config=data_config)
    total_timesteps = 10000000
    model_name = 'ppo'
    agent_params = {
        "n_steps": 2048,
        "ent_coef": 0.01,
        "learning_rate": 0.00025,
        "batch_size": 64,
    }
    current_working_dir = './test_ppo'
    agent = DRLAgent_sb3(env=env)

    model = agent.get_model(model_name, model_kwargs=agent_params, tensorboard_log='./tensorboard_log')
    trained_model = agent.train_model(model=model,
                                      tb_log_name=model_name,
                                      total_timesteps=total_timesteps)

    print('Training finished!')
    trained_model.save(current_working_dir)
    print('Trained model saved in ' + str(current_working_dir))

from __future__ import annotations
from core.config import TRAIN_START_DATE, TRAIN_END_DATE, INDICATORS, ERL_PARAMS
from core.config_tickers import DOW_30_TICKER
from core.meta.data_processor import DataProcessor
from core.agents.stablebaselines3.models import DRLAgent as DRLAgent_sb3


def train(
    start_date,
    end_date,
    ticker_list,
    data_source,
    time_interval,
    technical_indicator_list,
    drl_lib,
    env,
    model_name,
    if_vix=True,
    **kwargs,
):
    # download data
    dp = DataProcessor(data_source, **kwargs)
    data = dp.download_data(ticker_list, start_date, end_date, time_interval)
    data = dp.clean_data(data)
    data = dp.add_technical_indicator(data, technical_indicator_list)
    
    if if_vix:
        data = dp.add_vix(data)
    
    price_array, tech_array, turbulence_array = dp.df_to_array(data, if_vix)
    
    env_config = {
        "price_array": price_array,
        "tech_array": tech_array,
        "turbulence_array": turbulence_array,
        "if_train": True,
    }


    env_instance = env(config=env_config)

    # read parameters
    cwd = kwargs.get("cwd", "./" + str(model_name))

   
    total_timesteps = kwargs.get("total_timesteps", 1e6)
    agent_params = kwargs.get("agent_params")

    agent = DRLAgent_sb3(env=env_instance)
    
    model = agent.get_model(model_name, model_kwargs=agent_params)
    
    trained_model = agent.train_model(
            model=model, tb_log_name=model_name, total_timesteps=total_timesteps
        )

    print("Training is finished!")
    
    trained_model.save(cwd)
    
    print("Trained model is saved in " + str(cwd))
 



if __name__ == "__main__":
    env = StockTradingEnv

    # demo for elegantrl
    kwargs = (
        {}
    )  # in current meta, with respect yahoofinance, kwargs is {}. For other data sources, such as joinquant, kwargs is not empty
    train(
        start_date=TRAIN_START_DATE,
        end_date=TRAIN_END_DATE,
        ticker_list=DOW_30_TICKER,
        data_source="yahoofinance",
        time_interval="1D",
        technical_indicator_list=INDICATORS,
        drl_lib="elegantrl",
        env=env,
        model_name="ppo",
        cwd="./test_ppo",
        erl_params=ERL_PARAMS,
        break_step=1e5,
        kwargs=kwargs,
    )
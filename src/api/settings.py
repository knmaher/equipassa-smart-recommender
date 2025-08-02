from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    data_path: str = "data/user_tool_interactions.csv"
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = True

    class Config:
        env_file = ".env"

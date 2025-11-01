from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """環境変数から設定を読み込む"""
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra='ignore' 
    )
    GEMINI_API_KEY: str

settings = Settings()
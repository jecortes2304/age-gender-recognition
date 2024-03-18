import os

from pydantic import BaseSettings


class ConfigsSettings(BaseSettings):
    DATABASE_PORT: int
    POSTGRES_PASSWORD: str
    POSTGRES_USER: str
    POSTGRES_DB: str
    POSTGRES_HOST: str
    POSTGRES_HOSTNAME: str

    HOST: str
    PORT: str
    CLIENT_URL: str
    RELOAD: bool
    LOGS_PATH: str

    ROOT_USER: str
    ROOT_PASS: str
    ROOT_EMAIL: str

    LINKEDIN_USER: str
    LINKEDIN_PASS: str
    CRUNCHBASE_USER: str
    CRUNCHBASE_PASS: str

    JWT_SECRET_KEY: str
    JWT_ALGORITHM: str
    ACCESS_TOKEN_EXPIRE_MINUTES: int
    SCRIPT_API_KEY: str
    UPDATE_LIMIT: int

    class Config:
        env_file = os.getenv('FAST_API_ENV')


settings = ConfigsSettings()

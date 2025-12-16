from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    openai_api_key: str
    mongodb_uri: str = "mongodb://localhost:27017"
    mongodb_db: str = "rag_poc"
    faiss_index_path: str = "./faiss_index"
    upload_dir: str = "./uploads"
    
    class Config:
        env_file = ".env"


settings = Settings()


import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    WANDB_API_KEY = os.getenv("WANDB_API_KEY")
    WANDB_PROJECT = os.getenv("WANDB_PROJECT", "bobby-flai")
    WANDB_ENTITY = os.getenv("WANDB_ENTITY")
    
    class Config:
        case_sensitive = True

settings = Settings()
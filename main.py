from typing import List, Dict, Optional, Union
from datetime import datetime

import os
import wandb
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

current_time = datetime.utcnow().isoformat()
if not os.getenv("RUNNING_IN_DOCKER"):
    print("RUNNING_IN_DOCKER IS NOT SET, loading .env file")
    load_dotenv()
else:
    print("RUNNING_IN_DOCKER IS SET, skipping .env file loading")

WANDB_API_KEY = os.getenv("WANDB_API_KEY")
if not WANDB_API_KEY:
    print("Warning: WANDB_API_KEY not found in environment.")
    # Potentially raise an error or exit if it's critical and missing
    # raise ValueError("WANDB_API_KEY must be set in the environment or .env file")


app = FastAPI()

wandb.init(
    project="bobby-flAI",
    config={
        "environment": os.getenv("ENVIRONMENT", "development"),
        "service_version": os.getenv("SERVICE_VERSION", "1.0.0"),
        "model_defaults": {
            "temperature": 0.7,
            "max_tokens": 2000
        }
    },
    tags=["recipe-generation", os.getenv("ENVIRONMENT", "development")]
)

class AutoEvaluation(BaseModel):
    grammar: Optional[Dict[str, Union[float, List[str]]]] = None
    hallucination: Optional[Dict[str, Union[float, List[str]]]] = None
    coherence: Optional[Dict[str, Union[float, List[str]]]] = None

class RecipeTrace(BaseModel):
    sessionId: str
    traceId: str
    prompt: str
    promptUrl: Optional[str] = None
    model: str
    response: str
    responseUrl: Optional[str] = None
    postprocessed: Optional[str] = None
    temperature: float
    promptTokens: Optional[int] = None
    completionTokens: Optional[int] = None
    totalTokens: Optional[int] = None
    responseTimeMs: int
    retryCount: int = 0
    autoEval: Optional[AutoEvaluation] = None
    metadata: Optional[Dict] = None
    rating: Optional[float] = None
    userFeedback: Optional[str] = None
    errorTags: List[str] = []
    responseType: Optional[str] = None

@app.post("/log-trace")
async def log_trace(trace: RecipeTrace):
    try:
        if trace.responseTimeMs < 0 or trace.responseTimeMs > 300000:
            raise ValueError(f"Invalid response time: {trace.responseTimeMs}ms")

        metrics = {
            "response_time_ms": trace.responseTimeMs,
            "prompt_tokens": trace.promptTokens,
            "completion_tokens": trace.completionTokens,
            "total_tokens": trace.totalTokens,
            "retry_count": trace.retryCount,
            "rating": trace.rating,
            "temperature": trace.temperature,
            "prompt_length": len(trace.prompt),
            "response_length": len(trace.response),
            "has_error": len(trace.errorTags) > 0,
        }

        metrics = {k: v for k, v in metrics.items() if v is not None}

        artifacts = {
            "prompt": trace.prompt,
            "response:": trace.response,
            "postprocessed": trace.postprocessed,
        }

        metadata = {
            "model": trace.model,
            "session_id": trace.sessionId,
            "trace_id": trace.traceId,
            "prompt_url": trace.promptUrl,
            "response_url": trace.responseUrl,
            "response_type": trace.responseType,
            "error_tags": trace.errorTags,
            "user_feedback": trace.userFeedback,
            "rating": trace.rating,            
        }
        
        if trace.autoEval:
            metadata["auto_eval"] = trace.autoEval.model_dump(exclude_unset=True)
        
        if trace.metadata:
            metadata.update(trace.metadata)
        
        wandb.log({
            "metrics": metrics,
            "artifacts": wandb.Table(
                columns=["category", "key", "value"],
                data=[
                    [cat, k, str(v)] 
                    for cat, items in artifacts.items() 
                    for k, v in items.items() 
                    if v is not None
                ]
            ),
            "metadata": metadata,
            "timestamp": current_time,
            "session_group": trace.sessionId,
        })

        return {"status": "success", "timestamp": current_time}
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        wandb.log({
            "error": {
                "type": type(e).__name__,
                "message": str(e),
                "trace_id": getattr(trace, 'traceId', 'N/A')
            }
        })
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    try:
        if wandb.run is None:
            raise Exception("W&B run not initialized")
        return {
            "status": "healthy",

            "wandb_run_id": wandb.run.id,
            "wandb_project": wandb.run.project,
            "wandb_entity": os.getenv("WANDB_ENTITY", "default")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.on_event("shutdown")
async def shutdown_event():
    print("Shutting down...")
    wandb.finish()
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

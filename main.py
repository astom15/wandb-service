"""FastAPI service for logging recipe generation traces to Weights & Biases."""

import json
import os
import re
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any, Dict, List, Optional, Union

import weave
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import wandb
from wandb_visualizations import update_visualizations

current_time =datetime.now(UTC).isoformat()
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

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    print("Shutting down...")
    weave.finish()

app = FastAPI(lifespan=lifespan)
weave.init(
    "bobby-flAI",
    global_attributes= {
        "environment": os.getenv("ENVIRONMENT", "development"),
        "service_version": os.getenv("SERVICE_VERSION", "1.0.0"),
        "model_defaults_temperature": 0.7,
        "model_defaults_max_tokens": 2000
    }
)
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
    temperature: float
    postprocessed: Optional[str] = None
    promptTokens: Optional[int] = None
    completionTokens: Optional[int] = None
    totalTokens: Optional[int] = None
    responseTimeMs: int
    retryCount: Optional[int] = 0
    autoEval: Optional[AutoEvaluation] = None
    metadata: Optional[Dict[str, Any]] = None
    rating: Optional[float] = None
    userFeedback: Optional[str] = None
    errorTags: List[str] = []
    responseType: List[str] = []
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None

class JSONValidationRequest(BaseModel):
    content: str
    sessionId: str
    traceId: str
    metadata: Dict[str, Any]

def normalize_json(content: str) -> str:
    """Normalize JSON formatting to ensure consistent structure."""
    try:
        # Parse and re-stringify to normalize formatting
        data = json.loads(content)
        return json.dumps(data, indent=2)
    except json.JSONDecodeError:
        # If parsing fails, return original content
        return content

@weave.op(name="validate-json")
@app.post("/validate-json")
async def validate_json(request: JSONValidationRequest):
    start_time = datetime.now(UTC)
    try:
        # Basic format validation
        content = normalize_json(request.content.strip())
        validation_steps = {
            "array_format": False,
            "markdown_removed": False,
            "intro_text_removed": False,
            "json_parsed": False,
            "structure_validated": False
        }
        
        # Track array format validation
        if not content.startswith('[') or not content.endswith(']'):
            print(f"Array format validation failed. Content starts with: {content[:10]}... and ends with: ...{content[-10:]}")
            raise ValueError("Response must be a JSON array (starts with [ and ends with ])")
        validation_steps["array_format"] = True
        print("Array format validation passed")

        # Check for common GPT formatting issues
        markdown_patterns = [
            (r'```json\n', "Response contains markdown code block"),
            (r'```\n', "Response contains markdown code block"),
        ]
        
        intro_patterns = [
            (r'^Here\'s', "Response contains introductory text"),
            (r'^I\'ll', "Response contains introductory text"),
            (r'^Let me', "Response contains introductory text"),
            (r'^Here are', "Response contains introductory text"),
            (r'^The recipes', "Response contains introductory text"),
            (r'^Based on', "Response contains introductory text"),
            (r'^Here is', "Response contains introductory text"),
            (r'^I\'ve', "Response contains introductory text"),
            (r'^I have', "Response contains introductory text"),
            (r'^Here', "Response contains introductory text"),
            (r'^I', "Response contains introductory text"),
        ]

        # Track markdown removal
        for pattern, message in markdown_patterns:
            if re.search(pattern, content):
                print(f"Markdown validation failed. Found pattern: {pattern}")
                raise ValueError(f"Invalid response format: {message}")
        validation_steps["markdown_removed"] = True
        print("Markdown validation passed")

        # Track intro text removal
        for pattern, message in intro_patterns:
            if re.search(pattern, content):
                print(f"Intro text validation failed. Found pattern: {pattern}")
                raise ValueError(f"Invalid response format: {message}")
        validation_steps["intro_text_removed"] = True
        print("Intro text validation passed")

        # Parse and validate structure
        try:
            data = json.loads(content)
            validation_steps["json_parsed"] = True
            print("JSON parsing passed")
            
            if not isinstance(data, list):
                print(f"Structure validation failed. Expected list, got {type(data)}")
                raise ValueError("Response must be a JSON array")
            if len(data) == 0:
                print("Structure validation failed. Array is empty")
                raise ValueError("Response array cannot be empty")

            # Validate first recipe
            first_recipe = data[0]
            required_fields = ['name', 'prepTime', 'cookTime', 'totalTime', 'ingredients', 'steps']
            missing_fields = [field for field in required_fields if field not in first_recipe]
            if missing_fields:
                print(f"Structure validation failed. Missing fields: {missing_fields}")
                print(f"Available fields: {list(first_recipe.keys())}")
                raise ValueError(f"Missing required fields in recipe: {', '.join(missing_fields)}")
            
            validation_steps["structure_validated"] = True
            print("Structure validation passed")

            # Calculate validation duration
            validation_duration = int((datetime.now(UTC) - start_time).total_seconds() * 1000)

            # Prepare validation data for visualization
            validation_data = {
                "success": True,
                "validation_steps": validation_steps,
                "validation_duration_ms": validation_duration,
                "content_length": len(content),
                "recipe_count": len(data),
                "timestamp": current_time,
                "session_id": request.sessionId,
                "trace_id": request.traceId
            }

            # Update visualizations
            update_visualizations([validation_data])

            return {"status": "success", "message": "JSON validation passed"}

        except json.JSONDecodeError as e:
            print(f"JSON parsing failed. Error: {str(e)}")
            print(f"Error location: line {e.lineno}, column {e.colno}")
            print(f"Error message: {e.msg}")
            # Print the content around the error
            lines = content.split('\n')
            if e.lineno > 1:
                print(f"Line {e.lineno-1}: {lines[e.lineno-2]}")
            print(f"Line {e.lineno}: {lines[e.lineno-1]}")
            if e.lineno < len(lines):
                print(f"Line {e.lineno+1}: {lines[e.lineno]}")
            raise ValueError(f"Invalid JSON format: {str(e)}")

    except Exception as e:
        # Calculate validation duration
        validation_duration = int((datetime.now(UTC) - start_time).total_seconds() * 1000)
        
        # Prepare error data for visualization
        validation_data = {
            "success": False,
            "validation_steps": validation_steps,
            "validation_duration_ms": validation_duration,
            "content_length": len(request.content),
            "error_type": type(e).__name__,
            "error_message": str(e),
            "timestamp": current_time,
            "session_id": request.sessionId,
            "trace_id": request.traceId
        }

        # Update visualizations with error data
        update_visualizations([validation_data])
        
        raise HTTPException(status_code=400, detail=str(e))
    
@weave.op(name="log-trace")
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
            "response": trace.response,
            "postprocessed": trace.postprocessed,
            "metadata": str(trace.metadata) if trace.metadata else None,
            "auto_eval": str(trace.autoEval.model_dump()) if trace.autoEval else None,
            "response_type": str(trace.responseType) if trace.responseType else None
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
                columns=["category", "value"],
                data=[
                    [cat, str(v)] 
                    for cat, v in artifacts.items() 
                    if v is not None
                ]
            ),
            "metadata": metadata,
            "timestamp": current_time,
            "session_group": trace.sessionId,
        })

        return {"status": "success", "timestamp": current_time}
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve)) from ve
    except Exception as e:
        print(f"Error processing trace: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        wandb.log({
            "error": {
                "type": type(e).__name__,
                "message": str(e),
                "trace_id": getattr(trace, 'traceId', 'N/A')
            }
        })
        raise HTTPException(status_code=500, detail=str(e)) from e

@app.get("/health")
async def health_check():
    try:
        if not weave.is_initialized():
            raise TypeError("Weave not initialized")
        return {
            "status": "healthy",
            "weave_project": weave.get_project(),
            "weave_entity": os.getenv("WANDB_ENTITY", "default")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

"""
FastAPI server that runs inside the Docker container to execute Python code.
"""

import pickle
import base64
from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from subprocess_executor import execute_code_in_subprocess


class CodeRequest(BaseModel):
    code: str
    timeout: float = 30.0


class CodeResponse(BaseModel):
    success: bool
    result: Optional[str] = None  # Base64 encoded pickled result
    error: Optional[str] = None
    error_type: Optional[str] = None


app = FastAPI(title="Python Code Executor", version="1.0.0")


@app.post("/execute", response_model=CodeResponse)
async def execute_code(request: CodeRequest) -> CodeResponse:
    """Execute Python code using subprocess for better isolation and resource cleanup."""
    
    try:
        # Use the subprocess executor directly for isolation and cleanup
        result, error = execute_code_in_subprocess(request.code, timeout=request.timeout)
        
        if error is None:
            # Success - serialize the result for the response
            serialized_result = base64.b64encode(pickle.dumps(result)).decode('utf-8')
            return CodeResponse(
                success=True,
                result=serialized_result
            )
        else:
            # Error occurred during execution
            return CodeResponse(
                success=False,
                error=str(error),
                error_type=type(error).__name__
            )
            
    except Exception as e:
        # Unexpected error in the executor
        return CodeResponse(
            success=False,
            error=f"Executor error: {e}",
            error_type=type(e).__name__
        )


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7934)

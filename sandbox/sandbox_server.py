"""
FastAPI server that runs inside the Docker container to execute Python code.
"""

import pickle
import base64
import traceback
from typing import Any, Dict, Optional
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn


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
    """Execute Python code and return the result."""
    try:
        # Wrap the user code in a function
        wrapper_code = f"""
def user_function():
{chr(10).join('    ' + line for line in request.code.split(chr(10)))}

result = user_function()
"""
        
        # Create a namespace for execution
        namespace: Dict[str, Any] = {}
        
        # Execute the code
        exec(wrapper_code, namespace)
        result = namespace.get('result')
        
        # Serialize the result
        serialized_result = base64.b64encode(pickle.dumps(result)).decode('utf-8')
        
        return CodeResponse(
            success=True,
            result=serialized_result
        )
        
    except Exception as e:
        # Capture the full traceback
        error_traceback = traceback.format_exc()
        
        return CodeResponse(
            success=False,
            error=str(e),
            error_type=type(e).__name__
        )


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7934)

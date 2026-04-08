"""
Server entry-point for multi-mode OpenEnv deployment.
Exposes the FastAPI app from router_env.server.
"""

import uvicorn
from router_env.server import app


def main() -> None:
    """Run the RouterEnv-v1 server."""
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()

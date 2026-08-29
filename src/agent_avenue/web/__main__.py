"""Run the one-process local QA server."""

import uvicorn


def main() -> None:
    uvicorn.run("agent_avenue.web.app:app", host="0.0.0.0", port=8000, workers=1)


if __name__ == "__main__":
    main()

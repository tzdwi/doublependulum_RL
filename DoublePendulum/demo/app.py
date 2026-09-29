"""
Pendulum Game API Endpoint

REST API endpoints for interacting with a Session object
    - GET /models : see available RL models to play with
	- POST /session : Creates new session, gives cookies for continued interaction
	- POST /kick?scale=XX : kicks the pendulum with a XX N force
	- POST /step : increments timestep

Run from the repo root with: uvicorn DoublePendulum.demo.app:app --reload
"""

import io
import os
import uuid

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from pathlib import Path

import imageio.v3 as iio
from fastapi import APIRouter, Cookie, FastAPI, HTTPException, Response, status
from fastapi.staticfiles import StaticFiles

from .policies import MODEL_IDS, POLICIES
from .session import Session

router = APIRouter(prefix="/pendulum", tags=["pendulum_game"])

# session id (cookie) -> Session
SESSION_DICT: dict[str, Session] = {}
COOKIE = "pendulum_session"


def _frame_response(frame, session_status="holding", set_cookie=None):
	"""Encode an HxWx3 frame as JPEG. Status rides along in a header."""
	buf = io.BytesIO()
	iio.imwrite(buf, frame, extension=".jpg")
	resp = Response(buf.getvalue(), media_type="image/jpeg", headers={"X-Status": session_status})
	if set_cookie:
		resp.set_cookie(COOKIE, set_cookie, httponly=True, samesite="lax")
	return resp


def _get_session(session_id):
	if session_id is None or session_id not in SESSION_DICT:
		raise HTTPException(status.HTTP_404_NOT_FOUND, "No session; POST /pendulum/session first")
	return SESSION_DICT[session_id]


@router.get("/models", status_code=status.HTTP_200_OK)
def get_models():
	return {model_id: POLICIES[model_id]["Label"] for model_id in MODEL_IDS}


@router.post("/session", status_code=status.HTTP_200_OK)
def new_session(request: dict, pendulum_session: str | None = Cookie(default=None)):
	model_id = request.get("model")
	if model_id not in MODEL_IDS:
		raise HTTPException(status.HTTP_400_BAD_REQUEST, f"model must be one of {MODEL_IDS}")
	# drop any old session for this browser
	SESSION_DICT.pop(pendulum_session, None)
	session_id = uuid.uuid4().hex
	session = Session(model_id)
	SESSION_DICT[session_id] = session
	return _frame_response(session.start(), set_cookie=session_id)


@router.post("/kick", status_code=status.HTTP_200_OK)
def kick(scale: float = 0.1, pendulum_session: str | None = Cookie(default=None)):
	session = _get_session(pendulum_session)
	return _frame_response(session.kick(scale=scale), "working")


@router.post("/step", status_code=status.HTTP_200_OK)
def step(pendulum_session: str | None = Cookie(default=None)):
	session = _get_session(pendulum_session)
	frame, session_status = session.step()
	return _frame_response(frame, session_status)


app = FastAPI(title="Double Pendulum Demo")
app.include_router(router)
# Mounted last so /pendulum/* is handled by the router, not the static files.
app.mount("/", StaticFiles(directory=Path(__file__).parent / "static", html=True), name="static")

from __future__ import annotations

from requests import Session
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .config import DEFAULT_REQUEST_TIMEOUT, DEFAULT_USER_AGENT


def build_session(user_agent: str = DEFAULT_USER_AGENT) -> Session:
    session = Session()
    retry = Retry(
        total=3,
        connect=3,
        read=3,
        status=3,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({"User-Agent": user_agent})
    return session


def fetch_text(session: Session, url: str, timeout: int = DEFAULT_REQUEST_TIMEOUT) -> str:
    response = session.get(url, timeout=timeout)
    response.raise_for_status()
    apparent = getattr(response, "apparent_encoding", None)
    if apparent:
        response.encoding = apparent
    return response.text

"""Password hashing and JWT issuing/verification."""

from datetime import UTC, datetime, timedelta

import bcrypt
import jwt
from pydantic import BaseModel

from app.core.security.policy import Role

_ALGORITHM = "HS256"


class Principal(BaseModel):
    user_id: str
    username: str
    role: Role


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode(), hashed.encode())


def issue_token(principal: Principal, secret: str, ttl_minutes: int) -> str:
    now = datetime.now(UTC)
    payload = {
        "sub": principal.user_id,
        "username": principal.username,
        "role": principal.role.value,
        "iat": now,
        "exp": now + timedelta(minutes=ttl_minutes),
    }
    return jwt.encode(payload, secret, algorithm=_ALGORITHM)


def decode_token(token: str, secret: str) -> Principal:
    """Raises jwt.PyJWTError on any invalid / expired token."""
    payload = jwt.decode(token, secret, algorithms=[_ALGORITHM])
    return Principal(user_id=payload["sub"], username=payload["username"], role=payload["role"])

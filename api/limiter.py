"""Shared slowapi Limiter instance.

Split out from api/main.py so route modules can import it for the
`@limiter.limit(...)` decorator without a circular import (main.py includes
the route routers, so routes can't import the limiter from main).
"""

from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

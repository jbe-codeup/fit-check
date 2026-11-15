import os

# Unit tests are offline: an empty key keeps every model call in mock mode, and stops
# fitcheck.config from filling the key in from .env.
os.environ["ANTHROPIC_API_KEY"] = ""

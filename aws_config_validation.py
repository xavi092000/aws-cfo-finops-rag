"""Small side-effect-free checks for account-specific S3 destinations."""

from urllib.parse import urlsplit


def require_s3_uri(value: str, variable_name: str) -> str:
    """Validate a configured S3 prefix without querying AWS or reading secrets."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{variable_name} is required for this AWS operation; see .env.example")
    uri = value.strip()
    parts = urlsplit(uri)
    if (parts.scheme != "s3" or not parts.netloc or not parts.path.strip("/")
            or parts.query or parts.fragment or parts.username or parts.password
            or any(c.isspace() for c in uri) or parts.netloc == "YOUR-BUCKET"):
        raise ValueError(f"{variable_name} must be an S3 URI with a real bucket and prefix; see .env.example")
    return uri
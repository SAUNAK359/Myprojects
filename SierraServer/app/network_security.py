import ipaddress
import socket
from urllib.parse import urlsplit


def validate_public_https_endpoint(endpoint: str) -> str:
    value = endpoint.strip().rstrip("/")
    parsed = urlsplit(value)
    if parsed.scheme.lower() != "https":
        raise ValueError("provider endpoint must use HTTPS")
    if parsed.username or parsed.password:
        raise ValueError("provider endpoint must not contain URL credentials")
    if not parsed.hostname:
        raise ValueError("provider endpoint must include a hostname")
    if parsed.fragment:
        raise ValueError("provider endpoint must not contain a fragment")
    if parsed.port not in (None, 443):
        raise ValueError("provider endpoint must use standard HTTPS port 443")

    host = parsed.hostname.rstrip(".").lower()
    if host == "localhost" or host.endswith(".localhost"):
        raise ValueError("local provider endpoints are not allowed")

    try:
        addresses = {item[4][0] for item in socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)}
    except socket.gaierror as exc:
        raise ValueError("provider hostname could not be resolved") from exc

    if not addresses:
        raise ValueError("provider hostname did not resolve")

    for value_ip in addresses:
        ip = ipaddress.ip_address(value_ip.split("%", 1)[0])
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_multicast
            or ip.is_reserved
            or ip.is_unspecified
        ):
            raise ValueError("provider endpoint resolves to a non-public network address")
    return value

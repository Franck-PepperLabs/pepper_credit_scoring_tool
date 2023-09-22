def build_url(
    scheme: str | None = "http",
    hostname: str | None = "localhost",
    port: int | None = None,
    path: str | None = "/",
    query: dict | None = None,
    fragment: str | None = None 
) -> str:
    """
    Build a URL by combining its components.

    Parameters
    ----------
        scheme (str, optional): The URL scheme (e.g., "http", "https").
            Defaults to "http".
        hostname (str, optional): The hostname or domain name.
            Defaults to "localhost".
        port (int, optional): The port number, if applicable.
        path (str, optional): The path to the resource.
            Defaults to "/".
        query (dict, optional): A dictionary of query parameters.
        fragment (str, optional): The URL fragment identifier.

    Returns
    -------
        str: The constructed URL.

    Raises
    ------
        ValueError: If the types of input parameters are incorrect.
    """
    if not isinstance(scheme, str):
        raise ValueError("Scheme must be a string")
    if not isinstance(hostname, str):
        raise ValueError("Hostname must be a string")
    if port is not None and not isinstance(port, int):
        raise ValueError("Port must be an integer")
    if not isinstance(path, str):
        raise ValueError("Path must be a string")

    url = f"{scheme}://{hostname}"
    if port:
        url += f":{port}"
    url += path
    if query:
        url += "?" + "&".join([f"{k}={v}" for k, v in query.items()])
    if fragment:
        url += f"#{fragment}"
    return url

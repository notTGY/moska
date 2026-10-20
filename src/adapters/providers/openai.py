"""ChatGPT plan authorization for Moska, independent of Codex credentials.

Protocol: https://developers.openai.com/siwc/token-sharing-open-source/sign-in
Credentials live outside the repository and must never enter model context.
"""

import asyncio
import base64
import fcntl
import hashlib
import json
import os
import secrets
import tempfile
import time
import webbrowser
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlsplit
from uuid import uuid4

import httpx
import jwt

ISSUER = "https://auth.openai.com"
RESOURCE = "https://api.openai.com/v1"
SCOPES = "openid profile email offline_access resource.invoke chatgpt.tokens.use.direct"
PLAN_SCOPE = "chatgpt.tokens.use.direct"


from .base import ProviderError as ConnectionError


@dataclass(frozen=True)
class Account:
    client_id: str
    subject: str
    email: str
    plan_enabled: bool


class OpenAIConnection:
    def __init__(self, directory: Path | None = None):
        self.directory = directory or Path(
            os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config"))
        ) / "moska" / "openai"
        self.path = self.directory / "connections.json"

    def _read(self):
        if not self.path.exists():
            return {"host_id": f"urn:uuid:{uuid4()}", "active": None, "accounts": {}}
        try:
            data = json.loads(self.path.read_text())
            if not isinstance(data["accounts"], dict) or not data["host_id"].startswith("urn:uuid:"):
                raise ValueError
            return data
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            raise ConnectionError("Could not read the saved OpenAI connection.") from None

    def _save(self, data):
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        fd, name = tempfile.mkstemp(dir=self.directory, prefix=".connection-")
        try:
            with os.fdopen(fd, "w") as file:
                os.fchmod(file.fileno(), 0o600)
                json.dump(data, file)
                file.flush()
                os.fsync(file.fileno())
            os.replace(name, self.path)
        finally:
            if os.path.exists(name):
                os.unlink(name)

    @contextmanager
    def _locked(self):
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        fd = os.open(self.directory / ".lock", os.O_CREAT | os.O_RDWR, 0o600)
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise ConnectionError("Another Moska process is updating this connection. Try again when it finishes.") from None
            yield
        finally:
            os.close(fd)

    @staticmethod
    def _account(record):
        return Account(
            client_id=record["client_id"], subject=record["subject"],
            email=record.get("email", ""), plan_enabled=PLAN_SCOPE in record["scopes"],
        )

    def accounts(self):
        """Return public account information only, never tokens."""
        return [self._account(record) for record in self._read()["accounts"].values()]

    def current_account(self):
        data = self._read()
        record = data["accounts"].get(data["active"])
        return self._account(record) if record else None

    @staticmethod
    async def _discovery(client):
        response = await client.get(f"{ISSUER}/.well-known/openid-configuration")
        response.raise_for_status()
        discovery = response.json()
        if discovery.get("issuer") != ISSUER:
            raise ConnectionError("OpenAI returned an unexpected identity issuer.")
        for name in ("authorization_endpoint", "token_endpoint", "jwks_uri"):
            endpoint = urlsplit(discovery[name])
            if endpoint.scheme != "https" or endpoint.netloc != "auth.openai.com":
                raise ConnectionError("OpenAI returned an unexpected authorization endpoint.")
        return discovery

    @staticmethod
    async def _identity(client, discovery, token, client_id, nonce=None):
        try:
            header = jwt.get_unverified_header(token)
            algorithm = header["alg"]
            if algorithm not in ("RS256", "ES256"):
                raise ValueError
            response = await client.get(discovery["jwks_uri"])
            response.raise_for_status()
            keys = response.json()["keys"]
            key = next(key for key in keys if key.get("kid") == header["kid"])
            claims = jwt.decode(
                token, jwt.PyJWK.from_dict(key, algorithm=algorithm).key,
                algorithms=[algorithm], audience=client_id, issuer=ISSUER,
                leeway=5, options={"require": ["sub", "exp", "iat", "iss", "aud"]},
            )
            if not isinstance(claims["sub"], str) or not claims["sub"]:
                raise ValueError
            if claims.get("azp", client_id) != client_id:
                raise ValueError
            if nonce is not None and not secrets.compare_digest(claims.get("nonce", ""), nonce):
                raise ValueError
            return claims
        except (jwt.PyJWTError, ValueError, KeyError, TypeError, StopIteration):
            raise ConnectionError("OpenAI identity verification failed. Run /connect again.") from None

    @staticmethod
    async def _tokens(client, endpoint, form):
        response = await client.post(endpoint, data=form)
        if response.is_error:
            # Do not surface response bodies, request URLs, or OAuth artifacts.
            raise ConnectionError(f"OpenAI authorization failed (HTTP {response.status_code}). Run /connect again.")
        tokens = response.json()
        if not isinstance(tokens.get("access_token"), str) or not tokens["access_token"]:
            raise ConnectionError("OpenAI did not return an access token.")
        if str(tokens.get("token_type", "")).lower() != "bearer":
            raise ConnectionError("OpenAI returned an unsupported token type.")
        return tokens

    @staticmethod
    def _updated(record, tokens):
        scopes = tokens.get("scope")
        updated = {
            **record,
            **{key: tokens[key] for key in ("access_token", "refresh_token", "id_token") if key in tokens},
            "scopes": scopes.split() if isinstance(scopes, str) else record.get("scopes", []),
            "expires_at": time.time() + float(tokens["expires_in"]),
        }
        return updated

    async def sign_in(
        self, notify: Callable[[str], None], *, new_account=False,
        client_id: str | None = None, timeout=300, browser=webbrowser.open,
    ) -> Account:
        """Register/reauthorize in the browser; notify receives only a local URL.

        No credentials are persisted until identity verification succeeds. A
        returning registration must retain the same client ID and subject.
        """
        try:
            with self._locked():
                data = self._read()
                selected = client_id or data["active"]
                existing = None if new_account else data["accounts"].get(selected)
                if client_id and existing is None and not new_account:
                    raise ConnectionError("Unknown saved connection. Use /connect accounts to list them.")
                self._save(data)  # Persist the stable host ID even if sign-in is cancelled.
                async with httpx.AsyncClient(timeout=30) as client:
                    discovery = await self._discovery(client)
                    verifier = secrets.token_urlsafe(64)
                    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
                    state, nonce = secrets.token_urlsafe(32), secrets.token_urlsafe(32)
                    result = asyncio.get_running_loop().create_future()
                    authorization_url = ""

                    async def handle(reader, writer):
                        status, body, location = "400 Bad Request", "This sign-in link has expired. Return to Moska and run /connect again.", None
                        try:
                            request = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), 5)
                            method, target, _ = request.split(b"\r\n", 1)[0].decode("ascii").split(" ")
                            url = urlsplit(target)
                            if method == "GET" and url.path == "/auth/start" and not result.done():
                                status, body, location = "302 Found", "Continue with ChatGPT.", authorization_url
                            elif method == "GET" and url.path == "/auth/callback" and not result.done():
                                params = parse_qs(url.query, keep_blank_values=True)
                                if any(len(value) != 1 for value in params.values()):
                                    body = "The callback contains duplicate parameters. Return to Moska and run /connect again."
                                    raise ValueError
                                params = {key: value[0] for key, value in params.items()}
                                if not secrets.compare_digest(params.get("state", ""), state):
                                    body = "This callback belongs to a different sign-in attempt. Return to Moska and use its latest sign-in link."
                                    raise ValueError
                                result.set_result(params)
                                status, body = "200 OK", "Return to Moska to finish connecting. You may close this tab."
                        except asyncio.LimitOverrunError:
                            body = "The browser request is too large. Return to Moska and run /connect again."
                        except (ValueError, TypeError, UnicodeError, asyncio.IncompleteReadError, TimeoutError):
                            pass
                        finally:
                            headers = f"HTTP/1.1 {status}\r\nContent-Type: text/plain; charset=utf-8\r\nCache-Control: no-store\r\nReferrer-Policy: no-referrer\r\nConnection: close\r\n"
                            if location:
                                headers += f"Location: {location}\r\n"
                            payload = body.encode()
                            writer.write((headers + f"Content-Length: {len(payload)}\r\n\r\n").encode() + payload)
                            try:
                                await writer.drain()
                            except OSError:
                                pass
                            writer.close()
                            await writer.wait_closed()

                    # Browsers may send large localhost cookie headers shared
                    # with unrelated local apps; do not reject a valid callback.
                    server = await asyncio.start_server(handle, "127.0.0.1", 0, limit=131072)
                    async with server:
                        port = server.sockets[0].getsockname()[1]
                        redirect = f"http://127.0.0.1:{port}/auth/callback"
                        params = {
                            "client_id": existing["client_id"] if existing else "dynamic_agent_client",
                            "ext_agent_host_id": data["host_id"], "response_type": "code",
                            "redirect_uri": redirect, "scope": SCOPES, "resource": RESOURCE,
                            "state": state, "nonce": nonce,
                            "code_challenge_method": "S256", "code_challenge": challenge,
                        }
                        if existing:
                            params["id_token_hint"] = existing["id_token"]
                        else:
                            params["agent_name_hint"] = "Moska"
                        authorization_url = discovery["authorization_endpoint"] + "?" + urlencode(params)
                        start_url = f"http://127.0.0.1:{port}/auth/start"
                        notify(start_url)
                        try:
                            await asyncio.to_thread(browser, start_url)
                        except OSError:
                            pass  # The user can click the local URL shown by notify.
                        try:
                            callback = await asyncio.wait_for(result, timeout)
                        except TimeoutError:
                            raise ConnectionError("Sign-in timed out. Run /connect to try again.") from None

                    if "error" in callback:
                        raise ConnectionError("Sign-in was declined or cancelled. Run /connect to try again.")
                    issued = callback.get("client_id") or (existing["client_id"] if existing else None)
                    if not issued or issued == "dynamic_agent_client":
                        raise ConnectionError("OpenAI did not complete client registration. Run /connect again.")
                    if existing and issued != existing["client_id"]:
                        raise ConnectionError("The callback changed the selected connection's client ID.")
                    if not callback.get("code"):
                        raise ConnectionError("OpenAI did not return an authorization code.")
                    tokens = await self._tokens(client, discovery["token_endpoint"], {
                        "grant_type": "authorization_code", "client_id": issued,
                        "code": callback["code"], "code_verifier": verifier,
                        "redirect_uri": redirect, "resource": RESOURCE,
                    })
                    identity = await self._identity(client, discovery, tokens["id_token"], issued, nonce)
                    registered = data["accounts"].get(issued)
                    if registered and identity["sub"] != registered["subject"]:
                        raise ConnectionError("Signed-in identity differs from the selected account. Use /connect new.")
                    record = self._updated({
                        **(registered or {}),
                        "client_id": issued, "subject": identity["sub"],
                        "email": identity.get("email", ""), "issuer": ISSUER,
                    }, tokens)
                    if not record.get("refresh_token"):
                        raise ConnectionError("OpenAI did not grant offline access. Run /connect again.")
                    data["accounts"][issued] = record
                    data["active"] = issued
                    self._save(data)
                    return self._account(record)
        except httpx.HTTPError:
            raise ConnectionError("Could not reach OpenAI. Check your connection and run /connect again.") from None
        except (KeyError, ValueError, TypeError):
            raise ConnectionError("OpenAI returned an incomplete authorization response. Run /connect again.") from None
        except OSError:
            raise ConnectionError("Could not open the sign-in callback or save the connection locally.") from None

    async def access_token(self) -> str:
        """Get an active token, serializing refresh and atomic rotation on disk."""
        try:
            with self._locked():
                data = self._read()
                record = data["accounts"].get(data["active"])
                if not record:
                    raise ConnectionError("Connect your ChatGPT account with /connect first.")
                if PLAN_SCOPE not in record["scopes"]:
                    raise ConnectionError("ChatGPT plan usage is not enabled. Run /connect and allow plan usage.")
                if record["expires_at"] > time.time() + 30:
                    return record["access_token"]
                async with httpx.AsyncClient(timeout=30) as client:
                    discovery = await self._discovery(client)
                    tokens = await self._tokens(client, discovery["token_endpoint"], {
                        "grant_type": "refresh_token", "client_id": record["client_id"],
                        "refresh_token": record["refresh_token"], "resource": RESOURCE,
                    })
                    if not tokens.get("refresh_token"):
                        raise ConnectionError("OpenAI did not return a replacement refresh token. Run /connect again.")
                    if "id_token" in tokens:
                        identity = await self._identity(client, discovery, tokens["id_token"], record["client_id"])
                        if identity["sub"] != record["subject"]:
                            raise ConnectionError("Refreshed identity does not match the saved connection.")
                    updated = self._updated(record, tokens)
                    data["accounts"][record["client_id"]] = updated
                    self._save(data)
                    if PLAN_SCOPE not in updated["scopes"]:
                        raise ConnectionError("ChatGPT plan usage is no longer enabled. Run /connect again.")
                    return updated["access_token"]
        except httpx.HTTPError:
            raise ConnectionError("Could not refresh the OpenAI connection. Try /connect again.") from None
        except (OSError, KeyError, ValueError, TypeError):
            raise ConnectionError("Could not renew the saved connection. Run /connect again.") from None

    def selected_model(self) -> str | None:
        data = self._read()
        record = data['accounts'].get(data['active'])
        return record.get('model') if record else None

    def selected_reasoning_effort(self) -> str | None:
        data = self._read()
        record = data['accounts'].get(data['active'])
        return record.get('reasoning_effort') if record else None

    async def list_models(self) -> list[dict]:
        """Fetch account-specific choices in the provider's display order."""
        token = await self.access_token()
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.get(RESOURCE + '/models', headers={'Authorization': 'Bearer ' + token})
                if response.is_error:
                    raise ConnectionError(f'Could not load OpenAI models (HTTP {response.status_code}). Try /connect again.')
                models = response.json()['models']
                return [
                    {
                        'slug': model['slug'],
                        'display_name': model.get('display_name') or model['slug'],
                        'supported_reasoning_levels': model.get('supported_reasoning_levels') or [],
                        'default_reasoning_level': model.get('default_reasoning_level'),
                    }
                    for model in models if model.get('visibility') == 'list'
                ]
        except httpx.HTTPError:
            raise ConnectionError('Could not reach the OpenAI model catalog. Try /model again.') from None
        except (KeyError, TypeError, ValueError):
            raise ConnectionError('OpenAI returned an invalid model catalog.') from None

    async def select_model(self, slug: str, reasoning_effort: str | None = None) -> None:
        account = self.current_account()
        if not account:
            raise ConnectionError('Use /connect before choosing a model.')
        models = await self.list_models()
        model = next((model for model in models if model['slug'] == slug), None)
        if model is None:
            raise ConnectionError('That model is not available to this account. Use /model to choose one.')
        efforts = [level['effort'] for level in model['supported_reasoning_levels']]
        if reasoning_effort is None and efforts:
            reasoning_effort = model['default_reasoning_level'] or efforts[0]
        if reasoning_effort is not None and reasoning_effort not in efforts:
            raise ConnectionError('That reasoning effort is not supported by this model. Use /model to choose one.')
        with self._locked():
            data = self._read()
            if data['active'] != account.client_id:
                raise ConnectionError('The active account changed. Open /model again.')
            data['accounts'][account.client_id]['model'] = slug
            data['accounts'][account.client_id]['reasoning_effort'] = reasoning_effort
            self._save(data)

import asyncio
import base64
import hashlib
import json
import stat
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.parse import parse_qs, urlencode, urlsplit

import httpx
import jwt
from cryptography.hazmat.primitives.asymmetric import rsa

from adapters.providers.openai import ConnectionError, ISSUER, PLAN_SCOPE, OpenAIConnection

REAL_ASYNC_CLIENT = httpx.AsyncClient


class OpenAIConnectionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.connection = OpenAIConnection(Path(self.temp.name) / 'openai')
        self.key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        self.jwk = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(self.key.public_key()))
        self.jwk['kid'] = 'test-key'
        self.issued = 'oaiapp_test'
        self.subject = 'test-subject'
        self.claim_changes = {}
        self.token_changes = {}
        self.callbacks = []
        self.forms = []
        self.bad_state_first = False
        self.repeat_callback = False
        self.callback_error = None
        self.callback_client = None
        self.callback_cookie_bytes = 0
        self.catalog = [
            {'slug': 'first', 'display_name': 'First model', 'visibility': 'list'},
            {'slug': 'hidden', 'display_name': 'Hidden model', 'visibility': 'hide'},
            {'slug': 'second', 'display_name': 'Second model', 'visibility': 'list',
             'supported_reasoning_levels': [{'effort': 'low'}, {'effort': 'high'}],
             'default_reasoning_level': 'low'},
        ]
        self.transport = httpx.MockTransport(self.handle)
        self.patch = patch('adapters.providers.openai.httpx.AsyncClient', side_effect=self.make_client)
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def make_client(self, **kwargs):
        return REAL_ASYNC_CLIENT(transport=self.transport, **kwargs)

    def handle(self, request):
        if request.url.path == '/v1/models':
            self.assertEqual(request.headers['authorization'], 'Bearer test-access')
            return httpx.Response(200, json={'models': self.catalog})
        if request.url.path == '/.well-known/openid-configuration':
            return httpx.Response(200, json={
                'issuer': ISSUER,
                'authorization_endpoint': ISSUER + '/api/accounts/authorize',
                'token_endpoint': ISSUER + '/api/accounts/oauth/token',
                'jwks_uri': ISSUER + '/.well-known/jwks.json',
            })
        if request.url.path == '/.well-known/jwks.json':
            return httpx.Response(200, json={'keys': [self.jwk]})
        if request.url.path == '/api/accounts/oauth/token':
            form = {key: value[0] for key, value in parse_qs(request.content.decode()).items()}
            self.forms.append(form)
            if form['grant_type'] == 'authorization_code':
                authorization = self.callbacks[-1]
                challenge = base64.urlsafe_b64encode(hashlib.sha256(form['code_verifier'].encode()).digest()).rstrip(b'=').decode()
                self.assertEqual(challenge, authorization['code_challenge'])
                self.assertEqual(form['redirect_uri'], authorization['redirect_uri'])
                self.assertEqual(form['client_id'], self.issued)
            tokens = {
                'access_token': 'test-access', 'refresh_token': 'test-refresh',
                'id_token': self.id_token, 'expires_in': 3600,
                'scope': 'openid offline_access resource.invoke ' + PLAN_SCOPE,
                'token_type': 'Bearer',
                **self.token_changes,
            }
            return httpx.Response(200, json=tokens)
        raise AssertionError('Unexpected outbound request')

    def browser(self, start_url):
        with httpx.Client(trust_env=False) as client:
            response = client.get(start_url)
            self.assertEqual(response.status_code, 302)
            authorization = {k: v[0] for k, v in parse_qs(urlsplit(response.headers['location']).query).items()}
            self.callbacks.append(authorization)
            now = int(time.time())
            self.id_token = jwt.encode({
                'iss': ISSUER, 'aud': self.issued, 'sub': self.subject,
                'exp': now + 3600, 'iat': now, 'nonce': authorization['nonce'],
                'email': 'test@example.com', **self.claim_changes,
            }, self.key, algorithm='RS256', headers={'kid': 'test-key'})
            params = {'state': authorization['state'], 'code': 'test-code'}
            if authorization['client_id'] == 'dynamic_agent_client' or self.callback_client:
                params['client_id'] = self.callback_client or self.issued
            if self.callback_error:
                params['error'] = self.callback_error
            redirect = authorization['redirect_uri']
            if self.bad_state_first:
                invalid = client.get(redirect, params={**params, 'state': 'wrong-state'})
                self.assertEqual(invalid.status_code, 400)
            result = client.get(redirect, params=params, headers={
                'Cookie': 'test=' + 'x' * self.callback_cookie_bytes,
            })
            self.assertEqual(result.status_code, 200)
            if self.repeat_callback:
                replay = client.get(redirect, params=params)
                self.assertEqual(replay.status_code, 400)
        return True

    async def sign_in(self, **kwargs):
        urls = []
        account = await self.connection.sign_in(urls.append, browser=self.browser, **kwargs)
        self.assertTrue(all(url.startswith('http://127.0.0.1:') and url.endswith('/auth/start') for url in urls))
        return account

    async def test_registration_pkce_validation_and_private_storage(self):
        account = await self.sign_in()
        self.assertTrue(account.plan_enabled)
        self.assertEqual(account.email, 'test@example.com')
        self.assertNotIn('test-access', repr(account))
        self.assertEqual(stat.S_IMODE(self.connection.path.stat().st_mode), 0o600)
        self.assertEqual(stat.S_IMODE(self.connection.directory.stat().st_mode), 0o700)
        self.assertEqual(self.connection.current_account(), account)
        self.assertEqual(await self.connection.access_token(), 'test-access')
        self.assertEqual(len(self.forms), 1)  # A valid token does not refresh.

    async def test_bad_state_and_replayed_callback_are_not_redeemed(self):
        self.bad_state_first = self.repeat_callback = True
        await self.sign_in()
        self.assertEqual(len(self.forms), 1)

    async def test_large_browser_cookie_headers_do_not_reject_valid_callback(self):
        self.callback_cookie_bytes = 32768
        account = await self.sign_in()
        self.assertTrue(account.plan_enabled)

    async def test_reauthorization_reuses_client_and_host(self):
        await self.sign_in()
        await self.sign_in()
        first, second = self.callbacks
        self.assertEqual(first['client_id'], 'dynamic_agent_client')
        self.assertEqual(second['client_id'], self.issued)
        self.assertEqual(first['ext_agent_host_id'], second['ext_agent_host_id'])
        self.assertIn('id_token_hint', second)
        self.assertNotIn('agent_name_hint', second)
        self.assertNotEqual(first['state'], second['state'])
        self.assertNotEqual(first['nonce'], second['nonce'])

    async def test_reauthorization_rejects_changed_identity_without_overwrite(self):
        account = await self.sign_in()
        before = self.connection.path.read_bytes()
        self.subject = 'different-subject'
        with self.assertRaisesRegex(ConnectionError, 'identity differs'):
            await self.sign_in()
        self.assertEqual(self.connection.path.read_bytes(), before)
        self.assertEqual(self.connection.current_account(), account)

    async def test_reauthorization_rejects_changed_client(self):
        await self.sign_in()
        self.callback_client = 'oaiapp_different'
        with self.assertRaisesRegex(ConnectionError, 'client ID'):
            await self.sign_in()
        self.assertEqual(len(self.forms), 1)

    async def test_separate_accounts_are_retained(self):
        first = await self.sign_in()
        self.issued, self.subject = 'oaiapp_second', 'second-subject'
        second = await self.sign_in(new_account=True)
        self.assertEqual(self.connection.accounts(), [first, second])
        self.assertEqual(self.connection.current_account(), second)
        self.issued, self.subject = first.client_id, first.subject
        await self.sign_in(client_id=first.client_id)
        self.assertEqual(self.connection.current_account(), first)

    async def test_invalid_id_token_claims_rejected(self):
        for changed in ({'nonce': 'wrong'}, {'iss': 'https://wrong.example'},
                        {'aud': 'wrong-client'}, {'exp': int(time.time()) - 60},
                        {'sub': ''}, {'azp': 'wrong-client'}):
            with self.subTest(changed=changed):
                self.claim_changes = changed
                with self.assertRaisesRegex(ConnectionError, 'identity verification'):
                    await self.sign_in()
                self.assertIsNone(self.connection.current_account())

    async def test_invalid_signature_rejected(self):
        self.jwk = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(
            rsa.generate_private_key(public_exponent=65537, key_size=2048).public_key()
        ))
        self.jwk['kid'] = 'test-key'
        with self.assertRaisesRegex(ConnectionError, 'identity verification'):
            await self.sign_in()

    async def test_declined_signin_never_exchanges_code(self):
        self.callback_error = 'access_denied'
        with self.assertRaisesRegex(ConnectionError, 'declined'):
            await self.sign_in()
        self.assertEqual(self.forms, [])

    async def test_missing_plan_permission_is_not_inference_access(self):
        self.token_changes = {'scope': 'openid offline_access'}
        account = await self.sign_in()
        self.assertFalse(account.plan_enabled)
        with self.assertRaisesRegex(ConnectionError, 'not enabled'):
            await self.connection.access_token()

    async def test_refresh_rotates_tokens_and_preserves_registration(self):
        account = await self.sign_in()
        data = self.connection._read()
        data['accounts'][account.client_id]['expires_at'] = 0
        self.connection._save(data)
        self.token_changes = {'access_token': 'replacement-access', 'refresh_token': 'replacement-refresh'}
        self.assertEqual(await self.connection.access_token(), 'replacement-access')
        self.assertEqual(self.forms[-1], {
            'grant_type': 'refresh_token', 'client_id': account.client_id,
            'refresh_token': 'test-refresh', 'resource': 'https://api.openai.com/v1',
        })
        saved = self.connection._read()['accounts'][account.client_id]
        self.assertEqual(saved['refresh_token'], 'replacement-refresh')
        self.assertEqual(self.connection.current_account(), account)

    async def test_timeout_and_cancel_close_callback_listener(self):
        for cancel in (False, True):
            with self.subTest(cancel=cancel):
                urls = []
                task = asyncio.create_task(self.connection.sign_in(
                    urls.append, browser=lambda url: True, timeout=.15,
                ))
                while not urls:
                    await asyncio.sleep(.01)
                if cancel:
                    task.cancel()
                    with self.assertRaises(asyncio.CancelledError):
                        await task
                else:
                    with self.assertRaisesRegex(ConnectionError, 'timed out'):
                        await task
                async with REAL_ASYNC_CLIENT(trust_env=False) as client:
                    with self.assertRaises(httpx.ConnectError):
                        await client.get(urls[0])
                self.assertIsNone(self.connection.current_account())

    async def test_connection_lock_prevents_parallel_refresh(self):
        await self.sign_in()
        with self.connection._locked():
            with self.assertRaisesRegex(ConnectionError, 'Another Moska process'):
                await self.connection.access_token()

    async def test_model_catalog_selection_and_reauthorization_preserve_choice(self):
        await self.sign_in()
        models = await self.connection.list_models()
        self.assertEqual([model['slug'] for model in models], ['first', 'second'])
        await self.connection.select_model('second', reasoning_effort='high')
        self.assertEqual(self.connection.selected_model(), 'second')
        self.assertEqual(self.connection.selected_reasoning_effort(), 'high')
        await self.sign_in()
        self.assertEqual(self.connection.selected_model(), 'second')
        self.assertEqual(self.connection.selected_reasoning_effort(), 'high')
        with self.assertRaisesRegex(ConnectionError, 'not supported'):
            await self.connection.select_model('second', reasoning_effort='unsupported')
        with self.assertRaisesRegex(ConnectionError, 'not available'):
            await self.connection.select_model('hidden')
        self.assertEqual(self.connection.selected_model(), 'second')

    async def test_model_selection_is_account_specific(self):
        first = await self.sign_in()
        await self.connection.select_model('first')
        self.issued, self.subject = 'oaiapp_second', 'second-subject'
        await self.sign_in(new_account=True)
        self.assertIsNone(self.connection.selected_model())
        self.issued, self.subject = first.client_id, first.subject
        await self.sign_in(client_id=first.client_id)
        self.assertEqual(self.connection.selected_model(), 'first')

    async def test_corrupt_store_is_not_overwritten(self):
        self.connection.directory.mkdir()
        self.connection.path.write_text('broken')
        with self.assertRaisesRegex(ConnectionError, 'Could not read'):
            await self.sign_in()
        self.assertEqual(self.connection.path.read_text(), 'broken')


if __name__ == '__main__':
    unittest.main()

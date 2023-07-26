import secrets
import time

from authlib.integrations.flask_oauth2 import (AuthorizationServer,
                                               ResourceProtector)
from authlib.oauth2.rfc6749 import (AuthorizationCodeMixin, ClientMixin,
                                    TokenMixin, grants, list_to_scope,
                                    scope_to_list)
from authlib.oauth2.rfc6750 import BearerTokenValidator
from authlib.oauth2.rfc7009 import RevocationEndpoint
from authlib.oauth2.rfc7636 import CodeChallenge
from flask import jsonify, request, session, redirect, abort
from werkzeug.security import gen_salt
from authlib.oauth2 import OAuth2Error
from .models import db, User
from bson import ObjectId

from .models import F, User, db


# With codes adapted from https://github.com/authlib/example-oauth2-server and the Authlib library.


class HasUser(db.DbObject):
    user_id = ObjectId
    
    @property
    def user(self):
        return User.first(F.id == self.user_id)
    
    def on_initialize(self):
        super().on_initialize()
        self.ensure_index('user_id')


class OAuth2AuthorizationCode(HasUser, AuthorizationCodeMixin):
    code = str
    client_id = str
    redirect_uri = str
    response_type = str
    scope = str
    nonce = str
    auth_time = lambda: int(time.time())
    code_challenge = str
    code_challenge_method = str

    _TTL = 300

    def is_expired(self):
        return self.auth_time + OAuth2AuthorizationCode._TTL < time.time()

    def get_redirect_uri(self):
        return self.redirect_uri

    def get_scope(self):
        return self.scope

    def get_auth_time(self):
        return self.auth_time

    def get_nonce(self):
        return self.nonce
    

class OAuth2Client(HasUser, ClientMixin):
    client_id = str
    client_secret = str
    client_id_issued_at = int
    client_secret_expires_at = int
    client_metadata = dict

    def on_initialize(self):
        super().on_initialize()
        self.ensure_index('client_id')

    @property
    def client_info(self):
        return dict(
            client_id=self.client_id,
            client_secret=self.client_secret,
            client_id_issued_at=self.client_id_issued_at,
            client_secret_expires_at=self.client_secret_expires_at,
        )
    
    @property
    def redirect_uris(self):
        return self.client_metadata.get('redirect_uris', [])

    @property
    def token_endpoint_auth_method(self):
        return self.client_metadata.get(
            'token_endpoint_auth_method',
            'client_secret_basic'
        )

    @property
    def grant_types(self):
        return self.client_metadata.get('grant_types', [])

    @property
    def response_types(self):
        return self.client_metadata.get('response_types', [])

    @property
    def client_name(self):
        return self.client_metadata.get('client_name')

    @property
    def client_uri(self):
        return self.client_metadata.get('client_uri')

    @property
    def logo_uri(self):
        return self.client_metadata.get('logo_uri')

    @property
    def scope(self):
        return self.client_metadata.get('scope', '')

    @property
    def contacts(self):
        return self.client_metadata.get('contacts', [])

    @property
    def tos_uri(self):
        return self.client_metadata.get('tos_uri')

    @property
    def policy_uri(self):
        return self.client_metadata.get('policy_uri')

    @property
    def jwks_uri(self):
        return self.client_metadata.get('jwks_uri')

    @property
    def jwks(self):
        return self.client_metadata.get('jwks', [])

    @property
    def software_id(self):
        return self.client_metadata.get('software_id')

    @property
    def software_version(self):
        return self.client_metadata.get('software_version')

    def get_client_id(self):
        return self.client_id

    def get_default_redirect_uri(self):
        if self.redirect_uris:
            return self.redirect_uris[0]

    def get_allowed_scope(self, scope):
        if not scope:
            return ''
        allowed = set(self.scope.split())
        scopes = scope_to_list(scope)
        return list_to_scope([s for s in scopes if s in allowed])

    def check_redirect_uri(self, redirect_uri):
        return redirect_uri in self.redirect_uris

    def check_client_secret(self, client_secret):
        return secrets.compare_digest(self.client_secret, client_secret)

    def check_endpoint_auth_method(self, method, endpoint):
        if endpoint == 'token':
            return self.token_endpoint_auth_method == method
        # TODO
        return True

    def check_response_type(self, response_type):
        return response_type in self.response_types

    def check_grant_type(self, grant_type):
        return grant_type in self.grant_types


class OAuth2Token(HasUser, TokenMixin):
    client_id = str
    token_type = str
    access_token = str
    refresh_token = str
    scope = str
    issued_at = lambda: int(time.time())
    access_token_revoked_at = int
    refresh_token_revoked_at = int
    expires_in = int

    def on_initialize(self):
        super().on_initialize()
        self.ensure_index('refresh_token')

    def check_client(self, client):
        return self.client_id == client.get_client_id()

    def get_scope(self):
        return self.scope

    def get_expires_in(self):
        return self.expires_in

    def is_revoked(self):
        return self.access_token_revoked_at or self.refresh_token_revoked_at

    def is_expired(self):
        if not self.expires_in:
            return False

        expires_at = self.issued_at + self.expires_in
        return expires_at < time.time()


class AuthorizationCodeGrant(grants.AuthorizationCodeGrant):
    TOKEN_ENDPOINT_AUTH_METHODS = [
        'client_secret_basic',
        'client_secret_post',
        'none',
    ]

    def save_authorization_code(self, code, request):
        code_challenge = request.data.get('code_challenge')
        code_challenge_method = request.data.get('code_challenge_method')
        auth_code = OAuth2AuthorizationCode(
            code=code,
            client_id=request.client.client_id,
            redirect_uri=request.redirect_uri,
            scope=request.scope,
            user_id=request.user.id,
            code_challenge=code_challenge,
            code_challenge_method=code_challenge_method,
        )
        auth_code.save()
        return auth_code

    def query_authorization_code(self, code, client):
        auth_code = OAuth2AuthorizationCode.first(
            F.code==code, F.client_id==client.client_id)
        if auth_code and not auth_code.is_expired():
            return auth_code

    def delete_authorization_code(self, authorization_code):
        authorization_code.delete()

    def authenticate_user(self, authorization_code):
        return User.first(F.id == authorization_code.user_id)


class PasswordGrant(grants.ResourceOwnerPasswordCredentialsGrant):
    def authenticate_user(self, username, password):
        if User.authenticate(username, password):
            return User.first(F.username==username)


class RefreshTokenGrant(grants.RefreshTokenGrant):
    def authenticate_refresh_token(self, refresh_token):
        token = OAuth2Token.first(F.refresh_token==refresh_token)
        if token and token.is_refresh_token_active():
            return token

    def authenticate_user(self, credential):
        return User.first(F.id == credential.user_id)

    def revoke_old_credential(self, credential):
        credential.revoked = True
        credential.save()


def query_client(client_id):
    return OAuth2Token.first(F.client_id == client_id)


def save_token(token, request):
    if request.user:
        user_id = request.user.get_user_id()
    else:
        user_id = None
    client = request.client
    item = OAuth2Token(
        client_id=client.client_id,
        user_id=user_id,
        **token
    )
    item.save()


class Revocation(RevocationEndpoint):
    def query_token(self, token_string, token_type_hint):
        if token_type_hint == 'access_token':
            return OAuth2Token.first(F.access_token==token_string)
        elif token_type_hint == 'refresh_token':
            return OAuth2Token.first(F.refresh_token==token_string)
        # without token_type_hint
        return OAuth2Token.first(F.access_token==token_string) or OAuth2Token.first(F.refresh_token == token_string)
    
    def revoke_token(self, token, request):
        now = int(time.time())
        hint = request.form.get('token_type_hint')
        token.access_token_revoked_at = now
        if hint != 'access_token':
            token.refresh_token_revoked_at = now
        token.save()


class Bearer(BearerTokenValidator):
    def authenticate_token(self, token_string):
        return OAuth2Token.first(F.access_token == token_string)


authorization = AuthorizationServer(
    query_client=query_client,
    save_token=save_token,
)


require_oauth = ResourceProtector()


def config_oauth(app):
    authorization.init_app(app)

    # support all grants
    authorization.register_grant(grants.ImplicitGrant)
    authorization.register_grant(grants.ClientCredentialsGrant)
    authorization.register_grant(AuthorizationCodeGrant, [CodeChallenge(required=True)])
    authorization.register_grant(PasswordGrant)
    authorization.register_grant(RefreshTokenGrant)

    # support revocation
    authorization.register_endpoint(Revocation)

    # protect resource
    require_oauth.register_token_validator(Bearer())

    def current_user():
        if 'id' in session:
            uid = session['id']
            return User.query.get(uid)
        return None
    
    def split_by_crlf(s):
        return [v for v in s.splitlines() if v]

    @app.route('/api2/oauth', methods=('GET', 'POST'))
    def home():
        if request.method == 'POST':
            username = request.form.get('username')
            user = User.query.filter_by(username=username).first()
            if not user:
                user = User(username=username)
                user.save()
            session['id'] = str(user.id)
            # if user is not just to log in, but need to head back to the auth page, then go for it
            next_page = request.args.get('next')
            if next_page:
                return redirect(next_page)
            return jsonify({
                'success': True
            })
        
        user = current_user()
        if user:
            clients = OAuth2Client.filter(F.user_id==user.id)
        else:
            clients = []

        return jsonify({
            'user': user.as_dict(),
            'clients': [c.as_dict() for c in clients]
        })

    @app.route('/api2/oauth/logout')
    def logout():
        del session['id']
        return jsonify({
            'success': True
        })


    @app.route('/api2/oauth/create_client', methods=('GET', 'POST'))
    def create_client():
        user = current_user()
        if not user:
            abort(403)

        client_id = gen_salt(24)
        client_id_issued_at = int(time.time())
        client = OAuth2Client(
            client_id=client_id,
            client_id_issued_at=client_id_issued_at,
            user_id=user.id,
        )

        form = request.json()
        client_metadata = {
            "client_name": form["client_name"],
            "client_uri": form["client_uri"],
            "grant_types": split_by_crlf(form["grant_type"]),
            "redirect_uris": split_by_crlf(form["redirect_uri"]),
            "response_types": split_by_crlf(form["response_type"]),
            "scope": form["scope"],
            "token_endpoint_auth_method": form["token_endpoint_auth_method"]
        }
        client.set_client_metadata(client_metadata)

        if form['token_endpoint_auth_method'] == 'none':
            client.client_secret = ''
        else:
            client.client_secret = gen_salt(48)

        client.save()
        return jsonify({'success': True})


    @app.route('/api2/oauth/authorize', methods=['GET', 'POST'])
    def authorize():
        user = current_user()
        if not user:
            abort(403)

        if request.method == 'GET':
            try:
                grant = authorization.get_consent_grant(end_user=user)
            except OAuth2Error as error:
                return error.error
            return jsonify(dict(user=user.as_dict(), grant=grant))
        if not user and 'username' in request.form:
            username = request.form.get('username')
            user = User.first(username=username)
        if request.form['confirm']:
            grant_user = user
        else:
            grant_user = None
        return authorization.create_authorization_response(grant_user=grant_user)


    @app.route('/api2/oauth/token', methods=['POST'])
    def issue_token():
        return authorization.create_token_response()


    @app.route('/api2/oauth/revoke', methods=['POST'])
    def revoke_token():
        return authorization.create_endpoint_response('revocation')

# authentication

Flask and FastAPI



# Notes

Authentication: Refers to proving correct identity <br>
Authorization: Refers to allowing a certain action

### Session vs Token

Session = Cookies
- session stored on server
- session ID (cookie) stored on client
- session ID send with each request

Token 
- token stored on client
- JWT
- token send with each request

### HTTP Basic Authentication

- username:password into the request header
- not recommended

### HTTP Bearer Authentication

- bearer token
- string generated after login

### API Keys

- key generated for first time users
- often as URL query string = unsafe
- better: in Authorization header
- simple but key can be intercepted

### OAuth 2.0

- 1.0 used keyed hash
- access token like API key
- optionally refresh token after expired
- allows limited scope and time validity

Process
- user logs into system 
- system (requester) request authentication from user as token
- user request authentication from authentication server
- token is send to user 
- user sends token to system

Flows
- Authorization code
- Implicit
- Resource owner password
- Client Credentials

### JWT Auth

- JSON Web Tokens
- JWT is library to encode and decode JWTs
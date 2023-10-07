# Google OAuth2 Client

Add Google sign-in into your site

### Steps

Step 1: Create a project in Google Cloud Platform <br>
Step 2: Enable Google OAuth2 API <br>
Step 3: Create OAuth2 Credentials <br>
Step 4: Get OAuth2 Credentials: Client ID and Client Key <br>
Step 5: Add OAuth2 Credentials to Flask App <br>
Step 6: Add OAuth2 Credentials to Google Cloud Platform <br>
Step 7: Test

### Authentication Process
- obtain OAuth 2.0 client credentials from the Google API Console
- client application requests an access token from the Google Authorization Server
- Google Authorization Server authenticates the user and sends an access token to the client application
- client application sends the access token to a Google API it wants to access

### Get OAuth2 Credentials: Client ID and Client Key

https://www.balbooa.com/gridbox-documentation/how-to-get-google-client-id-and-client-secret

go to

https://console.cloud.google.com/apis/credentials

click 'Create Credentials' and select 'OAuth client ID'

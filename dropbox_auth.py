import dropbox

# Replace with your App Key and App Secret
APP_KEY = 'dp9ab1zlci0xxl1'
APP_SECRET = '8cntxwbw59snw9n'

auth_flow = dropbox.DropboxOAuth2FlowNoRedirect(APP_KEY, APP_SECRET)
authorize_url = auth_flow.start()
print("Go to the following URL and authorize the app:", authorize_url)

auth_code = input("Enter the authorization code: ").strip()
auth_result = auth_flow.finish(auth_code)  # Get the result object
access_token = auth_result.access_token    # Access the access token
print("Your access token is:", access_token)

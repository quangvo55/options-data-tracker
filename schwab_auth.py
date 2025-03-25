from schwab.auth import easy_client
import json

# Follow the instructions on the screen to authenticate your client.
c = easy_client(
        api_key='api_key',
        app_secret='app_secret',
        callback_url='https://127.0.0.1:8182',
        token_path='/tmp/token.json')

resp = c.get_option_chain('GME')
with open('data/options_data.json', 'w') as f:
    json.dump(resp.json(), f, indent=2)
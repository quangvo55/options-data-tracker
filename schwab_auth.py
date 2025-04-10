from schwab.auth import easy_client

# Follow the instructions on the screen to authenticate your client.
c = easy_client(
        api_key='key',
        app_secret='secret',
        callback_url='https://127.0.0.1:8182',
        token_path='/tmp/token.json')

def get_options_data(ticker):
        resp = c.get_option_chain(ticker)
        return resp.json()

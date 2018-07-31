def google_authenticate(timeout=30, PORT_NUMBER=8080):

    import httplib2
    from apiclient import discovery
    from oauth2client import client
    from oauth2client.file import Storage

    from IPython.display import HTML

    flow = client.flow_from_clientsecrets('client_secrets.json',
                                          scope="profile email",
                                          redirect_uri='http://localhost:'+str(PORT_NUMBER))

    auth_uri = flow.step1_get_authorize_url()
    print ("CLICK ON THIS LINK TO AUTHENTICATE WITH YOUR GMAIL ACCOUNT")
    print (auth_uri)
    userinfo=None
    auth_code = wait_for_auth(timeout, PORT_NUMBER)
    if auth_code==None:
        print ("No authentication")
        html = HTML("")
    else:
        credentials = flow.step2_exchange(auth_code)

        storage = Storage('/tmp/creds')
        storage.put(credentials)

        http_auth = credentials.authorize(httplib2.Http())

        oauth_service = discovery.build(serviceName='oauth2', version='v2', http=http_auth)
        userinfo = oauth_service.userinfo().get().execute()
        html = HTML("<table><tr><td><img src='"+userinfo["picture"]+"' width=60 height=60/></td><td>"
                 +userinfo["email"]+"<br/>"
                 +userinfo["given_name"]+" "+userinfo["family_name"]+"<br/>"
                 +"google id: "+userinfo["id"]+"<br/>"
                 +"authorization code: "+auth_code
                 +"</td></tr></table>")
    return html, auth_code, userinfo

def wait_for_auth(timeout=30, PORT_NUMBER=8080):

    from http.server import BaseHTTPRequestHandler,HTTPServer
    from urllib.parse import parse_qs, urlparse
    import sys
    global oauth_code
    oauth_code = None
    class myHandler(BaseHTTPRequestHandler):

            #Handler for the GET requests
            def do_GET(self):
                global oauth_code
                dummy="dummy"
                self.send_response(200)
                self.send_header('Content-type','text/html')
                self.end_headers()
                html = '<html><body onload="javascript:settimeout('+"'self.close()'"+',5000);"/>closing</html>'
                html = '<html><body onload="self.close();"/>closing</html>'
                self.wfile.write(html)
                # Send the html message
                q = urlparse(self.path).query
                tokens = q.split("=")
                if len(tokens)==2 and tokens[0]=="code":
                    print ("authentication succeeded")
                    oauth_code = tokens[1]
                else:
                    print (q)
                return
            def log_message(self, format, *args):
                return

    #Create a web server and define the handler to manage the
    #incoming request
    server = HTTPServer(('', PORT_NUMBER), myHandler)
    print ('waiting for authentication ...')
    sys.stdout.flush()
    server.timeout = timeout
    server.handle_request()
    server.server_close()

    return oauth_code
    
import base64
def encode(key, clear):
    enc = []
    clear = base64.urlsafe_b64encode("".join(clear))
    # just comment
    for i in range(len(clear)):
        key_c = key[i % len(key)]
        enc_c = chr((ord(clear[i]) + ord(key_c)) % 256)
        enc.append(enc_c)
    return base64.urlsafe_b64encode("".join(enc))

def decode(key, enc):
    dec = []
    enc = base64.urlsafe_b64decode(enc)
    for i in range(len(enc)):
        key_c = key[i % len(key)]
        dec_c = chr((256 + ord(enc[i]) - ord(key_c)) % 256)
        dec.append(dec_c)
    return base64.urlsafe_b64decode("".join(dec))


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
    print ("Haga click en el siguiente enlace para autenticarse con su cuenta de correo institucional")
    print (auth_uri)
    userinfo=None
    auth_code = wait_for_auth(timeout, PORT_NUMBER)
    if auth_code==None:
        print ("No authentication")
        html = HTML("")
    else:

        credentials = flow.step2_exchange(auth_code)

        http_auth = credentials.authorize(httplib2.Http())

        oauth_service = discovery.build(serviceName='oauth2', version='v2', http=http_auth)
        userinfo = oauth_service.userinfo().get().execute()
        html = HTML("<table><tr><td><img src='"+userinfo["picture"]+"' width=60 height=60/></td><td>"
                 +userinfo["email"]+"<br/>"
                 +userinfo["given_name"]+" "+userinfo["family_name"]+"<br/>"
                 +"google id: "+userinfo["id"]+"<br/>"
                 +"</td></tr></table>")
    return html, auth_code, userinfo

def wait_for_auth(timeout=30, PORT_NUMBER=8080):

    from http.server import BaseHTTPRequestHandler,HTTPServer
    from urllib.parse import urlparse
    import sys
    global oauth_code
    oauth_code = None
    class myHandler(BaseHTTPRequestHandler):

            #Handler for the GET requests
            def do_GET(self):
                global oauth_code
                self.send_response(200)
                self.send_header('Content-type','text/html')
                self.end_headers()
                #html = '<html><body onload="javascript:settimeout('+"'self.close()'"+',5000);"/>closing</html>'
                html = b'<html><body onload="self.close();"/>closing</html>'
                self.wfile.write(html)
                # Send the html message
                q = urlparse(self.path).query
                tokens = q.split("=")
                if len(tokens)==2 and tokens[0]=="code":
                    print ("authentication succeeded")
                    oauth_code = tokens[1]
                    print(self.path)
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
    


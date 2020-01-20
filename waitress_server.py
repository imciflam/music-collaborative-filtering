from waitress import serve
import flask_api_win
serve(flask_api_win.app, host='0.0.0.0', port=8080)

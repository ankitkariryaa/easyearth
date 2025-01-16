from easyearth import init_api

app = init_api().app  # Create the app as a module-level variable

@app.route('/')
def index():
    return "Welcome to Easy Earth API!"

@app.route('/favicon.ico')
def favicon():
    return '', 204


if __name__ == "__main__":
    app.run(port=3781)

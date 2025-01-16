from easyearth import init_api

app = init_api().app  # Create the app as a module-level variable


if __name__ == "__main__":
    app.run(port=3781)

from easyearth import init_api

def main():
    app = init_api()
    app.run(port=3781)

if __name__ == "__main__":
    main()


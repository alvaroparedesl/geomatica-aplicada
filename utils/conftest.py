def pytest_configure(config):
    config.addinivalue_line("markers", "red: pruebas que necesitan internet")

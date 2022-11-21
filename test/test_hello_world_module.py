from src.hello_world_module import hello_world_fn


def test_hello_world_module():
    assert hello_world_fn() == "Hello World!", "The hello world function does not say 'Hello World!'"

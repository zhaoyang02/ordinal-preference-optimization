import socket

def find_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.listen(1)
        port = s.getsockname()[1]
        return port

if __name__ == "__main__":
    free_port = find_port()
    print(f"{free_port}")
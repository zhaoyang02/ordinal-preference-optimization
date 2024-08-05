import socket

def find_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))  # 绑定到0意味着操作系统可以自动分配一个端口
        s.listen(1)
        port = s.getsockname()[1]  # 获取自动分配的端口号
        return port

if __name__ == "__main__":
    free_port = find_port()
    print(f"{free_port}")
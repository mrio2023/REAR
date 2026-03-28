class Tee:
    """将输出同时写入文件和终端"""
    def __init__(self, file, stream):
        self.file = file
        self.stream = stream

    def write(self, message: str) -> None:
        try:
            self.file.write(message)
            self.stream.write(message)
        except Exception as e:
            print(f"写入日志时出错: {e}", file=self.stream)

    def flush(self) -> None:
        self.file.flush()
        self.stream.flush()

    def close(self) -> None:
        if not self.file.closed:
            self.file.close()

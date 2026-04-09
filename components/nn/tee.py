class Tee:
    """Write output to both a file and a terminal stream."""
    def __init__(self, file, stream):
        self.file = file
        self.stream = stream

    def write(self, message: str) -> None:
        try:
            self.file.write(message)
            self.stream.write(message)
        except Exception as e:
            print(f"Error writing log: {e}", file=self.stream)

    def flush(self) -> None:
        self.file.flush()
        self.stream.flush()

    def close(self) -> None:
        if not self.file.closed:
            self.file.close()
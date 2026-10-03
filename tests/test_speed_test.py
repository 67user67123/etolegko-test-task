"""Tests use a local HTTP server and never need internet access."""

from contextlib import contextmanager
from http.client import HTTPException
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import unittest

from speed_test import Measurement, REQUEST_COUNT, measure_speed


PAYLOAD = bytes(range(256)) * 513
TAIL_DELAY = 0.01
SCRIPT = Path(__file__).resolve().parents[1] / "speed_test.py"


@contextmanager
def local_server():
    """Serve delayed downloads while recording concurrent requests."""
    state = {"paths": [], "active": 0, "max_active": 0}
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            with lock:
                state["paths"].append(self.path)
                state["active"] += 1
                state["max_active"] = max(state["max_active"], state["active"])

            active = True
            try:
                if self.path == "/error":
                    self.send_error(503, "Test failure")
                    return

                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream")
                if self.path == "/empty":
                    self.send_header("Content-Length", "0")
                elif self.path == "/length":
                    self.send_header("Content-Length", str(len(PAYLOAD)))
                elif self.path == "/truncated":
                    self.send_header("Content-Length", str(len(PAYLOAD) + 100))
                # HTTP/1.0 closes the connection to delimit the body. Omitting
                # Content-Length ensures the client counts downloaded bytes.
                self.end_headers()
                if self.path == "/empty":
                    return
                self.wfile.write(PAYLOAD[:-1])
                self.wfile.flush()
                time.sleep(0.5 if self.path == "/slow" else TAIL_DELAY)

                # Stop counting before the final byte, so the next request
                # cannot appear concurrent merely because this handler is
                # finishing cleanup after its response has been consumed.
                with lock:
                    state["active"] -= 1
                    active = False
                self.wfile.write(PAYLOAD[-1:])
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                # The timeout test deliberately closes its socket early.
                pass
            finally:
                if active:
                    with lock:
                        state["active"] -= 1

        def log_message(self, format, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(
        target=lambda: server.serve_forever(poll_interval=0.01), daemon=True
    )
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


class MeasurementTests(unittest.TestCase):
    def test_calculations_use_decimal_megabytes_and_eight_bits_per_byte(self):
        result = Measurement(
            request_count=10, total_bytes=5_000_000, total_seconds=2.0
        )

        self.assertAlmostEqual(result.average_seconds, 0.2)
        self.assertAlmostEqual(result.megabytes_per_second, 2.5)
        self.assertAlmostEqual(result.megabits_per_second, 20.0)


class DownloadTests(unittest.TestCase):
    def test_ten_sequential_complete_downloads(self):
        with local_server() as (base_url, state):
            result = measure_speed(base_url + "/payload?sample=1", timeout=2.0)

        self.assertEqual(REQUEST_COUNT, 10)
        self.assertEqual(result.request_count, 10)
        self.assertEqual(state["paths"], ["/payload?sample=1"] * 10)
        self.assertEqual(state["max_active"], 1)
        self.assertEqual(result.total_bytes, len(PAYLOAD) * 10)
        # Every complete download includes the server's delayed final byte.
        self.assertGreaterEqual(result.total_seconds, TAIL_DELAY * 10 * 0.9)
        self.assertAlmostEqual(result.average_seconds, result.total_seconds / 10)
        self.assertGreater(result.megabytes_per_second, 0)

    def test_http_failure_is_reported_and_stops_measurement(self):
        with local_server() as (base_url, state):
            with self.assertRaises((ValueError, OSError, HTTPException)):
                measure_speed(base_url + "/error", timeout=2.0)

        self.assertEqual(state["paths"], ["/error"])

    def test_successful_download_with_content_length(self):
        with local_server() as (base_url, state):
            result = measure_speed(base_url + "/length", timeout=2.0)

        self.assertEqual(result.request_count, 10)
        self.assertEqual(result.total_bytes, len(PAYLOAD) * 10)
        self.assertEqual(state["paths"], ["/length"] * 10)

    def test_truncated_response_is_rejected(self):
        with local_server() as (base_url, state):
            with self.assertRaises((ValueError, OSError, HTTPException)):
                measure_speed(base_url + "/truncated", timeout=2.0)

        self.assertEqual(state["paths"], ["/truncated"])

    def test_timeout_while_reading_body_is_reported(self):
        with local_server() as (base_url, state):
            with self.assertRaises(OSError):
                measure_speed(base_url + "/slow", timeout=0.1)

        self.assertEqual(state["paths"], ["/slow"])

    def test_empty_response_is_rejected(self):
        with local_server() as (base_url, state):
            with self.assertRaises(ValueError):
                measure_speed(base_url + "/empty", timeout=2.0)

        self.assertEqual(state["paths"], ["/empty"])

    def test_invalid_urls_are_rejected(self):
        for url in ("", "example.com/file", "ftp://example.com/file", "file:///tmp/a", "https://"):
            with self.subTest(url=url):
                with self.assertRaises(ValueError):
                    measure_speed(url)

    def test_invalid_timeouts_are_rejected_before_request(self):
        for timeout in (0, -1, float("nan"), float("inf")):
            with self.subTest(timeout=timeout):
                with self.assertRaises(ValueError):
                    measure_speed("http://127.0.0.1:1/file", timeout=timeout)


class CommandLineTests(unittest.TestCase):
    def run_script(self, *arguments):
        return subprocess.run(
            [sys.executable, str(SCRIPT), *arguments],
            capture_output=True,
            text=True,
            encoding="utf-8",
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
            timeout=10,
            check=False,
        )

    def test_successful_cli_prints_measurement(self):
        with local_server() as (base_url, state):
            completed = self.run_script(base_url + "/payload", "--timeout", "2")

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("Выполнено запросов: 10", completed.stdout)
        self.assertIn(f"{len(PAYLOAD) * 10} байт", completed.stdout)
        self.assertIn("МБ/с", completed.stdout)
        self.assertIn("Мбит/с", completed.stdout)
        self.assertEqual(completed.stderr, "")
        self.assertEqual(state["paths"], ["/payload"] * 10)
        self.assertEqual(state["max_active"], 1)

    def test_http_failure_has_nonzero_exit_and_diagnostic(self):
        with local_server() as (base_url, state):
            completed = self.run_script(base_url + "/error", "--timeout", "2")

        self.assertNotEqual(completed.returncode, 0)
        self.assertTrue(completed.stderr.strip())
        self.assertEqual(completed.stdout, "")
        self.assertEqual(state["paths"], ["/error"])

    def test_invalid_url_has_nonzero_exit(self):
        completed = self.run_script("ftp://example.com/file")

        self.assertNotEqual(completed.returncode, 0)
        self.assertTrue(completed.stderr.strip())
        self.assertEqual(completed.stdout, "")

    def test_unrepresentable_socket_timeout_has_clean_error(self):
        with local_server() as (base_url, state):
            completed = self.run_script(base_url + "/payload", "--timeout", "1e100")

        self.assertEqual(completed.returncode, 1)
        self.assertTrue(completed.stderr.strip())
        self.assertNotIn("Traceback", completed.stderr)
        self.assertEqual(completed.stdout, "")
        self.assertEqual(state["paths"], [])

    def test_missing_url_has_nonzero_exit(self):
        completed = self.run_script()

        self.assertNotEqual(completed.returncode, 0)
        self.assertTrue(completed.stderr.strip())
        self.assertEqual(completed.stdout, "")


if __name__ == "__main__":
    unittest.main()

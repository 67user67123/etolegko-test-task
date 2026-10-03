#!/usr/bin/env python3
"""Measure download speed with ten sequential HTTP requests."""

import argparse
from dataclasses import dataclass
from http.client import HTTPException
import math
import sys
from time import perf_counter
from urllib.error import URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


REQUEST_COUNT = 10
CHUNK_SIZE = 64 * 1024


@dataclass(frozen=True)
class Measurement:
    request_count: int
    total_bytes: int
    total_seconds: float

    @property
    def average_seconds(self) -> float:
        return self.total_seconds / self.request_count

    @property
    def megabytes_per_second(self) -> float:
        return self.total_bytes / self.total_seconds / 1_000_000

    @property
    def megabits_per_second(self) -> float:
        return self.megabytes_per_second * 8


def measure_speed(url: str, timeout: float = 30.0) -> Measurement:
    """Download the complete response ten times; abort on any failed request.

    The timeout limits an individual blocking socket operation, not the total
    duration of a download. Elapsed time includes connection setup and redirects.
    """
    parsed = urlsplit(url)
    if (
        parsed.scheme not in ("http", "https")
        or not parsed.hostname
        or any(char.isspace() for char in url)
    ):
        raise ValueError("Укажите полный HTTP(S) URL без пробелов.")
    # Accessing port also validates its format and range before any request.
    _ = parsed.port
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Тайм-аут должен быть положительным конечным числом.")

    total_bytes = 0
    total_seconds = 0.0
    for _ in range(REQUEST_COUNT):
        request = Request(
            url,
            headers={
                "User-Agent": "etolegko-speed-test/1.0",
                "Accept-Encoding": "identity",
                "Cache-Control": "no-cache, no-store",
                "Pragma": "no-cache",
            },
        )
        downloaded = 0
        started = perf_counter()
        with urlopen(request, timeout=timeout) as response:
            while chunk := response.read(CHUNK_SIZE):
                downloaded += len(chunk)

            # A fixed-size read may return EOF on a truncated response without
            # raising IncompleteRead. Content-Length is only an integrity check;
            # all speed calculations use the bytes actually read above.
            content_length = response.headers.get("Content-Length")
            if (
                not response.headers.get("Transfer-Encoding")
                and content_length is not None
                and downloaded != int(content_length)
            ):
                raise ValueError("Ответ скачан не полностью: неверный Content-Length.")

        elapsed = perf_counter() - started
        if downloaded == 0:
            raise ValueError("Сервер вернул пустой ответ. Выберите непустой файл.")
        total_bytes += downloaded
        total_seconds += elapsed

    return Measurement(REQUEST_COUNT, total_bytes, total_seconds)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Замер скорости: 10 последовательных скачиваний указанного URL."
    )
    parser.add_argument("url", help="HTTP(S) адрес большого файла, например изображения")
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        metavar="SECONDS",
        help="тайм-аут сетевой операции в секундах (по умолчанию: 30)",
    )
    args = parser.parse_args(argv)

    try:
        result = measure_speed(args.url, args.timeout)
    except (ValueError, OverflowError, OSError, URLError, HTTPException) as error:
        print(f"Ошибка замера: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nЗамер прерван.", file=sys.stderr)
        return 130

    print(f"Выполнено запросов: {result.request_count}")
    print(f"Среднее время запроса: {result.average_seconds:.3f} с")
    print(f"Скачано данных: {result.total_bytes} байт ({result.total_bytes / 1_000_000:.3f} МБ)")
    print(f"Суммарное время запросов: {result.total_seconds:.3f} с")
    print(
        f"Скорость: {result.megabytes_per_second:.3f} МБ/с "
        f"({result.megabits_per_second:.3f} Мбит/с)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

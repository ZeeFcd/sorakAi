import asyncio

import pytest

from sorakai.common.config import get_settings


@pytest.fixture(autouse=True)
def clear_settings_cache():
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def run_async(coro):
    return asyncio.run(coro)

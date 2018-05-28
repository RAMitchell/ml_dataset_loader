import sys

import datasets

if sys.version_info[0] >= 3:
    import urllib.request as urllib  # NOLINT
else:
    import urllib2 as urllib  # NOLINT


def url_is_alive(url):
    """
    Checks that a given URL is reachable.
    :param url: A URL
    :rtype: bool
    """
    request = urllib.Request(url)
    request.get_method = lambda: 'HEAD'

    try:
        urllib.urlopen(request)
        return True
    except urllib.HTTPError:
        return False


def test_get_higgs():
    assert url_is_alive(datasets.get_higgs_url)


def test_cover_type():
    n = 10
    X, y = datasets.get_cover_type(n)
    assert X.shape[0] == n
    assert y.shape[0] == n


def test_get_synthetic_regression():
    n = 10
    X, y = datasets.get_synthetic_regression(n)
    assert X.shape[0] == n
    assert y.shape[0] == n


def test_get_year():
    assert url_is_alive(datasets.get_year_url)


def test_get_url():
    assert url_is_alive(datasets.get_url_url)

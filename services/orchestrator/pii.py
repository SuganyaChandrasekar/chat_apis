from functools import lru_cache


class PiiMaskingUnavailable(Exception):
    pass


@lru_cache(maxsize=1)
def _engines():
    from presidio_analyzer import AnalyzerEngine
    from presidio_anonymizer import AnonymizerEngine

    return AnalyzerEngine(), AnonymizerEngine()


def initialize_pii_engines() -> None:
    """Initialize the cached Presidio engines before serving requests."""
    _engines()


def anonymize_message(content: str) -> str:
    try:
        analyzer, anonymizer = _engines()
        findings = analyzer.analyze(text=content, language="en")
        return anonymizer.anonymize(
            text=content, analyzer_results=findings
        ).text
    except Exception:
        raise PiiMaskingUnavailable from None